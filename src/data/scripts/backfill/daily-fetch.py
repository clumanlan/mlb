"""
daily-fetch.py

Scans S3 for missing MLB pipeline dates between a start and end date,
then re-runs the full fetch pipeline for each missing date.

Usage:
    # Dry run (default) — print missing dates, do nothing
    python backfill.py --start 2025-04-01 --end 2025-06-30

    # Actually process missing dates
    python backfill.py --start 2025-04-01 --end 2025-06-30 --no-dry-run

    # Use test prefix (mirrors lambda env=test behaviour)
    python backfill.py --start 2025-04-01 --end 2025-06-30 --no-dry-run --env test

    # Only check/backfill specific datasets
    python backfill.py --start 2025-04-01 --end 2025-06-30 --no-dry-run \
        --datasets schedule game_info
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import statsapi

import boto3
from botocore.exceptions import ClientError

from data.modules.fetchers import (
    fetch_batter_pitcher_boxscore,
    fetch_game_info,
    fetch_playbyplay_data,
    fetch_schedule_df,
    send_metrics_to_cloudwatch,
)
from data.modules.reference_fetch  import ReferenceFetcher

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AWS_BUCKET = "mlbdk"
AWS_REGION = "us-east-2"

# All datasets the pipeline writes, keyed by a short name.
# Values are callables: (year, date) -> s3_key  (no bucket, no prefix)
DATASET_KEYS = {
    "schedule":        lambda year, date: f"raw_data/games/schedule/{year}/{date}.parquet",
    "game_info":       lambda year, date: f"raw_data/games/game_info/{year}/{date}.parquet",
    "batter_boxscore": lambda year, date: f"raw_data/games/batter_boxscore/{year}/{date}.parquet",
    "pitcher_boxscore":lambda year, date: f"raw_data/games/pitcher_boxscore/{year}/{date}.parquet",
    "playbyplay":      lambda year, date: f"raw_data/playbyplay/{year}/{date}.parquet",
}

ALL_DATASETS = list(DATASET_KEYS.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DateStatus:
    date: str
    missing: list[str] = field(default_factory=list)   # dataset names absent in S3
    present: list[str] = field(default_factory=list)   # dataset names found in S3

    @property
    def needs_backfill(self) -> bool:
        return bool(self.missing)


@dataclass
class BackfillResult:
    date: str
    success: bool
    datasets_run: list[str]
    errors: dict = field(default_factory=dict)   # dataset -> error count
    skipped_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_key_exists(s3_client, key: str, s3_prefix: str = "") -> bool:
    """Return True if a key exists in the configured bucket."""
    full_key = f"{s3_prefix.rstrip('/')}/{key}" if s3_prefix else key
    try:
        s3_client.head_object(Bucket=AWS_BUCKET, Key=full_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        # Unexpected error — re-raise so the caller knows something is wrong
        raise


def check_date_in_s3(
    date: str,
    datasets: list[str],
    s3_prefix: str = "",
    s3_client=None,
) -> DateStatus:
    """
    Check which of the requested datasets exist in S3 for a given date.
    Returns a DateStatus with .missing / .present populated.
    """
    if s3_client is None:
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    year = datetime.strptime(date, "%Y-%m-%d").year
    status = DateStatus(date=date)

    for dataset in datasets:
        key_fn = DATASET_KEYS[dataset]
        key = key_fn(year, date)
        if _s3_key_exists(s3_client, key, s3_prefix):
            status.present.append(dataset)
        else:
            status.missing.append(dataset)

    return status


def get_missing_dates(
    start_date: str,
    end_date: str,
    datasets: list[str],
    s3_prefix: str = "",
) -> list[DateStatus]:
    """
    Walk every calendar date in [start_date, end_date] and return
    a list of DateStatus objects where at least one dataset is absent.
    """
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1

    logger.info(
        f"Scanning {total_days} dates ({start_date} → {end_date}) "
        f"for datasets: {', '.join(datasets)}"
    )

    missing_statuses: list[DateStatus] = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        status = check_date_in_s3(date_str, datasets, s3_prefix, s3_client)
        if status.needs_backfill:
            missing_statuses.append(status)
        current += timedelta(days=1)

    logger.info(
        f"Scan complete: {len(missing_statuses)} / {total_days} dates have missing data."
    )
    return missing_statuses


# ---------------------------------------------------------------------------
# Single-date pipeline (mirrors lambda_handler logic)
# ---------------------------------------------------------------------------

def _run_pipeline_for_date(
    date: str,
    datasets_to_run: list[str],
    s3_prefix: str = "",
) -> BackfillResult:
    """
    Run whichever pipeline steps are needed for a single date.
    Mirrors lambda_handler but is selective about which datasets to fetch.

    If 'schedule' is in datasets_to_run the schedule is fetched fresh.
    If 'schedule' is NOT in datasets_to_run but downstream datasets are,
    we still fetch the schedule (without saving) to obtain gamepks.
    """
    results = {}
    error_counts: dict[str, int] = {}
    year = datetime.strptime(date, "%Y-%m-%d").year

    # ------------------------------------------------------------------
    # Schedule — always fetch to get gamepks, save only when requested
    # ------------------------------------------------------------------
    save_schedule = "schedule" in datasets_to_run
    schedule_df, schedule_errors = fetch_schedule_df(
        date, save_to_s3=save_schedule, s3_prefix=s3_prefix
    )

    if save_schedule:
        results["schedule"] = {
            "games_found": len(schedule_df),
            "errors": len(schedule_errors),
        }
        error_counts["schedule"] = len(schedule_errors)

    if schedule_df.empty:
        logger.info(f"[{date}] No games found — nothing to backfill downstream.")
        return BackfillResult(
            date=date,
            success=True,
            datasets_run=datasets_to_run,
            errors=error_counts,
            skipped_reason="No games on schedule for this date",
        )

    # gamepks come from the schedule 'game_id' column (consistent with lambda)
    gamepks = schedule_df["game_id"].tolist()
    logger.info(f"[{date}] {len(gamepks)} game(s) found.")

    # ------------------------------------------------------------------
    # Game info
    # ------------------------------------------------------------------
    game_df = None
    if "game_info" in datasets_to_run:
        game_df, game_errors = fetch_game_info(
            gamepks, game_date=date, save_to_s3=True, s3_prefix=s3_prefix
        )
        results["game_info"] = {
            "games_processed": len(game_df["gamePk"].unique()) if not game_df.empty else 0,
            "errors": len(game_errors),
        }
        error_counts["game_info"] = len(game_errors)

    # ------------------------------------------------------------------
    # Batter / pitcher boxscores
    # ------------------------------------------------------------------
    batter_df = pitcher_df = None
    boxscore_datasets = [d for d in ("batter_boxscore", "pitcher_boxscore") if d in datasets_to_run]
    if boxscore_datasets:
        try:
            batter_df, pitcher_df, boxscore_errors = fetch_batter_pitcher_boxscore(
                gamepks, game_date=date, save_to_s3=True, s3_prefix=s3_prefix
            )
            results["boxscore"] = {
                "batters_boxscore_processed": len(batter_df["gamePk"].unique()) if not batter_df.empty else 0,
                "pitchers_boxscore_processed": len(pitcher_df["gamePk"].unique()) if not pitcher_df.empty else 0,
                "errors": len(boxscore_errors),
            }
            error_counts["batter_boxscore"] = len(boxscore_errors)
            error_counts["pitcher_boxscore"] = len(boxscore_errors)
        except KeyError as e:
            logger.error(f"[{date}] KeyError in boxscore fetch: {e}")
            for ds in boxscore_datasets:
                error_counts[ds] = 1
            results["boxscore"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Play-by-play
    # ------------------------------------------------------------------
    playbyplay_df = None
    if "playbyplay" in datasets_to_run:
        playbyplay_df, playbyplay_errors = fetch_playbyplay_data(
            gamepks, game_date=date, save_to_s3=True, s3_prefix=s3_prefix
        )
        results["playbyplay"] = {
            "games_processed": len(playbyplay_df["gamepk"].unique()) if not playbyplay_df.empty else 0,
            "errors": len(playbyplay_errors),
        }
        error_counts["playbyplay"] = len(playbyplay_errors)

    # ------------------------------------------------------------------
    # Player reference upsert (runs whenever boxscore or pbp was fetched)
    # ------------------------------------------------------------------
    if batter_df is not None or pitcher_df is not None or playbyplay_df is not None:
        try:
            person_ids = []
            if batter_df is not None and not batter_df.empty:
                person_ids.extend(batter_df["personId"].dropna().astype(int).tolist())
            if pitcher_df is not None and not pitcher_df.empty:
                person_ids.extend(pitcher_df["personId"].dropna().astype(int).tolist())
            if playbyplay_df is not None and not playbyplay_df.empty:
                person_ids.extend(playbyplay_df["batter_id"].dropna().astype(int).tolist())
                person_ids.extend(playbyplay_df["pitcher_id"].dropna().astype(int).tolist())

            person_ids = list(set(person_ids))
            ref = ReferenceFetcher()
            updated_players = ref.upsert_players(person_ids, year)
            results["player_info"] = {"total_players": len(updated_players)}
        except Exception as e:
            logger.error(f"[{date}] Failed to upsert player info: {e}")
            results["player_info"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # CloudWatch metrics
    # ------------------------------------------------------------------
    try:
        send_metrics_to_cloudwatch(results, date)
    except Exception as e:
        logger.warning(f"[{date}] CloudWatch metrics failed (non-fatal): {e}")

    total_errors = sum(error_counts.values())
    logger.info(
        f"[{date}] Done. datasets={datasets_to_run} "
        f"errors={json.dumps(error_counts)}"
    )

    return BackfillResult(
        date=date,
        success=(total_errors == 0),
        datasets_run=datasets_to_run,
        errors=error_counts,
    )


# ---------------------------------------------------------------------------
# Main backfill orchestrator
# ---------------------------------------------------------------------------

def run_backfill(
    start_date: str,
    end_date: str,
    datasets: list[str] = ALL_DATASETS,
    s3_prefix: str = "",
    dry_run: bool = True,
    inter_date_sleep: float = 2.0,
) -> list[BackfillResult]:
    """
    Main entry point.

    1. Scans S3 for every date in the range.
    2. Reports what's missing (always).
    3. If dry_run=False, processes each missing date sequentially.

    Args:
        start_date:        Inclusive start, 'YYYY-MM-DD'
        end_date:          Inclusive end,   'YYYY-MM-DD'
        datasets:          Subset of ALL_DATASETS to check/backfill
        s3_prefix:         Mirrors the lambda s3_prefix ('test' or '')
        dry_run:           If True, only report — never write to S3
        inter_date_sleep:  Seconds to pause between dates (rate-limit courtesy)

    Returns:
        List of BackfillResult (empty if dry_run=True)
    """
    # ---- 1. Discover missing dates ------------------------------------
    missing_statuses = get_missing_dates(start_date, end_date, datasets, s3_prefix)

    if not missing_statuses:
        logger.info("✅  All dates present in S3. Nothing to backfill.")
        return []

    # ---- 2. Report ----------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  MISSING DATES ({len(missing_statuses)} total)")
    logger.info(f"{'='*60}")
    for status in missing_statuses:
        logger.info(
            f"  {status.date}  missing={status.missing}  present={status.present}"
        )
    logger.info(f"{'='*60}\n")

    if dry_run:
        logger.info("DRY RUN — no data will be fetched or written. "
                    "Pass --no-dry-run to process.")
        return []

    # ---- 3. Process ---------------------------------------------------
    backfill_results: list[BackfillResult] = []

    for i, status in enumerate(missing_statuses, start=1):
        logger.info(
            f"[{i}/{len(missing_statuses)}] Backfilling {status.date} "
            f"(missing: {status.missing})"
        )

        # Re-check right before processing — idempotency guard.
        # Another process may have already written this date since the scan.
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        live_status = check_date_in_s3(status.date, status.missing, s3_prefix, s3_client)
        still_missing = live_status.missing

        if not still_missing:
            logger.info(f"[{status.date}] Already present (written since scan) — skipping.")
            backfill_results.append(
                BackfillResult(
                    date=status.date,
                    success=True,
                    datasets_run=[],
                    skipped_reason="Already present at processing time",
                )
            )
            continue

        result = _run_pipeline_for_date(status.date, still_missing, s3_prefix)
        backfill_results.append(result)

        if i < len(missing_statuses):
            time.sleep(inter_date_sleep)

    # ---- 4. Summary ---------------------------------------------------
    succeeded = [r for r in backfill_results if r.success]
    failed    = [r for r in backfill_results if not r.success]

    logger.info(f"\n{'='*60}")
    logger.info(f"  BACKFILL COMPLETE")
    logger.info(f"  ✅  {len(succeeded)} succeeded   ❌  {len(failed)} failed")
    if failed:
        logger.info("  Failed dates:")
        for r in failed:
            logger.info(f"    {r.date}  errors={r.errors}")
    logger.info(f"{'='*60}\n")

    return backfill_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill missing MLB pipeline dates in S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--season", type=int, help="MLB season year, e.g. 2024. Resolves start/end automatically.")
    parser.add_argument("--start",   help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",  help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        metavar="DATASET",
        help=(
            f"Datasets to check/backfill. Defaults to all: {ALL_DATASETS}. "
            "e.g. --datasets schedule game_info"
        ),
    )
    parser.add_argument(
        "--env",
        default="",
        choices=["", "test"],
        help="Set to 'test' to use the test S3 prefix (mirrors lambda env=test)",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        default=True,
        help="Actually fetch and write data. Without this flag the script only reports.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        dest="inter_date_sleep",
        help="Seconds to pause between dates (default: 2.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # in __main__
    if args.season:
        season = statsapi.get('season', {'seasonId': args.season, 'sportId': 1})['seasons'][0]
        start_date = season['regularSeasonStartDate']
        end_date   = season['regularSeasonEndDate']
        logger.info(f"Season {args.season}: {start_date} → {end_date}")
    elif args.start and args.end:
        start_date = args.start
        end_date   = args.end
    else:
        parser.error("Must provide either --season or both --start and --end")


    s3_prefix = "test" if args.env == "test" else ""

    run_backfill(
        start_date=start_date,
        end_date=end_date,
        datasets=args.datasets,
        s3_prefix=s3_prefix,
        dry_run=args.dry_run,
        inter_date_sleep=args.inter_date_sleep,
    )

    

