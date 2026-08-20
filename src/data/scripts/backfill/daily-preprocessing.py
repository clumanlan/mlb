"""
daily-preprocessing.py

Scans S3 for missing *processed* MLB data between a start and end date,
reads the corresponding raw parquet files, runs the processing pipeline,
and writes results to the processed S3 prefix.

Two-stage pipeline:
  Stage 1 — process_*  : cleans a single dataset in isolation (no cross-dataset deps)
                         covers: schedule, game_info, batter_boxscore, pitcher_boxscore
                         note: play-by-play has no process step — raw goes straight to prepare

  Stage 2 — prepare_*  : joins across processed datasets (needs schedule + player_info)
                         covers: prepared_batter_boxscore, prepared_pitcher_boxscore,
                                 prepared_playbyplay (reads raw play-by-play directly)

S3 layout:
  raw inputs:
    raw_data/games/schedule/{year}/{date}.parquet
    raw_data/games/game_info/{year}/{date}.parquet
    raw_data/games/batter_boxscore/{year}/{date}.parquet
    raw_data/games/pitcher_boxscore/{year}/{date}.parquet
    raw_data/playbyplay/{year}/{date}.parquet
    raw_data/player_info/player_info.parquet            <- reference table, not date-partitioned

  processed outputs:
    processed_data/games/schedule/{year}/{date}.parquet
    processed_data/games/game_info/{year}/{date}.parquet
    processed_data/games/batter_boxscore/{year}/{date}.parquet
    processed_data/games/pitcher_boxscore/{year}/{date}.parquet
    processed_data/prepared/batter_boxscore/{year}/{date}.parquet
    processed_data/prepared/pitcher_boxscore/{year}/{date}.parquet
    processed_data/prepared/playbyplay/{year}/{date}.parquet

Usage:
    # Dry run (default) - print missing dates, do nothing
    python daily-preprocessing.py --start 2025-04-01 --end 2025-06-30

    # Run both stages
    python daily-preprocessing.py --start 2025-04-01 --end 2025-06-30 --no-dry-run

    # Run only stage 1 (process_*)
    python daily-preprocessing.py --start 2025-04-01 --end 2025-06-30 --no-dry-run --stage 1

    # Run only stage 2 (prepare_*)
    python daily-preprocessing.py --start 2025-04-01 --end 2025-06-30 --no-dry-run --stage 2

    # Use test prefix
    python daily-preprocessing.py --start 2025-04-01 --end 2025-06-30 --no-dry-run --env test

    # Resolve date range from MLB season API
    python daily-preprocessing.py --season 2024 --no-dry-run
"""

import argparse
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import boto3
import pandas as pd
import statsapi
from botocore.exceptions import ClientError

from data.modules.preprocessing import (
    prepare_batter_boxscore,
    prepare_pitcher_boxscore,
    prepare_playbyplay,
    process_batter_boxscore,
    process_game_info,
    process_pitcher_boxscore,
    process_player_info,
    process_schedule,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AWS_BUCKET = "mlbdk"
AWS_REGION = "us-east-2"

# Raw S3 paths (inputs) — mirrors fetch pipeline output layout
RAW_KEYS = {
    "schedule":         lambda year, date: f"raw_data/games/schedule/{year}/{date}.parquet",
    "game_info":        lambda year, date: f"raw_data/games/game_info/{year}/{date}.parquet",
    "batter_boxscore":  lambda year, date: f"raw_data/games/batter_boxscore/{year}/{date}.parquet",
    "pitcher_boxscore": lambda year, date: f"raw_data/games/pitcher_boxscore/{year}/{date}.parquet",
    "playbyplay":       lambda year, date: f"raw_data/playbyplay/{year}/{date}.parquet",
    "player_info": lambda year, date: f"reference/player_info/player_info.parquet",
    }

# Processed S3 paths (outputs)
# Note: no processed/playbyplay path — play-by-play has no process_* step
PROCESSED_KEYS = {
    "schedule":                  lambda year, date: f"processed_data/games/schedule/{year}/{date}.parquet",
    "game_info":                 lambda year, date: f"processed_data/games/game_info/{year}/{date}.parquet",
    "batter_boxscore":           lambda year, date: f"processed_data/games/batter_boxscore/{year}/{date}.parquet",
    "pitcher_boxscore":          lambda year, date: f"processed_data/games/pitcher_boxscore/{year}/{date}.parquet",
    "prepared_batter_boxscore":  lambda year, date: f"processed_data/prepared/batter_boxscore/{year}/{date}.parquet",
    "prepared_pitcher_boxscore": lambda year, date: f"processed_data/prepared/pitcher_boxscore/{year}/{date}.parquet",
    "prepared_playbyplay":       lambda year, date: f"processed_data/prepared/playbyplay/{year}/{date}.parquet",
}

# Stage 1: datasets with a process_* step. play-by-play intentionally excluded.
STAGE_1_DATASETS = ["schedule", "game_info", "batter_boxscore", "pitcher_boxscore"]

# Stage 2: all three prepare_* outputs
STAGE_2_DATASETS = ["prepared_batter_boxscore", "prepared_pitcher_boxscore", "prepared_playbyplay"]

# Maps stage 1 dataset -> (process function, raw key name)
PROCESS_FN = {
    "schedule":         (process_schedule,        "schedule"),
    "game_info":        (process_game_info,        "game_info"),
    "batter_boxscore":  (process_batter_boxscore,  "batter_boxscore"),
    "pitcher_boxscore": (process_pitcher_boxscore, "pitcher_boxscore"),
}

# Maps stage 2 dataset -> (prepare function, primary input key, primary key source)
# primary_source is "processed" or "raw" — play-by-play reads raw directly,
# batter/pitcher read from their processed outputs.
# schedule is always read from processed and handled separately inside _run_stage2_for_date.
PREPARE_FN = {
    "prepared_batter_boxscore":  (prepare_batter_boxscore,  "batter_boxscore",  "processed"),
    "prepared_pitcher_boxscore": (prepare_pitcher_boxscore, "pitcher_boxscore", "processed"),
    "prepared_playbyplay":       (prepare_playbyplay,        "playbyplay",       "raw"),
}

# Processed dependencies that must exist before each prepare_* can run.
# play-by-play only needs processed schedule — its primary input comes from raw.
PREPARE_PROCESSED_DEPS = {
    "prepared_batter_boxscore":  ["batter_boxscore", "schedule"],
    "prepared_pitcher_boxscore": ["pitcher_boxscore", "schedule"],
    "prepared_playbyplay":       ["schedule"],
}

# Raw dependencies that must exist before each prepare_* can run.
PREPARE_RAW_DEPS = {
    "prepared_batter_boxscore":  [],
    "prepared_pitcher_boxscore": [],
    "prepared_playbyplay":       ["playbyplay"],
}

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
    missing: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)

    @property
    def needs_backfill(self) -> bool:
        return bool(self.missing)


@dataclass
class StageResult:
    date: str
    success: bool
    datasets_run: list[str] = field(default_factory=list)
    datasets_skipped: dict[str, str] = field(default_factory=dict)  # dataset -> reason
    errors: dict[str, str] = field(default_factory=dict)            # dataset -> error message


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _full_key(key: str, s3_prefix: str) -> str:
    return f"{s3_prefix.rstrip('/')}/{key}" if s3_prefix else key


def _s3_key_exists(s3_client, key: str, s3_prefix: str = "") -> bool:
    try:
        s3_client.head_object(Bucket=AWS_BUCKET, Key=_full_key(key, s3_prefix))
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _read_parquet_from_s3(s3_client, key: str, s3_prefix: str = "") -> pd.DataFrame:
    full_key = _full_key(key, s3_prefix)
    try:
        obj = s3_client.get_object(Bucket=AWS_BUCKET, Key=full_key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as e:
        raise RuntimeError(f"Failed to read s3://{AWS_BUCKET}/{full_key}: {e}") from e


def _write_parquet_to_s3(df: pd.DataFrame, key: str, s3_client, s3_prefix: str = "") -> None:
    full_key = _full_key(key, s3_prefix)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3_client.put_object(Bucket=AWS_BUCKET, Key=full_key, Body=buffer.getvalue())


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

def _check_date_in_s3(
    date: str,
    datasets: list[str],
    key_map: dict,
    s3_client,
    s3_prefix: str = "",
) -> DateStatus:
    """Check which processed outputs exist in S3 for a given date."""
    year = datetime.strptime(date, "%Y-%m-%d").year
    status = DateStatus(date=date)
    for dataset in datasets:
        key = key_map[dataset](year, date)
        if _s3_key_exists(s3_client, key, s3_prefix):
            status.present.append(dataset)
        else:
            status.missing.append(dataset)
    return status


def _get_missing_dates(
    start_date: str,
    end_date: str,
    datasets: list[str],
    key_map: dict,
    s3_prefix: str = "",
) -> list[DateStatus]:
    """Walk every calendar date in [start_date, end_date] and return dates with missing outputs."""
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1

    logger.info(
        f"Scanning {total_days} dates ({start_date} -> {end_date}) "
        f"for datasets: {', '.join(datasets)}"
    )

    missing_statuses = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        status = _check_date_in_s3(date_str, datasets, key_map, s3_client, s3_prefix)
        if status.needs_backfill:
            missing_statuses.append(status)
        current += timedelta(days=1)

    logger.info(f"Scan complete: {len(missing_statuses)} / {total_days} dates have missing data.")
    return missing_statuses


# ---------------------------------------------------------------------------
# Stage 1 — process_*
# ---------------------------------------------------------------------------

def _run_stage1_for_date(
    date: str,
    datasets: list[str],
    s3_client,
    s3_prefix: str = "",
) -> StageResult:
    """
    For each requested dataset:
      - Check raw file exists; if not, skip and warn.
      - Read raw parquet, run process_*, write processed parquet.
      - On failure, record error and continue to next dataset.
    """
    year = datetime.strptime(date, "%Y-%m-%d").year
    result = StageResult(date=date, success=True)

    for dataset in datasets:
        process_fn, raw_key_name = PROCESS_FN[dataset]
        raw_key = RAW_KEYS[raw_key_name](year, date)
        processed_key = PROCESSED_KEYS[dataset](year, date)

        if not _s3_key_exists(s3_client, raw_key, s3_prefix):
            reason = f"raw file missing: {raw_key}"
            logger.warning(f"[{date}] Skipping {dataset} — {reason}")
            result.datasets_skipped[dataset] = reason
            continue

        try:
            raw_df = _read_parquet_from_s3(s3_client, raw_key, s3_prefix)
            processed_df = process_fn(raw_df)
            _write_parquet_to_s3(processed_df, processed_key, s3_client, s3_prefix)
            result.datasets_run.append(dataset)
            logger.info(f"[{date}] OK  {dataset} — {len(processed_df)} rows written")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{date}] FAIL  {dataset} — {error_msg}")
            result.errors[dataset] = error_msg
            result.success = False

    return result


# ---------------------------------------------------------------------------
# Stage 2 — prepare_*
# ---------------------------------------------------------------------------

def _run_stage2_for_date(
    date: str,
    datasets: list[str],
    player_info: pd.DataFrame,
    s3_client,
    s3_prefix: str = "",
) -> StageResult:
    """
    For each requested dataset:
      - Check all processed and raw dependencies exist; if any missing, skip and warn.
      - Read primary input (processed or raw depending on dataset) and processed schedule.
      - Run prepare_* with shared player_info, write prepared parquet.
      - On failure, record error and continue to next dataset.

    player_info is passed in (loaded once by the orchestrator).
    schedule_df is loaded once per date and reused across all prepare calls.
    """
    year = datetime.strptime(date, "%Y-%m-%d").year
    result = StageResult(date=date, success=True)

    # Load schedule once per date — all three prepare_* functions need it
    schedule_df = None

    for dataset in datasets:
        prepare_fn, primary_key_name, primary_source = PREPARE_FN[dataset]
        processed_key = PROCESSED_KEYS[dataset](year, date)

        # Check processed dependencies (batter/pitcher boxscore + schedule)
        missing_processed = [
            dep for dep in PREPARE_PROCESSED_DEPS[dataset]
            if not _s3_key_exists(s3_client, PROCESSED_KEYS[dep](year, date), s3_prefix)
        ]

        # Check raw dependencies (play-by-play)
        missing_raw = [
            dep for dep in PREPARE_RAW_DEPS[dataset]
            if not _s3_key_exists(s3_client, RAW_KEYS[dep](year, date), s3_prefix)
        ]

        missing_deps = missing_processed + missing_raw
        if missing_deps:
            reason = f"dependencies missing: {missing_deps}"
            logger.warning(f"[{date}] Skipping {dataset} — {reason}")
            result.datasets_skipped[dataset] = reason
            continue

        try:
            # Load schedule once and reuse
            if schedule_df is None:
                schedule_key = PROCESSED_KEYS["schedule"](year, date)
                schedule_df = _read_parquet_from_s3(s3_client, schedule_key, s3_prefix)

            # Load primary input from the right source
            if primary_source == "processed":
                primary_key = PROCESSED_KEYS[primary_key_name](year, date)
            else:
                primary_key = RAW_KEYS[primary_key_name](year, date)

            primary_df = _read_parquet_from_s3(s3_client, primary_key, s3_prefix)

            prepared_df = prepare_fn(primary_df, player_info, schedule_df)
            _write_parquet_to_s3(prepared_df, processed_key, s3_client, s3_prefix)
            result.datasets_run.append(dataset)
            logger.info(f"[{date}] OK  {dataset} — {len(prepared_df)} rows written")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{date}] FAIL  {dataset} — {error_msg}")
            result.errors[dataset] = error_msg
            result.success = False

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_backfill_processing(
    start_date: str,
    end_date: str,
    stage: str = "all",
    s3_prefix: str = "",
    dry_run: bool = True,
    inter_date_sleep: float = 1.0,
) -> dict[str, list[StageResult]]:
    """
    Main entry point.

    1. Scans S3 for missing processed outputs across the date range.
    2. Reports what's missing (always).
    3. If dry_run=False, processes each missing date sequentially.

    Args:
        start_date:        Inclusive start, 'YYYY-MM-DD'
        end_date:          Inclusive end,   'YYYY-MM-DD'
        stage:             '1', '2', or 'all'
        s3_prefix:         Mirrors the lambda s3_prefix ('test' or '')
        dry_run:           If True, only report — never write to S3
        inter_date_sleep:  Seconds to pause between dates

    Returns:
        Dict with keys 'stage1' and/or 'stage2', each a list of StageResult.
    """
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    all_results: dict[str, list[StageResult]] = {}

    run_stage1 = stage in ("1", "all")
    run_stage2 = stage in ("2", "all")

    # -----------------------------------------------------------------------
    # Stage 1 — process_*
    # -----------------------------------------------------------------------
    if run_stage1:
        logger.info("\n" + "=" * 60)
        logger.info("  STAGE 1: process_*")
        logger.info("=" * 60)

        missing = _get_missing_dates(
            start_date, end_date, STAGE_1_DATASETS, PROCESSED_KEYS, s3_prefix
        )

        if not missing:
            logger.info("All stage 1 outputs present. Nothing to process.")
        else:
            _log_missing_report(missing)

            if dry_run:
                logger.info("DRY RUN — pass --no-dry-run to process.")
            else:
                stage1_results = []
                for i, status in enumerate(missing, start=1):
                    logger.info(
                        f"[{i}/{len(missing)}] Stage 1 — {status.date} "
                        f"(missing: {status.missing})"
                    )
                    result = _run_stage1_for_date(
                        status.date, status.missing, s3_client, s3_prefix
                    )
                    stage1_results.append(result)
                    if i < len(missing):
                        time.sleep(inter_date_sleep)

                all_results["stage1"] = stage1_results
                _log_stage_summary("Stage 1", stage1_results)

    # -----------------------------------------------------------------------
    # Stage 2 — prepare_*
    # -----------------------------------------------------------------------
    if run_stage2:
        logger.info("\n" + "=" * 60)
        logger.info("  STAGE 2: prepare_*")
        logger.info("=" * 60)

        missing = _get_missing_dates(
            start_date, end_date, STAGE_2_DATASETS, PROCESSED_KEYS, s3_prefix
        )

        if not missing:
            logger.info("All stage 2 outputs present. Nothing to prepare.")
        else:
            _log_missing_report(missing)

            if dry_run:
                logger.info("DRY RUN — pass --no-dry-run to process.")
            else:
                # Load player_info once — shared across all dates and all prepare_* calls
                player_info = _load_player_info(s3_client, s3_prefix)
                if player_info is None:
                    logger.error("Cannot run stage 2 — player_info failed to load. Aborting.")
                    return all_results

                stage2_results = []
                for i, status in enumerate(missing, start=1):
                    logger.info(
                        f"[{i}/{len(missing)}] Stage 2 — {status.date} "
                        f"(missing: {status.missing})"
                    )
                    result = _run_stage2_for_date(
                        status.date, status.missing, player_info, s3_client, s3_prefix
                    )
                    stage2_results.append(result)
                    if i < len(missing):
                        time.sleep(inter_date_sleep)

                all_results["stage2"] = stage2_results
                _log_stage_summary("Stage 2", stage2_results)

    return all_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_player_info(s3_client, s3_prefix: str) -> Optional[pd.DataFrame]:
    """
    Load and process the player_info reference table.
    Returns None and logs an error if the raw file is missing or processing fails.
    """
    raw_key = RAW_KEYS["player_info"](None, None)
    if not _s3_key_exists(s3_client, raw_key, s3_prefix):
        logger.error(f"player_info not found at {raw_key} — run the reference fetcher first.")
        return None
    try:
        raw_df = _read_parquet_from_s3(s3_client, raw_key, s3_prefix)
        player_info = process_player_info(raw_df)
        logger.info(f"Loaded player_info — {len(player_info)} players.")
        return player_info
    except Exception as e:
        logger.error(f"Failed to load/process player_info: {e}")
        return None


def _log_missing_report(missing: list[DateStatus]) -> None:
    logger.info(f"  {len(missing)} date(s) with missing outputs:")
    for status in missing:
        logger.info(f"  {status.date}  missing={status.missing}  present={status.present}")
    logger.info("")


def _log_stage_summary(stage_name: str, results: list[StageResult]) -> None:
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    skipped_any = [r for r in results if r.datasets_skipped]

    logger.info(f"\n  {stage_name} complete:")
    logger.info(f"  {len(succeeded)} succeeded   {len(failed)} failed")
    if skipped_any:
        total_skipped = sum(len(r.datasets_skipped) for r in skipped_any)
        logger.info(f"  {total_skipped} dataset(s) skipped across {len(skipped_any)} date(s)")
    if failed:
        logger.info("  Failed dates:")
        for r in failed:
            logger.info(f"    {r.date}  errors={json.dumps(r.errors)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill missing MLB *processed* data in S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--season",
        type=int,
        help="MLB season year, e.g. 2024. Resolves start/end automatically via MLB API.",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",   help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="Which stage to run: 1=process_*, 2=prepare_*, all=both (default: all)",
    )
    parser.add_argument(
        "--env",
        default="",
        choices=["", "test"],
        help="Set to 'test' to use the test S3 prefix.",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        default=True,
        help="Actually process and write data. Without this flag the script only reports.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        dest="inter_date_sleep",
        help="Seconds to pause between dates (default: 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.season:
        season = statsapi.get("season", {"seasonId": args.season, "sportId": 1})["seasons"][0]
        start_date = season["regularSeasonStartDate"]
        end_date = season["regularSeasonEndDate"]
        logger.info(f"Season {args.season}: {start_date} -> {end_date}")
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        raise SystemExit("Must provide either --season or both --start and --end")

    s3_prefix = "test" if args.env == "test" else ""

    run_backfill_processing(
        start_date=start_date,
        end_date=end_date,
        stage=args.stage,
        s3_prefix=s3_prefix,
        dry_run=args.dry_run,
        inter_date_sleep=args.inter_date_sleep,
    )