"""
k_predictor production backtest -- edge report against real 2026 DraftKings
pitcher_strikeouts lines, scored across the 72 dates player_props/2026/ has
on S3 (ROADMAP.md's Near-term backlog (k_predictor) item 6(c)).

Same devig -> P(over line) -> edge computation as edge_report.py's original
2025/7-day version, with one structural difference: 2025's odds came from
The Odds API's historical endpoint (10x credit rate), pre-fetched and saved
locally by fetch_2025_odds.py as flat parquet files. 2026's odds were
already collected by the live daily_odds_fetch Lambda at the normal rate --
they sit in s3://mlbdk/raw_data/odds/player_props/2026/ in the Lambda's raw
nested API-envelope shape (one row per game, `game`/`props` columns), not
the flat pitcher/side/line/price shape edge_report.py's load_odds() expects.
This script reads that raw shape directly from S3 and reuses
fetch_2025_odds.py's flatten_props (identical schema -- get_all_player_props
and get_all_historical_player_props both call the same underlying
props-envelope shape) rather than duplicating that parsing.

Run from src/models/k_predictor/ with:
    python backtest/edge_report_2026.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2) -- no
Odds API credits spent, this only reads already-collected S3 data.
"""
import unicodedata
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))
from models.hit_predictor.utils.odds_math import devig_two_way
from models.hit_predictor.utils.count_distribution import prob_exceeds_line

sys.path.insert(0, os.path.dirname(__file__))
from fetch_2025_odds import flatten_props

BACKTEST_DIR = Path(__file__).parent
BUCKET, REGION = "mlbdk", "us-east-2"
boto_session = boto3.Session(region_name=REGION)

# Same 72 dates as score_2026_test_dates.py's BACKTEST_DATES (duplicated,
# not imported -- importing that module would re-run its entire top-level
# data-load/fit/score pipeline as a side effect, not just grab the constant).
BACKTEST_DATES = [
    "2026-04-30",
    "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
    "2026-05-09", "2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13",
    "2026-05-14", "2026-05-15", "2026-05-16", "2026-05-17", "2026-05-18",
    "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",
    "2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09", "2026-06-10",
    "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14", "2026-06-15",
    "2026-06-16",
    "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05",
    "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10",
    "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15",
    "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19", "2026-07-20",
    "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05",
    "2026-08-06", "2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
    "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14", "2026-08-15",
    "2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31",
]


def normalize_name(name):
    """Strip accents for matching -- same odds-API/player_info name mismatch
    edge_report.py's own normalize_name handles."""
    if pd.isna(name):
        return None
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip()


def load_odds_2026():
    frames = []
    for date in BACKTEST_DATES:
        path = f"s3://{BUCKET}/raw_data/odds/player_props/2026/{date}.parquet"
        try:
            raw = wr.s3.read_parquet(path=path, boto3_session=boto_session)
        except Exception as e:
            print(f"  skipping {date}: {e}")
            continue
        if raw.empty:
            print(f"  {date}: 0 games (empty file, e.g. all-star break)")
            continue
        game_props = raw.to_dict("records")
        df = flatten_props(date, game_props)
        print(f"  {date}: {len(game_props)} games, {len(df)} prop rows")
        frames.append(df)
    odds = pd.concat(frames, ignore_index=True)
    odds["date"] = pd.to_datetime(odds["date"])
    odds["name_norm"] = odds["pitcher_name"].apply(normalize_name)

    over = odds[odds["side"] == "Over"][["date", "event_id", "pitcher_name", "name_norm", "line", "price"]].rename(columns={"price": "over_price"})
    under = odds[odds["side"] == "Under"][["date", "event_id", "name_norm", "line", "price"]].rename(columns={"price": "under_price"})
    merged = over.merge(under, on=["date", "event_id", "name_norm", "line"], how="inner")
    return merged


def main():
    pred_df = pd.read_parquet(BACKTEST_DIR / "pred_df_test2026.parquet")
    pred_df["name_norm"] = pred_df["player_name"].apply(normalize_name)
    pred_df["date"] = pd.to_datetime(pred_df["game_date"])

    print(f"Loading real DK pitcher_strikeouts odds for {len(BACKTEST_DATES)} dates from S3...")
    odds = load_odds_2026()

    matched = pred_df.merge(odds, on=["date", "name_norm"], how="inner")
    unmatched_odds_names = set(odds["name_norm"]) - set(matched["name_norm"])

    print(f"\nMatched {len(matched)} pitcher-starts to real DK lines "
          f"(of {len(pred_df)} scored starts, {odds['name_norm'].nunique()} unique odds pitchers, "
          f"{len(odds)} total odds rows).")
    if unmatched_odds_names:
        print(f"Odds pitchers with no matching scored start ({len(unmatched_odds_names)}) -- "
              f"first 20: {sorted(unmatched_odds_names)[:20]}")

    matched["market_p_over"], matched["market_p_under"] = zip(*matched.apply(
        lambda r: devig_two_way(r["over_price"], r["under_price"]), axis=1
    ))
    matched["model_p_over"] = matched.apply(lambda r: prob_exceeds_line(r["pmf"], r["line"]), axis=1)
    matched["edge"] = matched["model_p_over"] - matched["market_p_over"]
    matched["realized_over"] = (matched["realized_k"] > matched["line"]).astype(int)
    matched["model_favored_over"] = matched["model_p_over"] > 0.5
    matched["market_favored_over"] = matched["market_p_over"] > 0.5
    matched["disagree_direction"] = matched["model_favored_over"] != matched["market_favored_over"]

    print(f"\n{'=' * 72}\nAGGREGATE ({len(matched)} matched pitcher-starts, {matched['date'].nunique()} dates)\n{'=' * 72}")
    print(f"Mean edge (model - devigged market): {matched['edge'].mean():+.4f}")
    print(f"Mean |edge|:                         {matched['edge'].abs().mean():.4f}")
    print(f"Starts where model & market disagree on direction: "
          f"{matched['disagree_direction'].sum()} / {len(matched)} "
          f"({matched['disagree_direction'].mean():.1%})")
    print(f"Realized over-rate on these lines: {matched['realized_over'].mean():.1%} "
          f"(model's mean P(over): {matched['model_p_over'].mean():.1%}, "
          f"market's mean devigged P(over): {matched['market_p_over'].mean():.1%})")

    disagreement = matched[matched["disagree_direction"]]
    if len(disagreement):
        follow_model_correct = (disagreement["model_favored_over"] == disagreement["realized_over"].astype(bool))
        print(f"\nOn the {len(disagreement)} disagreement starts, following the model's favored side "
              f"was correct {follow_model_correct.mean():.1%} of the time "
              f"(50% is a coin flip; break-even against a typical -110 line is ~52.4%).")

    out_cols = [
        "date", "player_name", "gamepk", "line", "over_price", "under_price",
        "model_p_over", "market_p_over", "edge", "realized_k", "realized_over",
    ]
    out_path = BACKTEST_DIR / "edge_report_2026.parquet"
    matched[out_cols].sort_values(["date", "player_name"]).to_parquet(out_path, index=False)
    print(f"\nWrote {len(matched)} rows to {out_path}")


if __name__ == "__main__":
    main()
