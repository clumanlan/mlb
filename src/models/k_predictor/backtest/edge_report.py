"""
k_predictor production backtest -- Epic 4: does v6 show a real edge over
real 2025 DraftKings pitcher_strikeouts lines?

Joins score_2025_test_dates.py's model output (per-start pmf) to
fetch_2025_odds.py's real DK lines, devigs the market side, computes
P(total_K > line) from the model's pmf, and reports the edge
(model_p_over - devigged market_p_over) per day and in aggregate.

7 days is nowhere near enough for a real CLV/profitability verdict (see
BENCHMARKS.md's own framing of CLV as the standard practitioner benchmark,
which needs far more than 196 starts) -- this is a first directional read:
does an edge show up at all, and is it inside or outside the vig band.

Run from src/models/k_predictor/ with:
    python backtest/edge_report.py
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))
from models.hit_predictor.utils.odds_math import devig_two_way, compute_edge
from models.hit_predictor.utils.count_distribution import prob_exceeds_line

BACKTEST_DIR = Path(__file__).parent
DATA_DIR = BACKTEST_DIR / "data"


def normalize_name(name):
    """Strip accents for matching -- the odds API and player_info disagree
    on accented characters (e.g. "Carlos Rodon" vs "Carlos Rodón"). A
    handful of pred_df rows have no player_name (a boxscore name-join gap,
    unrelated to this backtest) -- those simply won't match any real odds
    name, which is the correct behavior."""
    if pd.isna(name):
        return None
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip()


def load_odds():
    files = sorted(DATA_DIR.glob("player_props_2025-*.parquet"))
    odds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    odds["date"] = pd.to_datetime(odds["date"])
    odds["name_norm"] = odds["pitcher_name"].apply(normalize_name)

    # one row per (date, event_id, pitcher, line): Over + Under prices side by side
    over = odds[odds["side"] == "Over"][["date", "event_id", "pitcher_name", "name_norm", "line", "price"]].rename(columns={"price": "over_price"})
    under = odds[odds["side"] == "Under"][["date", "event_id", "name_norm", "line", "price"]].rename(columns={"price": "under_price"})
    merged = over.merge(under, on=["date", "event_id", "name_norm", "line"], how="inner")
    return merged


def main():
    pred_df = pd.read_parquet(BACKTEST_DIR / "pred_df_test2025.parquet")
    pred_df["name_norm"] = pred_df["player_name"].apply(normalize_name)
    pred_df["date"] = pd.to_datetime(pred_df["game_date"])

    odds = load_odds()

    matched = pred_df.merge(odds, on=["date", "name_norm"], how="inner")
    unmatched_odds_names = set(odds["name_norm"]) - set(matched["name_norm"])
    unmatched_pred_starts = len(pred_df) - matched["personId"].nunique()

    print(f"Matched {len(matched)} pitcher-starts to real DK lines "
          f"(of {len(pred_df)} scored starts, {odds['name_norm'].nunique()} unique odds pitchers).")
    if unmatched_odds_names:
        print(f"Odds pitchers with no matching scored start ({len(unmatched_odds_names)}): "
              f"{sorted(unmatched_odds_names)}")

    matched["market_p_over"], matched["market_p_under"] = zip(*matched.apply(
        lambda r: devig_two_way(r["over_price"], r["under_price"]), axis=1
    ))
    matched["model_p_over"] = matched.apply(lambda r: prob_exceeds_line(r["pmf"], r["line"]), axis=1)
    matched["edge"] = matched.apply(lambda r: compute_edge(r["model_p_over"], r["market_p_over"]), axis=1)
    matched["realized_over"] = (matched["realized_k"] > matched["line"]).astype(int)
    matched["model_favored_over"] = matched["model_p_over"] > 0.5
    matched["market_favored_over"] = matched["market_p_over"] > 0.5
    matched["disagree_direction"] = matched["model_favored_over"] != matched["market_favored_over"]

    print(f"\n{'=' * 72}\nPER-DAY EDGE\n{'=' * 72}")
    per_day = matched.groupby(matched["date"].dt.date).agg(
        n=("edge", "size"),
        mean_edge=("edge", "mean"),
        mean_abs_edge=("edge", lambda s: s.abs().mean()),
        n_disagree_direction=("disagree_direction", "sum"),
    )
    print(per_day.to_string())

    print(f"\n{'=' * 72}\nAGGREGATE ({len(matched)} matched pitcher-starts)\n{'=' * 72}")
    print(f"Mean edge (model - devigged market): {matched['edge'].mean():+.4f}")
    print(f"Mean |edge|:                         {matched['edge'].abs().mean():.4f}")
    print(f"Starts where model & market disagree on direction: "
          f"{matched['disagree_direction'].sum()} / {len(matched)} "
          f"({matched['disagree_direction'].mean():.1%})")
    print(f"Realized over-rate on these lines: {matched['realized_over'].mean():.1%} "
          f"(model's mean P(over): {matched['model_p_over'].mean():.1%}, "
          f"market's mean devigged P(over): {matched['market_p_over'].mean():.1%})")

    # a balanced two-sided market's vig alone is typically ~2-5 percentage
    # points of raw-probability distance between the two sides' RAW implied
    # probs before devigging -- edge smaller than that is not clearly
    # distinguishable from vig noise on a sample this small.
    VIG_BAND = 0.03
    inside_vig = (matched["edge"].abs() < VIG_BAND).mean()
    print(f"Starts with |edge| < {VIG_BAND:.0%} (inside a typical vig band, not clearly actionable): "
          f"{inside_vig:.1%}")

    # Sanity check: on starts where model and market disagree on direction,
    # would "follow the model" have actually won more than the 50% a coin
    # flip gets on a -110-ish two-sided line? Only meaningful on the
    # disagreement subset -- agreement rows aren't a real test of the model
    # since following it there is identical to following the market.
    disagreement = matched[matched["disagree_direction"]]
    if len(disagreement):
        follow_model_correct = (disagreement["model_favored_over"] == disagreement["realized_over"].astype(bool))
        print(f"\nOn the {len(disagreement)} disagreement starts, following the model's favored side "
              f"was correct {follow_model_correct.mean():.1%} of the time "
              f"(50% is a coin flip; break-even against a typical -110 line is ~52.4%).")
    print("\nCaveat: 131 matched starts across 7 days is far too small a sample for a real "
          "CLV/profitability verdict (see BENCHMARKS.md) -- this is a first directional read, "
          "not a go/no-go on production betting.")

    out_cols = [
        "date", "player_name", "gamepk", "line", "over_price", "under_price",
        "model_p_over", "market_p_over", "edge", "realized_k", "realized_over",
    ]
    out_path = BACKTEST_DIR / "edge_report.parquet"
    matched[out_cols].sort_values(["date", "player_name"]).to_parquet(out_path, index=False)
    print(f"\nWrote {len(matched)} rows to {out_path}")


if __name__ == "__main__":
    main()
