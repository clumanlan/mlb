"""
Baseline short_outing_predictor — starting-pitcher short-outing probability
(<=4 IP realized, see processing/schema.py for the explicit-opener caveat).
Run from src/models/short_outing_predictor/ with: python baseline/model/run.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Split strategy (same convention as hit_predictor/k_predictor/n_pa_predictor/
bb_predictor):
  train = all train_seasons except val_season and test_season
  val   = val_season  (iterate against this during development)
  test  = test_season (locked away — final eval only in a future train.py)

Different grain from every prior model here: one row per (personId, gamepk)
STARTING-PITCHER START, not per PA or per batter-game. README's mechanism for
this target ("recent workload/pitch-count trend... a planned bullpen day is
the extreme end of the same short-outing spectrum as an early pull") is
already exactly what hit_predictor's season_stats.py/game_context.py build
for a different purpose (n_pa_predictor's opposing-starter-depth feature):
expected_start_innings, a shrinkage blend of the SAME pitcher's own last-
season baseline IP/start and this-season rolling IP/start. Zero new feature
engineering needed — this script only builds the start-grain is_short_outing
label (short_outing_predictor.processing) and wires it against that existing
blend:
  1. Naive (most frequent class): predict "not short" for every start.
  2. Naive (per expected-start-innings bucket rate): predict the train-set
     short-outing rate for starts with the same rounded expected_start_innings
     — the harder floor, since the pre-game innings estimate alone is a real
     (if coarse) signal.
  3. Logistic regression / XGBoost classifier: the actual candidates.
"""
import yaml
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, ConfusionMatrixDisplay, confusion_matrix

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context

import models.short_outing_predictor.processing.pipeline as pipeline

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STARTER_IP_SHRINKAGE_K = 5.0  # same default as game_context.build_expected_start_innings / n_pa_predictor's reuse of it


# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET          = cfg["bucket"]
REGION          = cfg["region"]
TRAIN_SEASONS   = cfg["train_seasons"]
FEATURE_SEASONS = cfg["feature_seasons"]
TARGET          = cfg["target_column"]
DATE_COL        = cfg["date_column"]
TEST_SEASON     = cfg["test_season"]
VAL_SEASON      = cfg["val_season"]
MODEL_NAME      = cfg["model_name"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
# Same reason as every sibling model: build_pitcher_start_ip_stats needs a
# prior season's pbp for the shift, which isn't loaded for 2016.
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


# ── 2. Load data from S3 ─────────────────────────────────────────────────────
def read_parquet_seasons(path_tpl, seasons, chunked=False):
    frames = []
    for season in seasons:
        path = path_tpl.format(bucket=BUCKET, season=season)
        print(f"  {path}")
        if chunked:
            for chunk in wr.s3.read_parquet(path=path, chunked=True, boto3_session=boto_session):
                if "spin_direction" in chunk.columns:
                    chunk["spin_direction"] = chunk["spin_direction"].astype("float64")
                frames.append(chunk)
        else:
            frames.append(wr.s3.read_parquet(path=path, boto3_session=boto_session))
    return pd.concat(frames, ignore_index=True)


print("\nLoading play-by-play...")
pbp = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True,
)

print("\nLoading schedule...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons,
)

print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", all_boxscore_seasons,
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)


# ── 3. Build start-grain DataFrame ────────────────────────────────────────────
print("\nBuilding start-grain DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)

start_outcome = pipeline.create_start_outcome(pitcher_boxscore, pbp)

# ------------------------- STARTER INNINGS ESTIMATE (own pitcher, not opponent) --- #
# Same building blocks n_pa_predictor already wires up for the OPPOSING
# starter's expected innings — reused here for the pitcher's OWN start,
# which is exactly the "recent workload/pitch-count trend" mechanism
# README's sub-problem menu describes for this target.
pitcher_start_ip_last_season = season_stats.build_pitcher_start_ip_stats(pitcher_boxscore, pbp)
league_avg_start_ip = season_stats.build_league_avg_start_ip(pitcher_start_ip_last_season)
pitcher_start_ip_this_season = game_context.build_pitcher_start_ip_this_season(pitcher_boxscore, pbp)
expected_start_innings = game_context.build_expected_start_innings(
    pitcher_start_ip_last_season, pitcher_start_ip_this_season, league_avg_start_ip,
    k=STARTER_IP_SHRINKAGE_K,
)
expected_start_innings["personId"] = expected_start_innings["personId"].astype(str)
expected_start_innings["gamepk"] = expected_start_innings["gamepk"].astype(str)

start_outcome = start_outcome.drop(columns=["game_date", "game_season"]).merge(
    expected_start_innings, on=["personId", "gamepk"], how="left",
)

# ------------------------- HANDEDNESS ----------------------------------------- #
# Constant per pitcher — already computed at PA grain by build_pbp_features'
# own _add_pbp_handedness, just collapsed to one row per pitcher here.
pitcher_hand = (
    pbp[["pitcher_id", "pitcher_throw_hand"]]
    .drop_duplicates(subset=["pitcher_id"])
    .rename(columns={"pitcher_id": "personId"})
)
start_outcome = start_outcome.merge(pitcher_hand, on="personId", how="left")


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    "pitcher_last_season_start_ip_avg_ip_per_start",
    "pitcher_last_season_start_ip_n_starts",
    "pitcher_this_season_start_ip_avg_ip_per_start",
    "pitcher_this_season_start_ip_starts_n",
    "league_last_season_avg_ip_per_start",
    "expected_start_innings",
    "expected_start_innings_weight",
    "pitcher_throw_hand",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in start_outcome.columns]

# Naive-by-expected-innings-bucket floor needs expected_start_innings even
# if it's dropped from FEATURE_COLS for other models — it's already in
# FEATURE_COLS above.
NAIVE_BUCKET_COL = "expected_start_innings"

model_df = start_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"Short-outing rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_val   = val_df[FEATURE_COLS]
y_val   = val_df[TARGET]

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(X_train[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]


def encode(X_tr, X_ev, cat_cols, num_cols):
    X_tr = X_tr.copy()
    X_ev = X_ev.copy()

    if num_cols:
        X_tr[num_cols] = X_tr[num_cols].apply(pd.to_numeric, errors="coerce")
        X_ev[num_cols] = X_ev[num_cols].apply(pd.to_numeric, errors="coerce")
    if cat_cols:
        # .astype(object) alone leaves pandas' nullable-string missing marker
        # (pd.NA) in place, which SimpleImputer can't handle (pd.NA has no
        # unambiguous bool value) — fillna(np.nan) normalizes to the plain
        # float NaN sklearn expects.
        X_tr[cat_cols] = X_tr[cat_cols].astype(object).fillna(np.nan)
        X_ev[cat_cols] = X_ev[cat_cols].astype(object).fillna(np.nan)

    num_imp = SimpleImputer(strategy="median")
    Xtr_num = num_imp.fit_transform(X_tr[num_cols]) if num_cols else np.empty((len(X_tr), 0))
    Xev_num = num_imp.transform(X_ev[num_cols])     if num_cols else np.empty((len(X_ev), 0))

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        Xtr_cat_imp = cat_imp.fit_transform(X_tr[cat_cols])
        Xev_cat_imp = cat_imp.transform(X_ev[cat_cols])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr_cat = enc.fit_transform(Xtr_cat_imp)
        Xev_cat = enc.transform(Xev_cat_imp)
    else:
        Xtr_cat = np.empty((len(X_tr), 0))
        Xev_cat = np.empty((len(X_ev), 0))

    return np.hstack([Xtr_num, Xtr_cat]), np.hstack([Xev_num, Xev_cat])


Xtr, Xval = encode(X_train, X_val, cat_cols, num_cols)


# ── 5. Train models, evaluate on val ─────────────────────────────────────────
results = {}


def _eval(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    results[name] = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "prob": y_prob,
        "pred": y_pred,
    }


print("\nEvaluating naive (most frequent class)...")
naive_global = DummyClassifier(strategy="most_frequent")
naive_global.fit(Xtr, y_train)
_eval("Naive (most frequent)", y_val, naive_global.predict_proba(Xval)[:, 1])

print("Evaluating naive (per expected-start-innings-bucket rate)...")
# Buckets expected_start_innings to the nearest whole inning and looks up
# the train-set short-outing rate for that bucket — falls back to the
# train-set global rate wherever expected_start_innings is missing or the
# bucket wasn't seen in train, same fallback shape as
# build_expected_start_innings' own baseline chain.
train_bucket = train_df[NAIVE_BUCKET_COL].round(0)
bucket_rate = train_df.assign(_bucket=train_bucket).groupby("_bucket")[TARGET].mean()
val_bucket = X_val[NAIVE_BUCKET_COL].round(0)
naive_bucket_pred = val_bucket.map(bucket_rate).fillna(y_train.mean())
_eval("Naive (per-innings-bucket rate)", y_val, naive_bucket_pred.to_numpy())

print("Training logistic regression...")
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xval_sc = scaler.transform(Xval)
lr = LogisticRegression(max_iter=1000)
lr.fit(Xtr_sc, y_train)
_eval("Logistic regression", y_val, lr.predict_proba(Xval_sc)[:, 1])

print("Training XGBoost...")
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss")
xgb_model.fit(Xtr, y_train)
_eval("XGBoost", y_val, xgb_model.predict_proba(Xval)[:, 1])


# ── 6. Print results ──────────────────────────────────────────────────────────
naive_pr_auc = results["Naive (most frequent)"]["pr_auc"]
bucket_pr_auc = results["Naive (per-innings-bucket rate)"]["pr_auc"]
NAIVE_NAMES = ("Naive (most frequent)", "Naive (per-innings-bucket rate)")

print(f"\n{'='*72}")
print(f"BASELINE RESULTS — {MODEL_NAME}")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} starts)  |  Test season {TEST_SEASON} locked")
print("Primary: PR-AUC (higher=better)  |  Secondary: ROC-AUC (higher=better)")
print("=" * 72)
print(f"{'Model':<32} {'PR-AUC':>8} {'vs best naive':>14}  {'ROC-AUC':>8}")
print("-" * 72)
best_naive_name, best_naive_pr_auc = max(
    (("Naive (most frequent)", naive_pr_auc), ("Naive (per-innings-bucket rate)", bucket_pr_auc)),
    key=lambda t: t[1],
)
for name, res in results.items():
    delta = f"{res['pr_auc'] - best_naive_pr_auc:+.4f}" if name not in NAIVE_NAMES else "—"
    print(f"{name:<32} {res['pr_auc']:>8.4f} {delta:>14}  {res['roc_auc']:>8.4f}")
print("=" * 72)

print("\nInterpretation:")
candidates = {n: r for n, r in results.items() if n not in NAIVE_NAMES}
beats_floor = {n: r for n, r in candidates.items() if r["pr_auc"] > best_naive_pr_auc}

if not beats_floor:
    print(f"  No model beats the best naive floor ({best_naive_name}, PR-AUC {best_naive_pr_auc:.4f}).")
    print("  Current feature set (last-season/this-season/league starter IP,")
    print("  expected_start_innings blend, handedness) has no demonstrated signal")
    print("  beyond reading off the pre-game innings estimate alone — do not carry")
    print("  this feature set into a train.py as-is.")
else:
    best_name, best = max(beats_floor.items(), key=lambda t: t[1]["pr_auc"])
    print(f"  {best_name} beats the best naive floor ({best_naive_name}, PR-AUC {best_naive_pr_auc:.4f}) —")
    print(f"  {best_name} PR-AUC {best['pr_auc']:.4f}. Worth carrying forward into a real experiment.")

lr_pr = results["Logistic regression"]["pr_auc"]
xgb_pr = results["XGBoost"]["pr_auc"]
if abs(xgb_pr - lr_pr) / max(lr_pr, 1e-9) < 0.02:
    print("  Linear vs XGBoost: within ~2% — relationship appears largely linear.")
else:
    print("  Linear vs XGBoost: >2% gap — some nonlinear structure XGBoost is picking up.")
print("=" * 72)


# ── 7. Plots ───────────────────────────────────────────────────────────────
PLOT_DIR = BASE_DIR / "plots" / "baseline-model"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4))
train_df[TARGET].value_counts().sort_index().plot(kind="bar", color="steelblue", ax=ax)
ax.set_title(f"is_short_outing distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_xlabel("is_short_outing")
ax.set_ylabel("starts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'target_distribution.png'}")

fig, ax = plt.subplots(figsize=(5, 5))
cm = confusion_matrix(y_val, results["XGBoost"]["pred"])
ConfusionMatrixDisplay(cm, display_labels=["Full start", "Short outing"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"XGBoost confusion matrix — val ({VAL_SEASON})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'confusion_matrix.png'}")

feature_names = num_cols + cat_cols
importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, max(4, len(importances) * 0.35)))
ax.barh(importances.index, importances.values, color="steelblue")
ax.set_title("Feature importance — XGBoost")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'feature_importance.png'}")


# ── 8. Write baseline_results.md ──────────────────────────────────────────────
md_lines = [
    f"# Baseline Results — {MODEL_NAME}",
    "",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
    "**Task:** Binary classification (starting-pitcher-start grain)  ",
    "**Target:** Will this starting pitcher have a short outing (<=4 IP)?  ",
    "**Primary metric:** PR-AUC (higher = better)  ",
    "**Diagnostics:** ROC-AUC, confusion matrix (XGBoost @ 0.5)  ",
    f"**Data:** s3://{BUCKET}  ",
    "",
    "## Split",
    "",
    "| Split | Seasons | Rows | Short-outing rate |",
    "|-------|---------|------|--------------------|",
    f"| Train | {FIT_SEASONS} | {len(train_df):,} | {y_train.mean():.3f} |",
    f"| Val   | {VAL_SEASON} | {len(val_df):,} | {y_val.mean():.3f} |",
    f"| Test  | {TEST_SEASON} | {len(test_df):,} | locked — not evaluated here |",
    "",
    "## Results (evaluated on val)",
    "",
    "Two naive floors are reported: most-frequent-class, and the train-set",
    "short-outing rate conditioned on the pre-game expected_start_innings blend",
    "(rounded to the nearest whole inning). The real bar a model must clear is",
    "the BETTER of the two.",
    "",
    "| Model | PR-AUC | Δ vs best naive | ROC-AUC |",
    "|-------|--------|------------------|---------|",
]
for name, res in results.items():
    delta = f"{res['pr_auc'] - best_naive_pr_auc:+.4f}" if name not in NAIVE_NAMES else "—"
    md_lines.append(f"| {name} | {res['pr_auc']:.4f} | {delta} | {res['roc_auc']:.4f} |")

if not beats_floor:
    interpretation = (
        f"No model beats the best naive floor (**{best_naive_name}**, PR-AUC {best_naive_pr_auc:.4f}). "
        "The current feature set (last-season/this-season/league starter IP-per-start, the "
        "expected_start_innings shrinkage blend, handedness) has no demonstrated signal beyond "
        "reading off the pre-game innings estimate alone — do not carry this feature set into a "
        "real experiment as-is. Consider recent-start-count workload trend (starts since last "
        "IL stint) and opponent platoon-advantage depth next, per README's mechanism description."
    )
else:
    best_name, best = max(beats_floor.items(), key=lambda t: t[1]["pr_auc"])
    interpretation = (
        f"**{best_name}** beats the best naive floor (**{best_naive_name}**, PR-AUC {best_naive_pr_auc:.4f}) "
        f"with PR-AUC {best['pr_auc']:.4f} — worth carrying forward into a real experiment."
    )

md_lines += [
    "",
    "## Interpretation",
    "",
    interpretation,
    "",
    "## Setup",
    "",
    f"- Features: {FEATURE_COLS}",
    "- No new feature engineering — pitcher_last_season_start_ip_*, ",
    "  pitcher_this_season_start_ip_*, league_last_season_avg_ip_per_start, and",
    "  expected_start_innings/_weight already exist in hit_predictor's",
    "  season_stats.py / game_context.py (built for n_pa_predictor's OPPOSING-",
    "  starter feature — reused here for the pitcher's OWN start instead).",
    "- Grain: one row per (personId, gamepk) starting-pitcher start — different",
    "  from every sibling model's PA or batter-game grain. Scoped to REALIZED",
    "  pitcher_role == 'sp' by construction (a bullpen boxscore row isn't a",
    "  'start' at all), not as a population-scoping choice.",
    "- Label: is_short_outing (realized ip <= 4.0),",
    "  short_outing_predictor.processing.pipeline.create_start_outcome. README's",
    "  mechanism also includes an explicit planned-opener flag not yet wired up",
    "  here — see processing/schema.py's SHORT_OUTING_IP_THRESHOLD docstring.",
    "",
    "## Plots",
    "",
    "- `plots/baseline-model/target_distribution.png`",
    "- `plots/baseline-model/confusion_matrix.png` — XGBoost @ 0.5 threshold",
    "- `plots/baseline-model/feature_importance.png`",
    "",
    "## Next steps",
    "",
    "- If a model beats the best naive floor: move to a real experiment, add",
    "  opponent platoon-advantage depth and bullpen rest state/day-after-",
    "  doubleheader flags — both named in README's mechanism but not yet wired",
    "  in here to keep this baseline minimal.",
    "- If not: this feature set isn't ready. The pre-game innings estimate alone",
    "  may be too coarse — recent-start pitch-count trend (not just IP) could",
    "  carry more signal about an imminent short outing.",
    "- Add explicit-opener detection to the label (see schema.py) before calling",
    "  this experiment-ready — currently undercounts planned bullpen days that",
    "  happen to log >4 IP via a long relief follow.",
    "- Final evaluation on test season (2025) only once in a real experiment.",
]

results_path = BASE_DIR / "baseline_results.md"
results_path.write_text("\n".join(md_lines) + "\n")
print(f"\nSaved {results_path}")
