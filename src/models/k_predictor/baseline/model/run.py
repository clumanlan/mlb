"""
Baseline k_predictor — strikeout probability per plate appearance.
Run from src/models/k_predictor/ with: python baseline/model/run.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Split strategy (same convention as hit_predictor/n_pa_predictor):
  train = all train_seasons except val_season and test_season
  val   = val_season  (iterate against this during development)
  test  = test_season (locked away — final eval only in a future train.py)

No new feature engineering here — hit_predictor's season_stats.py and
expected_role.py already compute pitcher/batter strikeout_rate by
(pre-game-estimable) role, which is exactly what this target needs. This
script only builds the PA-grain is_strikeout label (k_predictor.processing)
and wires three baselines against the existing rate features:
  1. Naive (most frequent class): predict "no strikeout" for every PA.
  2. Naive (per expected-pitcher-role K rate): predict the train-set K rate
     for that role (sp vs bullpen) — the harder floor, since role alone is
     a real (if coarse) signal.
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
from models.hit_predictor.processing.features import expected_role

import models.k_predictor.processing.pipeline as pipeline

BASE_DIR = Path(__file__).resolve().parent.parent.parent


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
# Same reason as hit_predictor/n_pa_predictor: pitcher_start_depth_stats needs
# a prior season's pbp for the shift, which isn't loaded for 2016.
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

print("\nLoading game info...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/", TRAIN_SEASONS,
)

print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", all_boxscore_seasons,
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)


# ── 3. Build PA-grain DataFrame ───────────────────────────────────────────────
print("\nBuilding PA-grain DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome_strikeout(pbp, batter_boxscore, game_info, schedule)
pa_outcome["game_season"] = pa_outcome["game_date"].dt.year

# ------------------------- EXPECTED (PRE-GAME) PITCHER ROLE ------------------ #
# Realized pitcher_role leaks in-game information (who a manager actually
# left in) — expected_pitcher_role/expected_pitcher_key_id are pre-game-
# knowable estimates, same reasoning as hit_predictor's own experiments.
pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(
    pa_outcome, pitcher_start_depth_stats, league_avg_start_depth
)

# ------------------------- SEASON-LEVEL STRIKEOUT RATES ---------------------- #
# Already computed by hit_predictor — no new feature engineering needed here.
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})

batter_season_stats = season_stats.build_pbp_batter_feats(pbp)[
    ["batter_id", "game_season", "batter_last_season_pa_strikeout_rate"]
]

pa_outcome = pa_outcome.merge(
    pitcher_role_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome = pa_outcome.merge(batter_season_stats, on=["game_season", "batter_id"], how="left")


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    "expected_pitcher_role",
    "pitcher_last_season_pa_strikeout_rate",
    "batter_last_season_pa_strikeout_rate",
    "pitcher_throw_hand",
    "batter_bat_side",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in pa_outcome.columns]

# Naive-by-role floor needs expected_pitcher_role even if it's dropped from
# FEATURE_COLS for other models — it's already in FEATURE_COLS above.
NAIVE_ROLE_COL = "expected_pitcher_role"

model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"Strikeout rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

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

print("Evaluating naive (per expected-pitcher-role K rate)...")
# Falls back to the train-set global K rate wherever expected_pitcher_role is
# missing — same fallback shape as build_expected_start_innings' baseline chain.
role_rate = train_df.groupby(NAIVE_ROLE_COL)[TARGET].mean()
naive_role_pred = X_val[NAIVE_ROLE_COL].map(role_rate).fillna(y_train.mean()) if NAIVE_ROLE_COL in X_val else pd.Series(
    y_train.mean(), index=X_val.index
)
_eval("Naive (per-role K rate)", y_val, naive_role_pred.to_numpy())

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
role_pr_auc = results["Naive (per-role K rate)"]["pr_auc"]

print(f"\n{'='*72}")
print(f"BASELINE RESULTS — {MODEL_NAME}")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} PAs)  |  Test season {TEST_SEASON} locked")
print("Primary: PR-AUC (higher=better)  |  Secondary: ROC-AUC (higher=better)")
print("=" * 72)
print(f"{'Model':<28} {'PR-AUC':>8} {'vs best naive':>14}  {'ROC-AUC':>8}")
print("-" * 72)
best_naive_name, best_naive_pr_auc = max(
    (("Naive (most frequent)", naive_pr_auc), ("Naive (per-role K rate)", role_pr_auc)),
    key=lambda t: t[1],
)
for name, res in results.items():
    delta = f"{res['pr_auc'] - best_naive_pr_auc:+.4f}" if name not in ("Naive (most frequent)", "Naive (per-role K rate)") else "—"
    print(f"{name:<28} {res['pr_auc']:>8.4f} {delta:>14}  {res['roc_auc']:>8.4f}")
print("=" * 72)

print("\nInterpretation:")
candidates = {n: r for n, r in results.items() if n not in ("Naive (most frequent)", "Naive (per-role K rate)")}
beats_floor = {n: r for n, r in candidates.items() if r["pr_auc"] > best_naive_pr_auc}

if not beats_floor:
    print(f"  No model beats the best naive floor ({best_naive_name}, PR-AUC {best_naive_pr_auc:.4f}).")
    print("  Current feature set (expected_pitcher_role, season-level pitcher/batter")
    print("  strikeout_rate, handedness) has no demonstrated signal beyond reading off")
    print("  a coarse pre-game role split — do not carry this feature set into a")
    print("  train.py as-is.")
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
ax.set_title(f"is_strikeout distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_xlabel("is_strikeout")
ax.set_ylabel("PAs")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'target_distribution.png'}")

fig, ax = plt.subplots(figsize=(5, 5))
cm = confusion_matrix(y_val, results["XGBoost"]["pred"])
ConfusionMatrixDisplay(cm, display_labels=["No K", "K"]).plot(ax=ax, cmap="Blues", colorbar=False)
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
    "**Task:** Binary classification (PA grain)  ",
    "**Target:** Will this plate appearance end in a strikeout?  ",
    "**Primary metric:** PR-AUC (higher = better)  ",
    "**Diagnostics:** ROC-AUC, confusion matrix (XGBoost @ 0.5)  ",
    f"**Data:** s3://{BUCKET}  ",
    "",
    "## Split",
    "",
    "| Split | Seasons | Rows | K rate |",
    "|-------|---------|------|--------|",
    f"| Train | {FIT_SEASONS} | {len(train_df):,} | {y_train.mean():.3f} |",
    f"| Val   | {VAL_SEASON} | {len(val_df):,} | {y_val.mean():.3f} |",
    f"| Test  | {TEST_SEASON} | {len(test_df):,} | locked — not evaluated here |",
    "",
    "## Results (evaluated on val)",
    "",
    "Two naive floors are reported: most-frequent-class, and the train-set K",
    "rate conditioned on expected pitcher role (sp vs bullpen). The real bar a",
    "model must clear is the BETTER of the two.",
    "",
    "| Model | PR-AUC | Δ vs best naive | ROC-AUC |",
    "|-------|--------|------------------|---------|",
]
for name, res in results.items():
    delta = f"{res['pr_auc'] - best_naive_pr_auc:+.4f}" if name not in ("Naive (most frequent)", "Naive (per-role K rate)") else "—"
    md_lines.append(f"| {name} | {res['pr_auc']:.4f} | {delta} | {res['roc_auc']:.4f} |")

if not beats_floor:
    interpretation = (
        f"No model beats the best naive floor (**{best_naive_name}**, PR-AUC {best_naive_pr_auc:.4f}). "
        "The current feature set (expected_pitcher_role, season-level pitcher/batter strikeout_rate, "
        "handedness) has no demonstrated signal beyond a coarse pre-game role split — do not carry "
        "this feature set into a real experiment as-is. Consider rolling-window K rate (recent form, "
        "already computed by hit_predictor's rolling_stats.py) and times_through_order next."
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
    "- No new feature engineering — pitcher/batter strikeout_rate and expected_pitcher_role",
    "  already exist in hit_predictor's season_stats.py / expected_role.py.",
    "- Label: is_strikeout (play_result in {'Strikeout', 'Strikeout Double Play'}),",
    "  k_predictor.processing.pipeline.create_pa_outcome_strikeout",
    "",
    "## Plots",
    "",
    "- `plots/baseline-model/target_distribution.png`",
    "- `plots/baseline-model/confusion_matrix.png` — XGBoost @ 0.5 threshold",
    "- `plots/baseline-model/feature_importance.png`",
    "",
    "## Next steps",
    "",
    "- If a model beats the best naive floor: move to a real experiment, add rolling-window",
    "  K rate (recent form) and times_through_order — both already computed by hit_predictor's",
    "  rolling_stats.py, just not wired in here to keep this baseline minimal.",
    "- If not: this feature set isn't ready. Add rolling-window recent-form features before",
    "  concluding there's no signal — season-level rate alone may be too coarse.",
    "- Final evaluation on test season (2025) only once in a real experiment.",
]

with open(BASE_DIR / "baseline_results.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"\nSaved {BASE_DIR / 'baseline_results.md'}")
