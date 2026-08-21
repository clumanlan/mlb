# Feature Glossary Gap Analysis

**Compiled:** 2026-08-19
**Full designed version:** [Hit Predictor Feature Audit artifact](https://claude.ai/code/artifact/8839cdf2-eead-4168-935a-c6ab22c0b234)

Audits `FEATURE_GLOSSARY.md` (20 implemented / 9 partial / 17 known-gap features as of this date) against two external research passes: current Statcast/sabermetric releases not already named in the glossary, and academic/industry work on plate-appearance-level outcome prediction (Sloan Sports papers, SABR research, betting-model writeups). 26 additional candidates found, tiered by what it costs to act on them.

**Read `ROADMAP.md` first.** The standing near-term priority is the per-game aggregation experiment — settling whether `hit_predictor`'s flat AUC (0.51–0.52 across v1–v3) is a grain problem or a feature-set ceiling — *before* adding more per-PA features. This doc is reference for once that resolves in favor of continued per-PA feature work, not an instruction to start building today.

---

## Tier 1 — Build now (no new ingestion)

Derivable from columns already in `schema.py` / `preprocessing.py` — a new aggregation or join against data already pulled, not a new source.

| Feature | Why it's predictive | Where it hooks in |
|---|---|---|
| Count-state splits | wOBA/hit-rate by ball-strike leverage (ahead/even/behind); pitch selection and swing aggression shift sharply by count | extends `_create_batter_plate_discipline_stats` — add balls/strikes bucket to `extra_group_cols` (already used for §5B) |
| 2-strike approach shift | Swing/contact/whiff rate specifically under 2 strikes — sibling of the existing two-strike foul rate | sibling of `_create_batter_two_strike_foul_stats` |
| Pitch-tunneling proxies | Release-point consistency across pitch types + plate-crossing differential; tight tunneling suppresses hit odds on the second pitch | computable now from `release_pos_x/y/z`, `plate_x/z`, `extension` |
| Arsenal entropy | Shannon entropy of a pitcher's pitch-type mix; low-entropy pitchers are measurably easier to sit on | one groupby on `pitch_type`, joins into `_create_pitcher_stuff_command_stats` |
| Effective/perceived velocity | start_speed adjusted for extension + release-point offset | `_create_pitcher_stuff_command_stats`, combine `extension` + `release_pos_x` + `start_speed` |
| Extended log5 matchup | Batter rate × pitcher rate ÷ league rate, generalized to the 7-outcome multinomial (1B/2B/3B/HR/BB/HBP/out) | all 3 ingredients already exist (player tables + `build_league_*` tables, §5B) — join + formula only |
| Batter-vs-archetype history | Bucket pitchers by stuff/arsenal similarity, look up batter history vs. that archetype — more stable than thin-sample BvP | cluster on existing `_create_pitcher_stuff_command_stats` outputs |
| Velocity-decline trend | Fastball velocity dip as a leading fatigue/injury indicator | reuse `interaction_feats.py`'s trend-ratio machinery on `start_speed`/`spin_rate` |
| Getaway-day / day-after-night | Same-day scheduling fatigue, distinct from days-of-rest | `game_context.py`'s `build_datetime_features` already parses `game_datetime` |
| Stat-type-specific park factors | A park can suppress hits while inflating HR — one scalar conflates asymmetric effects | extend `park_factors.py` to separate HR/2B/BB factor columns |
| On-deck lineup protection | Whether pitchers attack the zone differently with a weaker on-deck hitter | `batting_order` + existing season-quality stats |
| Sample-confidence flags | Explicit "how thin is this sample" indicator on every shrunk feature | systematize §5D's `n_pitches`/`pa_total`, `expected_role.py`'s league fallback |
| Spray angle / pull% profile | Pull tendency is a well-documented sticky skill metric (more stable year-over-year than BABIP-adjacent stats) — combined with EV/LA, indicates whether a power profile is shift-exploitable | `hit_coord_x`/`hit_coord_y` already ingested, unused — glossary §4 already flags this as computable-but-not-built |
| Barrel% — proper EV/LA curve | Current implementation (`batter_season_contact_hard_hit_rate`-style single threshold) is a blunt approximation of what the glossary itself calls one of the most predictive, sticky contact-quality metrics — the real MLB definition is a curve (qualifying angle range widens as EV increases), not a fixed cutoff | same inputs already used (`launch_speed`, `launch_angle`) in `_create_batter_in_play_contact_stats`, better formula only |

**Cross-referenced against an initial 4-candidate proposal drafted before this doc was found** (2026-08-19): the two above were real gaps in this analysis — both zero-new-ingestion, both already independently flagged as promising in `FEATURE_GLOSSARY.md` itself, neither captured by the external-literature pass this doc ran. Two other candidates from that proposal are deliberately **not** added here: the `o_swing_rate`/`z_swing_rate` denominator bug is a correctness fix on an existing feature, not a new candidate — out of this doc's stated scope (line 6) and already tracked in `FEATURE_GLOSSARY.md` §2, so duplicating it here would just be scope creep. Batter-level days-since-last-game overlaps the "getaway-day / day-after-night" row below closely enough (team-level rest already captures most individual bench-day variance) that it's noted here rather than added as a near-duplicate row.

## Tier 2 — Needs new ingestion

Real signal, blocked on a data source not currently pulled — mostly extensions of gaps already flagged in glossary §1–2.

| Feature | Why it's predictive | What's blocking it |
|---|---|---|
| Bat-tracking suite | Squared-up rate, blast%, swing path tilt, attack direction, ideal-attack-angle rate, time-to-contact | same missing Statcast bat-tracking source as the already-flagged bat-speed/swing-length/attack-angle gap |
| Catcher effects | Blocking runs saved, game-calling/sequencing tendencies, batter-catcher familiarity | needs catcher identity joined into pbp — same gap already noted for framing |
| Shift-adjusted spray suppression | Teams still shade within shift-ban-era legal limits | fielder positioning data largely proprietary |
| IL return / ramp-up games | Performance dip in first N games back from injury | needs a transactions log, joined by date to boxscore appearances |
| Travel / time-zone fatigue | Cross-country travel and circadian disruption | needs venue lat/long geo lookup joined to schedule |
| Swing/Take & Decision value | Savant's composite swing-decision-quality run values | confirm whether MLB Stats API pull exposes underlying components |

## Tier 3 — Modeling-technique upgrades (architecture, not columns)

| Technique | What it changes | Source |
|---|---|---|
| Player embeddings | batter2vec/pitcher2vec-style latent representations from PA sequences instead of hand-built rate stats — a different paradigm, future experiment branch | Sloan Sports Conference (Alcorn) |
| Bayesian-adjusted TTOP | **Actionable now:** much of the raw TTOP effect disappears once batter/pitcher quality + handedness are controlled for — worth checking `expected_role.py`'s agreement diagnostics against this | arXiv:2210.06724 (Brill) |
| Low-rank batter×pitcher interaction | Nuclear-penalized multinomial regression — alternative to raw log5, benchmark against it once Tier 1's log5 exists | arXiv:1706.10272 |
| Pitch-sequencing motifs | Directed-graph embeddings clustering at-bats into "setup"/"knockout" pitch roles | Sloan Sports Conference (Prasad) |
| Marcel-style shrinkage | 5/4/3 recency-weighted 3-year average + age curve — concrete upgrade to the ad hoc `k=10` shrinkage already in §5E | Tom Tango, "Marcel the Monkey" |
| Market/CLV features | Closing-line movement + implied-probability deltas as model inputs, not just an eval metric | `BENCHMARKS.md`, arXiv:2410.21484 |
| RE24 base-out state value | Markov-chain run-expectancy weight of base/out state at PA start — knowable *at* the PA, not pre-game; distinct scope from `game_context.py`'s pre-game-only features | openWAR, 24-state Markov chain literature |

---

## Suggested starting order (once Tier 1 work is unblocked)

1. Extended log5 matchup — every ingredient already exists, join + formula only.
2. Spray angle/pull% + proper Barrel% curve — same tier as #1: existing columns, formula-only, both independently flagged as high-value in `FEATURE_GLOSSARY.md`.
3. Count-state splits + 2-strike approach shift — same pattern already built twice.
4. Bayesian sanity-check on TTOP — an afternoon, not a build.
5. Pitch-tunneling proxies + arsenal entropy — zero new ingestion, natural home in `_create_pitcher_stuff_command_stats`.
6. Stat-type-specific park factors — small extension of an owned module.
7. Bat-tracking ingestion — highest one-time cost, but unlocks six Tier-1-quality features at once; batch it.

## Sources

- Baseball Savant — [Swing/Take leaderboard](https://baseballsavant.mlb.com/visuals/swing-take), [swing path/attack angle changelog](https://baseballsavant.mlb.com/changelog/2025-05-20-swing-path-attack-angle)
- FanGraphs — [bat-tracking metrics](https://blogs.fangraphs.com/test-driving-statcasts-newest-bat-tracking-metrics/)
- Baseball Prospectus — [pitch tunnels](https://www.baseballprospectus.com/news/article/31030/prospectus-feature-introducing-pitch-tunnels/), [arsenal metrics](https://www.baseballprospectus.com/news/article/96026/introducing-new-arsenal-metrics/), [Singlearity](https://www.baseballprospectus.com/news/article/59993/singlearity-using-a-neural-network-to-predict-the-outcome-of-plate-appearances/)
- Driveline — [effective velocity](https://www.drivelinebaseball.com/2019/05/calling-right-pitch-investigating-effective-velocity-mlb-level/)
- SABR — [matchup probabilities / log5](https://sabr.org/journal/article/matchup-probabilities-in-major-league-baseball/), [Lichtman on TTOP](https://sabr.org/latest/lichtman-the-penalty-for-pitchers-going-through-the-batting-order/)
- Sloan Sports Conference — [batter/pitcher2vec](https://www.sloansportsconference.com/research-papers/batter-pitcher-2vec-statistic-free-talent-modeling-with-neural-player-embeddings)
- arXiv — [1706.10272](https://arxiv.org/pdf/1706.10272) (nuclear-penalized multinomial regression), [2210.06724](https://arxiv.org/pdf/2210.06724) (Bayesian TTOP), [2206.09654](https://arxiv.org/pdf/2206.09654) (LSTM performance forecasting), [1312.7158](https://arxiv.org/pdf/1312.7158) (openWAR), [2410.21484](https://arxiv.org/pdf/2410.21484) (ML in sports betting survey)
- [Empirical Bayes for batting averages](http://varianceexplained.org/r/empirical_bayes_baseball/), [Marcel methodology](https://www.baseball-reference.com/about/marcels.shtml), [Bayesian Marcel](https://www.pymc-labs.com/blog-posts/bayesian-marcel)
- [MLB Tech Blog — xBA + sprint speed](https://technology.mlblogs.com/augmenting-statcast-expected-batting-average-with-sprint-speed-6be7f60770d2)
