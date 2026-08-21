# research/

Point-in-time investigations and literature reviews for `hit_predictor` that don't belong in the living reference docs (`FEATURE_GLOSSARY.md`, `BENCHMARKS.md`) or the standing plan (`ROADMAP.md`, repo root).

**What goes here:** gap analyses, external-research summaries (academic papers, industry writeups), one-off diagnostic write-ups — anything that's a snapshot compiled on a given date, not a doc meant to be kept current.

**What doesn't:** anything that needs to stay accurate as the codebase changes. `FEATURE_GLOSSARY.md` is still the single source of truth for current implementation status — update it directly rather than letting a research doc's status claims drift out of sync.

**Naming convention:** `{topic}.md`, dated inline in the doc body (not the filename) — matches how `FEATURE_GLOSSARY.md`/`BENCHMARKS.md` self-date their "compiled as of" footers.

| Doc | What it covers |
|---|---|
| `feature_glossary_gap_analysis.md` | Audit of `FEATURE_GLOSSARY.md` against current Statcast releases and academic/industry PA-outcome-prediction research — candidate features not yet in the glossary, tiered by build cost. Compiled 2026-08-19. |
