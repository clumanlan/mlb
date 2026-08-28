// gameColor.js — one small shared color per game_pk, used to visually pair a game's
// row in Today's Slate with its two rows in Starting Pitcher Predictions.
//
// WHY A SEPARATE FILE? Both TodaysSlate.jsx and StartingPitcherPredictions.jsx need
// the *exact same* color for a given game_pk, and they're independent components
// with no shared state — a plain function both can import is the simplest way to
// keep them in sync without introducing a global store just for this.
//
// Colors pulled from the user's own portfolio-site design tokens (src/styles/global.css
// there) rather than invented from scratch, so the accent set matches their brand.
const PALETTE = [
  '#2E5C57', // petrol-teal
  '#28304F', // vangogh-navy
  '#A9863E', // wheat-gold
  '#5B6B3E', // olive
  '#B66C53', // terracotta
  '#8A97A6', // cloud-grey
]

// game_pk % PALETTE.length is deterministic regardless of which order either
// component happens to render its rows in — no shared index/state needed.
export function getGameColor(gamePk) {
  return PALETTE[gamePk % PALETTE.length]
}
