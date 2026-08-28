// StartingPitcherPredictions.jsx — Section 4 of the dashboard.
// One row per starting pitcher (not per game — a game has two starters), showing
// each of the three pitcher-based predictions side by side so they're easy to scan
// and compare across today's whole slate.
//
// SAMPLE DATA: none of batters_faced_predictor, k_predictor, or short_outing_predictor
// have a production inference path yet, so the backend serves fixed placeholder
// numbers attached to today's REAL games (see dashboard/backend/starting_pitcher_predictions.py).
// This section exists to settle the table's layout and the game<->pitcher link now;
// swap in real numbers once a model is wired up — no change needed here, only on the backend.
import { useState, useEffect } from 'react'
import { getGameColor } from '../lib/gameColor.js'

function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

// American odds display convention: positive gets an explicit "+", negative
// already has its own "-" from the number itself.
function formatOdds(odds) {
  return odds > 0 ? `+${odds}` : `${odds}`
}

// Edge is model probability minus devigged market probability, in probability
// points (e.g. 0.089 -> "+8.9 pts"). Positive = model likes the Over more than
// the market prices it.
function formatEdge(edge) {
  const pts = (edge * 100).toFixed(1)
  return `${edge >= 0 ? '+' : ''}${pts} pts`
}

function EdgeCell({ pitcher }) {
  const isPositive = pitcher.strikeouts_edge >= 0
  const tooltip = `model P(over) ${formatPercent(pitcher.model_prob_strikeouts_over)} vs. ` +
    `devigged market P(over) ${formatPercent(pitcher.fair_prob_strikeouts_over)}`

  return (
    <td className="mono" title={tooltip}>
      <span className={`badge ${isPositive ? 'badge-ok' : 'badge-stale'}`}>
        {formatEdge(pitcher.strikeouts_edge)}
      </span>
    </td>
  )
}

function PitcherRow({ pitcher }) {
  return (
    <tr>
      <td className="col-matchup">{pitcher.pitcher_name}</td>
      <td>{pitcher.team}</td>
      <td className="muted">{pitcher.opponent}</td>
      <td className="mono">{pitcher.batters_faced_pred}</td>
      <td className="mono">{pitcher.strikeouts_pred}</td>
      <td className="mono">{formatPercent(pitcher.early_out_probability)}</td>
      <td className="mono">
        O/U {pitcher.strikeout_line} ({formatOdds(pitcher.strikeout_over_odds)}/{formatOdds(pitcher.strikeout_under_odds)})
      </td>
      <EdgeCell pitcher={pitcher} />
    </tr>
  )
}

// Groups the flat pitchers list into one entry per game_pk, preserving each
// game's two rows together — this is what lets a single <tbody> per game act
// as the scroll/highlight target from Today's Slate (see TodaysSlate.jsx's
// scrollToPredictions, which targets #predictions-game-{game_pk}).
function groupByGame(pitchers) {
  const byGame = new Map()
  for (const pitcher of pitchers) {
    if (!byGame.has(pitcher.game_pk)) byGame.set(pitcher.game_pk, [])
    byGame.get(pitcher.game_pk).push(pitcher)
  }
  return [...byGame.entries()]
}

function GameGroup({ gamePk, rows }) {
  return (
    <tbody id={`predictions-game-${gamePk}`} className="predictions-game-group">
      <tr>
        <td colSpan={8} style={{ padding: '2px 12px 0' }}>
          <span className="game-dot" style={{ background: getGameColor(gamePk) }} />
          <span className="muted mono" style={{ fontSize: '11px' }}>{rows[0].team} vs {rows[0].opponent}</span>
        </td>
      </tr>
      {rows.map(pitcher => (
        <PitcherRow key={`${pitcher.game_pk}-${pitcher.team}`} pitcher={pitcher} />
      ))}
    </tbody>
  )
}

export default function StartingPitcherPredictions() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/starting-pitcher-predictions')
      .then(res => {
        if (!res.ok) throw new Error(`Server error: ${res.status}`)
        return res.json()
      })
      .then(json => {
        setData(json)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <section className="section">
        <div className="section-header">
          <span className="section-label">Starting Pitcher Predictions</span>
        </div>
        <div className="section-loading">Loading predictions…</div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="section">
        <div className="section-header">
          <span className="section-label">Starting Pitcher Predictions</span>
        </div>
        <div className="section-error">Failed to load: {error}</div>
      </section>
    )
  }

  const { pitchers, is_sample_data } = data
  const gameGroups = groupByGame(pitchers)

  return (
    <section className="section">
      <div className="section-header">
        <span className="section-label">Starting Pitcher Predictions</span>
        {is_sample_data && <span className="stage-pill">sample data</span>}
      </div>
      <span className="section-caption">predicted, not realized</span>

      <div style={{ overflowX: 'auto' }}>
        <table className="slate-table">
          <thead>
            <tr>
              <th>Pitcher</th>
              <th>Team</th>
              <th>Opponent</th>
              <th>Batters Faced</th>
              <th>Strikeouts</th>
              <th>Early Out %</th>
              <th title="DraftKings pitcher_strikeouts market — sample data, not live">K Line (O/U)</th>
              <th title="Model P(over) minus devigged market P(over) — hover a value for the breakdown">K Edge</th>
            </tr>
          </thead>
          {gameGroups.map(([gamePk, rows]) => (
            <GameGroup key={gamePk} gamePk={gamePk} rows={rows} />
          ))}
        </table>
      </div>
    </section>
  )
}
