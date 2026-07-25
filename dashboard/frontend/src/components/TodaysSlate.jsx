// TodaysSlate.jsx — Section 2 of the dashboard.
// Shows today's game table: time, matchup, lineup status, odds coverage,
// and a placeholder prediction column for Stage 2.
//
// WHY IS THIS A SEPARATE COMPONENT FROM PipelineStatus?
// Each section fetches its own data independently. If the pipeline status API
// is slow, the slate still loads. If the slate API fails, the pipeline cards
// still render. Independent components = independent failure modes.
import { useState, useEffect } from 'react'

// === LineupBadge sub-component ===
// Props: status = "PENDING" | "CONFIRMED" | "SKIPPED"
//
// WHY A SUB-COMPONENT?
// The badge rendering logic (pick a CSS class, maybe wrap in <s>) would
// clutter the table row if inlined. Extracting it into a named component
// makes the table row readable and keeps the styling logic in one place.
function LineupBadge({ status }) {
  // Map each status value to a CSS class defined in index.css.
  // This is conditional class assignment — a very common React pattern.
  const classMap = {
    CONFIRMED: 'badge-confirmed',
    PENDING:   'badge-pending',
    SKIPPED:   'badge-skipped',
  }
  const cls = classMap[status] ?? 'badge-pending'

  // SKIPPED games show the text with a strikethrough to signal inactivity.
  // The ?? operator means "use this fallback if the left side is null/undefined".
  return (
    <span className={`badge ${cls}`}>
      {status === 'SKIPPED' ? <s>{status}</s> : status}
    </span>
  )
}

// === GameRow sub-component ===
// Props: game = one item from the API response's games array.
//
// Receives a single game object and renders one <tr>. Keeping this separate
// from the table makes it easy to later add onClick, tooltips, or row styling.
function GameRow({ game }) {
  return (
    <tr>
      {/* Time in Central — monospace so digits align vertically */}
      <td className="col-time mono">{game.game_time_ct}</td>

      {/* Matchup: "Away Team @ Home Team" */}
      <td className="col-matchup">{game.away_team} @ {game.home_team}</td>

      {/* Lineup badge — color signals confidence in the lineup */}
      <td className="col-lineup">
        <LineupBadge status={game.lineup_status} />
      </td>

      {/* Odds: ✓ if odds data exists for this game, — if not.
          Missing odds on late games is normal — the odds API sometimes
          posts lines within a few hours of game time. */}
      <td className="col-odds">
        {game.has_odds ? '✓' : <span className="muted">—</span>}
      </td>

      {/* Prediction: always — for now. Stage 2 will fill this in. */}
      <td className="col-prediction muted">—</td>
    </tr>
  )
}

// === Main component ===
export default function TodaysSlate() {
  // Same loading/error pattern as PipelineStatus.
  // WHY REPEAT THE PATTERN? Each component owns its own state — they're
  // isolated. If you extract loading logic to a shared hook later, great,
  // but duplication here is fine and clear for learning purposes.
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/today-slate')
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
          <span className="section-label">Today's Slate</span>
          <span className="stage-pill">stage 2 — in progress</span>
        </div>
        <div className="section-loading">Loading today's games…</div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="section">
        <div className="section-header">
          <span className="section-label">Today's Slate</span>
          <span className="stage-pill">stage 2 — in progress</span>
        </div>
        <div className="section-error">Failed to load: {error}</div>
      </section>
    )
  }

  const { games, date, lineup_last_checked } = data

  const lineupCheckedLabel = lineup_last_checked
    ? `lineups checked ${new Date(lineup_last_checked + 'Z').toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago' })} CT`
    : null

  return (
    <section className="section">
      <div className="section-header">
        <span className="section-label">Today's Slate</span>
        <span className="stage-pill">stage 2 — in progress</span>
        {/* Date comes from the API so it matches the data, not the user's clock */}
        <span className="mono muted">{date}</span>
        {lineupCheckedLabel && <span className="mono muted">{lineupCheckedLabel}</span>}
      </div>

      {games.length === 0 ? (
        <div className="section-loading">
          No data yet for {date} — schedule is fetched daily at 10 AM UTC (5–6 AM ET).
        </div>
      ) : (
        <table className="slate-table">
          <thead>
            <tr>
              <th>Time (CT)</th>
              <th>Matchup</th>
              <th>Lineup</th>
              <th>Odds</th>
              <th>Prediction</th>
            </tr>
          </thead>

          {/* WHY key={game.game_pk}?
              game_pk is the MLB's unique game identifier — stable and unique.
              React uses it to efficiently update only the rows that change. */}
          <tbody>
            {games.map(game => (
              <GameRow key={game.game_pk} game={game} />
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
