// SeasonCompleteness.jsx — Season 2026 data-completeness audit.
// Compares the season's schedule against batter_boxscore/pitcher_boxscore/playbyplay —
// the tables k_predictor's feature pipeline reads from — and flags any game the
// schedule says was played but a downstream table never picked up.
//
// This is a different question from Section 1's "did yesterday's Lambda run ok":
// a Lambda can report success every day and a table can still be missing games from
// weeks ago (a partial write, a since-fixed bug, a backfill gap). This section checks
// the actual data, not the run logs.
import { useState, useEffect } from 'react'

const TABLE_LABELS = {
  batter_boxscore: 'Batter boxscore',
  pitcher_boxscore: 'Pitcher boxscore',
  playbyplay: 'Play-by-play',
}

function TableRow({ tableKey, table }) {
  const isComplete = table.missing_count === 0

  return (
    <li className="games-processed">
      <span className="gp-key">{TABLE_LABELS[tableKey] ?? tableKey}</span>
      <span className={`badge ${isComplete ? 'badge-ok' : 'badge-stale'}`}>
        {isComplete ? '✓ complete' : `${table.missing_count} missing`}
      </span>
    </li>
  )
}

export default function SeasonCompleteness() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/season-completeness')
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
          <span className="section-label">Season Data Completeness</span>
        </div>
        <div className="section-loading">Auditing season data…</div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="section">
        <div className="section-header">
          <span className="section-label">Season Data Completeness</span>
        </div>
        <div className="section-error">Failed to load: {error}</div>
      </section>
    )
  }

  const { year, total_scheduled_games, tables, is_complete } = data

  return (
    <section className="section">
      <div className="section-header">
        <span className="section-label">Season Data Completeness</span>
        <span className={`badge ${is_complete ? 'badge-ok' : 'badge-stale'}`}>
          {is_complete ? '✓ complete' : 'gaps found'}
        </span>
      </div>

      <div className="card-meta">
        <span className="mono">{year} regular season</span>
        {' · '}
        <span className="mono">{total_scheduled_games} scheduled games</span>
      </div>

      <ul className="games-processed">
        {Object.entries(tables).map(([tableKey, table]) => (
          <TableRow key={tableKey} tableKey={tableKey} table={table} />
        ))}
      </ul>

      {/* Missing-game detail — only rendered for tables that actually have gaps. */}
      {Object.entries(tables).some(([, t]) => t.missing_count > 0) && (
        <div className="card-error">
          {Object.entries(tables)
            .filter(([, t]) => t.missing_count > 0)
            .map(([tableKey, t]) => (
              <div key={tableKey}>
                {TABLE_LABELS[tableKey] ?? tableKey}: missing{' '}
                {t.missing_games.map(g => `${g.gamepk} (${g.game_date})`).join(', ')}
              </div>
            ))}
        </div>
      )}
    </section>
  )
}
