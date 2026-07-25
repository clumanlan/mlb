// PipelineStatus.jsx — Section 1 of the dashboard.
// Shows three cards, one per Lambda (MLB fetch, process data, odds fetch).
// Each card shows whether the Lambda ran recently, how long it took,
// and how many games were processed at each step.
import { useState, useEffect } from 'react'

// === WHAT ARE HOOKS? ===
// Hooks are special React functions that let you "hook into" React features
// from a plain function component. The two we use here:
//
// useState(initialValue) — creates a reactive variable. When you call the
//   setter (e.g. setData), React re-renders the component with the new value.
//   Think of it like a variable that React watches.
//
// useEffect(fn, deps) — runs a side effect after React renders.
//   A "side effect" is anything that reaches outside React: fetch calls,
//   timers, DOM manipulation. The empty array [] as the second argument means
//   "run this once, when the component first appears on the page."
//   Without it, the effect would run on every render — infinite fetch loop.

// --- Sub-component: a single pipeline card ---
// Props are how parent components pass data down to children.
// Think of them like function arguments. Destructured here: { card }.
function PipelineCard({ card }) {
  const isStale = card.is_stale
  const hasError = !!card.error

  return (
    <div className={`pipeline-card ${isStale ? 'card-stale' : ''}`}>
      {/* Card header: title on left, status badge on right */}
      <div className="card-header">
        <span className="card-title">{card.title}</span>
        <span className={`badge ${isStale ? 'badge-stale' : 'badge-ok'}`}>
          {isStale ? 'stale' : '✓ ok'}
        </span>
      </div>

      {/* Run metadata: date · duration */}
      <div className="card-meta">
        <span className="mono">{card.run_date ?? '—'}</span>
        {card.duration_seconds != null && (
          <>
            {' · '}
            <span className="mono">{card.duration_seconds.toFixed(1)}s</span>
          </>
        )}
      </div>

      {/* games_processed entries — one line per step.
          Object.entries() converts { schedule: 15, game_info: 15 }
          into [["schedule", 15], ["game_info", 15]] so we can map over it. */}
      {card.games_processed && Object.keys(card.games_processed).length > 0 && (
        <ul className="games-processed">
          {Object.entries(card.games_processed).map(([key, count]) => (
            <li key={key}>
              <span className="gp-key">{key}</span>
              <span className="mono">{count}</span>
            </li>
          ))}
        </ul>
      )}

      {/* Error message — only rendered when error is non-null */}
      {hasError && (
        <div className="card-error">{card.error}</div>
      )}
    </div>
  )
}

// --- Main component ---
export default function PipelineStatus() {
  // These three state variables track the full lifecycle of the fetch:
  // data = null while loading, then the parsed JSON once it arrives.
  // loading = true until we get a response (success or error).
  // error = null unless the fetch throws or the server returns bad status.
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // WHY .then() chains instead of async/await?
    // useEffect's callback can't be async directly (React limitation).
    // .then() chains are the idiomatic alternative inside useEffect.
    fetch('/api/pipeline-status')
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
  }, []) // [] = run once on mount

  // Render different UI depending on the current state.
  // This is called "conditional rendering" — a core React pattern.
  if (loading) {
    return (
      <section className="section">
        <div className="section-header">
          <span className="section-label">Data Pipeline Status</span>
          <span className="stage-pill">stage 1</span>
        </div>
        <div className="section-loading">Loading pipeline status…</div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="section">
        <div className="section-header">
          <span className="section-label">Data Pipeline Status</span>
          <span className="stage-pill">stage 1</span>
        </div>
        <div className="section-error">Failed to load: {error}</div>
      </section>
    )
  }

  const { cards, summary } = data

  return (
    <section className="section">
      <div className="section-header">
        <span className="section-label">Data Pipeline Status</span>
        <span className="stage-pill">stage 1</span>
      </div>

      {/* WHY key={card.title}?
          When React renders a list, it needs a stable unique identifier per item
          so it knows which DOM nodes to add, remove, or update efficiently.
          card.title is stable and unique across the three cards. */}
      <div className="pipeline-cards">
        {cards.map(card => (
          <PipelineCard key={card.title} card={card} />
        ))}
      </div>

      {/* Summary bar below the cards */}
      <div className="pipeline-summary">
        <span className="mono">{summary.ok_count}/{cards.length} ok</span>
        {summary.stale_count > 0 && (
          <span className="summary-stale">
            {' · '}{summary.stale_count} stale
          </span>
        )}
      </div>
    </section>
  )
}
