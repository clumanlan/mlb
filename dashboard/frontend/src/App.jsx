// App.jsx — the root component. React renders this into the <div id="root"> in index.html.
//
// WHY THIS STRUCTURE:
// Each section (PipelineStatus, TodaysSlate) is a self-contained component.
// App just arranges them vertically. To add Stage 2 (model predictions), you drop in
// <ModelPredictions /> below <TodaysSlate /> — no other changes needed here.
// This pattern is called "component composition".
import PipelineStatus from './components/PipelineStatus.jsx'
import SeasonCompleteness from './components/SeasonCompleteness.jsx'
import TodaysSlate from './components/TodaysSlate.jsx'
import StartingPitcherPredictions from './components/StartingPitcherPredictions.jsx'

// Format today's date as "Saturday, May 9, 2026" for the header.
// toLocaleDateString() is a built-in JS method — no library needed.
const todayLabel = new Date().toLocaleDateString('en-US', {
  weekday: 'long',
  year: 'numeric',
  month: 'long',
  day: 'numeric',
})

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>MLB pre-game</h1>
        <span className="app-date">{todayLabel}</span>
      </header>

      <main className="app-main">
        <PipelineStatus />
        <SeasonCompleteness />
        <TodaysSlate />
        <StartingPitcherPredictions />

        {/* Stage 4: <BetRecommendations /> goes here when built */}
      </main>
    </div>
  )
}
