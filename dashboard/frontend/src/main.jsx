// main.jsx — the entry point React uses to mount the app into the HTML page.
// It finds the <div id="root"> in index.html and renders <App /> inside it.
// StrictMode is a development helper that highlights potential problems.
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
