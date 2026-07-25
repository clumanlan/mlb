import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config for the MLB dashboard frontend.
// The proxy setting routes /api/* calls to the FastAPI backend during development,
// so the frontend doesn't need to know the backend's port — it just calls /api/...
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
