import { defineConfig } from "vite"

// During development the Python API runs separately (uvicorn on :8000);
// proxying /api keeps the frontend same-origin in both dev and production.
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
})
