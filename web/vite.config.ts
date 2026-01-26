import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-only proxy avoids CORS by serving API under the same origin.
const apiTarget = process.env.VITE_API_TARGET || "http://localhost:8000";
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": apiTarget,
      "/health": apiTarget,
      "/docs": apiTarget,
      "/openapi.json": apiTarget,
      "/redoc": apiTarget
    }
  }
});
