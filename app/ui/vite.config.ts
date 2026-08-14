import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The studio server prints {"status":"ready","port":N} on stdout, so in real dev the
// port is not known ahead of time. VITE_SERVER_URL overrides; otherwise we proxy
// /api and /media to the conventional 8756 so the app is same-origin either way.
const target = process.env.VITE_SERVER_URL || "http://127.0.0.1:8756";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1430,
    strictPort: true,
    proxy: {
      "/api": { target, changeOrigin: true },
      "/media": { target, changeOrigin: true },
      "/health": { target, changeOrigin: true },
    },
  },
});
