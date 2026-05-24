/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "#F5F5F7",
        surface: "#FFFFFF",
        subtle: "#F2F2F7",
        border: "#E5E5EA",
        "border-strong": "#D1D1D6",
        primary: "#007AFF",
        "primary-hover": "#0A84FF",
        "selection-bg": "#EAF3FF",
        "text-primary": "#1C1C1E",
        "text-muted": "#6E6E73",
        "text-disabled": "#8E8E93",
        error: "#FF3B30",
      },
      fontFamily: {
        sans: [
          "SF Pro Text",
          "SF Pro Display",
          "Helvetica Neue",
          "Segoe UI",
          "Arial",
          "sans-serif",
        ],
        mono: ["SF Mono", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 2px rgba(0, 0, 0, 0.04)",
      },
    },
  },
  plugins: [],
}
