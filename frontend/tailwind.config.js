/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Glass — a native macOS instrument: wallpaper, vibrancy, system blue.
        background: "#E7EDF7",
        surface: "rgba(255,255,255,0.72)",
        subtle: "rgba(240,245,252,0.72)",
        border: "rgba(23,35,61,0.10)",
        "border-strong": "rgba(23,35,61,0.18)",
        primary: "#0A84FF",
        "primary-hover": "#0774E8",
        "primary-soft": "rgba(10,132,255,0.16)",
        "selection-bg": "rgba(10,132,255,0.10)",
        "text-primary": "#1D2433",
        "text-muted": "#5D6B82",
        "text-disabled": "#8B97AB",
        error: "#E5484D",
        success: "#30A46C",
      },
      fontFamily: {
        sans: [
          "SF Pro Text",
          "SF Pro Display",
          "Inter",
          "Helvetica Neue",
          "Segoe UI",
          "Arial",
          "sans-serif",
        ],
        mono: ["SF Mono", "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 3px rgba(23, 35, 61, 0.08)",
        card: "0 2px 10px -4px rgba(23, 35, 61, 0.12)",
        float: "0 18px 50px -14px rgba(23, 35, 61, 0.35)",
        "primary-glow": "0 3px 12px rgba(10, 132, 255, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.35)",
      },
      borderRadius: {
        xl: "0.875rem",
      },
    },
  },
  plugins: [],
}
