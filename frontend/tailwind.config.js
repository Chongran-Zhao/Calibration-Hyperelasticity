/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Ink & Azure — a precision calibration instrument.
        background: "#EFF2F8",
        surface: "#FFFFFF",
        subtle: "#F3F6FC",
        border: "#E3E9F2",
        "border-strong": "#CBD5E6",
        primary: "#2563EB",
        "primary-hover": "#1D4ED8",
        "primary-soft": "#DCE7FF",
        "selection-bg": "#ECF2FE",
        "text-primary": "#0E1B33",
        "text-muted": "#5B6B84",
        "text-disabled": "#98A4B8",
        error: "#DC2626",
        success: "#0F9D6B",
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
        panel: "0 1px 2px rgba(15, 27, 52, 0.05), 0 1px 1px rgba(15, 27, 52, 0.03)",
        card: "0 1px 3px rgba(15, 27, 52, 0.06), 0 6px 18px -8px rgba(15, 27, 52, 0.10)",
        float: "0 8px 30px -6px rgba(15, 27, 52, 0.16)",
        "primary-glow": "0 6px 16px -4px rgba(37, 99, 235, 0.45)",
      },
      borderRadius: {
        xl: "0.875rem",
      },
    },
  },
  plugins: [],
}
