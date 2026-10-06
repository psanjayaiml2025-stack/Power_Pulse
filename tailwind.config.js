/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        slateblue: "#1e293b",
        accent: "#22d3ee",
        accent2: "#a78bfa",
        warn: "#f59e0b",
        danger: "#f87171",
        good: "#34d399",
      },
    },
  },
  plugins: [],
}
