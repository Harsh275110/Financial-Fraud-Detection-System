/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'fraud-red': '#FF4444',
        'safe-green': '#44FF44',
        'warning-yellow': '#FFFF44',
      },
    },
  },
  plugins: [],
} 