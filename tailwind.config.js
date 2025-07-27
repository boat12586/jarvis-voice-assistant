/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      animation: {
        'gradient-shift': 'gradient-shift 15s ease infinite',
        'sparkle': 'sparkle 2s ease-in-out infinite',
        'bounce-soft': 'bounce-soft 2s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        'gradient-shift': {
          '0%, 100%': { 'background-position': '0% 50%' },
          '50%': { 'background-position': '100% 50%' },
        },
        'sparkle': {
          '0%, 100%': { transform: 'rotate(0deg) scale(1)', opacity: '0.7' },
          '25%': { transform: 'rotate(90deg) scale(1.2)', opacity: '1' },
          '50%': { transform: 'rotate(180deg) scale(0.8)', opacity: '0.5' },
          '75%': { transform: 'rotate(270deg) scale(1.1)', opacity: '0.9' },
        },
        'bounce-soft': {
          '0%, 100%': { transform: 'translateY(0px) scale(1)' },
          '50%': { transform: 'translateY(-5px) scale(1.02)' },
        },
        'glow': {
          'from': { 'box-shadow': '0 0 20px rgba(139, 92, 246, 0.3)' },
          'to': { 'box-shadow': '0 0 30px rgba(139, 92, 246, 0.6)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px) rotate(0deg)' },
          '33%': { transform: 'translateY(-20px) rotate(120deg)' },
          '66%': { transform: 'translateY(10px) rotate(240deg)' },
        },
      },
      fontFamily: {
        'poppins': ['Poppins', 'sans-serif'],
        'inter': ['Inter', 'sans-serif'],
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};