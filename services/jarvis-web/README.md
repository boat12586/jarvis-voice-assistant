# Jarvis Voice Assistant v2.0 - Web Interface

A futuristic, sci-fi inspired web interface for the Jarvis Voice Assistant, built with Next.js, React, and TypeScript.

## Features

### 🎨 Futuristic UI Design
- **Iron Man HUD inspired** interface with holographic panels
- **Glassmorphism effects** with neon blue/cyan highlights
- **Animated radar scanner** as centerpiece with voice activity detection
- **Floating UI modules** with glowing outlines and grid overlays
- **Dark mode optimized** with cyberpunk aesthetics
- **Responsive design** that works on all devices

### 🔊 Voice Interface
- **Real-time voice activity** visualization with waveforms
- **Voice recording controls** with visual feedback
- **Push-to-talk** and continuous listening modes
- **Volume level indicators** around radar scanner
- **Voice status monitoring** (idle, listening, processing, speaking)

### 💬 Chat Interface
- **Real-time messaging** with WebSocket support
- **Message history** with timestamps and confidence scores
- **Typing indicators** and status updates
- **Quick action buttons** for common commands
- **Auto-scroll** and message management

### 📊 System Monitoring
- **Health status** dashboard with service monitoring
- **Performance metrics** (CPU, memory, network)
- **Active connections** and session tracking
- **Real-time system statistics**
- **Connection status** indicators

### 🌐 Connectivity
- **WebSocket real-time** communication
- **REST API** integration with FastAPI backend
- **Automatic reconnection** with exponential backoff
- **Error handling** and recovery mechanisms
- **Session management** with persistence

## Technology Stack

- **Frontend Framework**: Next.js 14 with App Router
- **UI Library**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom animations
- **Animations**: Framer Motion for smooth transitions
- **Icons**: Lucide React for consistent iconography
- **WebSocket**: Socket.IO for real-time communication
- **HTTP Client**: Axios for API calls
- **State Management**: React hooks with custom connection manager

## Getting Started

### Prerequisites
- Node.js 18+ and npm/yarn
- Jarvis Core Service running on `localhost:8000`

### Installation

1. **Install dependencies**:
   ```bash
   cd services/jarvis-web
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local with your configuration
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**:
   Navigate to `http://localhost:3000`

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm run start
```

## Project Structure

```
src/
├── app/                    # Next.js app router
│   ├── layout.tsx         # Root layout with background effects
│   ├── page.tsx           # Main dashboard page
│   └── globals.css        # Global styles and animations
├── components/            # React components
│   ├── FuturisticRadarScanner.tsx    # Central radar with voice visualization
│   ├── HolographicPanel.tsx          # Glassmorphic panel wrapper
│   ├── VoiceWaveform.tsx             # Voice activity waveform
│   ├── ChatInterface.tsx             # Real-time chat interface
│   └── SystemMonitor.tsx             # System health dashboard
├── hooks/                 # Custom React hooks
│   └── useJarvisConnection.ts        # Main connection manager
├── lib/                   # Utility libraries
│   ├── api.ts            # REST API client
│   └── websocket.ts      # WebSocket manager
└── types/                 # TypeScript definitions
    └── jarvis.ts         # Interface definitions
```

## Component Architecture

### Radar Scanner
- **360° rotating scanner** with customizable speed
- **Multi-ring design** with status-based colors
- **Voice volume indicators** arranged in circle
- **Status icons** (microphone, processing, speaker)
- **Corner brackets** and grid overlays for sci-fi effect

### Holographic Panels
- **Glassmorphism design** with backdrop blur
- **Customizable glow colors** (cyan, blue, green, yellow, red, purple)
- **Multiple variants** (solid, glass, outline)
- **Animated corners** and scanning line effects
- **Grid patterns** and holographic overlays

### Voice Waveform
- **Real-time audio visualization** with 12+ bars
- **Color-coded status** (cyan=active, red=error, etc.)
- **Responsive height** based on voice volume
- **Smooth animations** with staggered delays

### Chat Interface
- **Bubble-style messages** with user/assistant distinction
- **WebSocket integration** for real-time responses
- **Voice controls** integrated with text input
- **Quick action buttons** for common commands
- **Auto-scroll** and message persistence

### System Monitor
- **Service health** status with color indicators
- **Performance metrics** with animated progress bars
- **Connection statistics** and uptime tracking
- **Network activity** visualization

## Customization

### Themes
Modify `tailwind.config.js` to customize colors:
```javascript
colors: {
  'jarvis': {
    // Custom color palette
  }
}
```

### Animations
Add custom animations in `globals.css`:
```css
@keyframes custom-animation {
  /* Your animation keyframes */
}
```

### Components
All components are modular and accept props for customization:
- Colors, sizes, and variants
- Animation speeds and effects
- Content and layout options

## Performance Optimizations

- **Code splitting** with Next.js dynamic imports
- **Image optimization** with Next.js Image component
- **Bundle analysis** and tree shaking
- **CSS purging** with Tailwind
- **WebSocket connection pooling**
- **API response caching**

## Browser Support

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Development Commands

```bash
# Development
npm run dev              # Start dev server
npm run build           # Build for production
npm run start           # Start production server
npm run lint            # Run ESLint
npm run type-check      # TypeScript checking
```

## Environment Variables

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000     # FastAPI backend URL
NEXT_PUBLIC_WS_URL=ws://localhost:8000        # WebSocket URL
NEXT_PUBLIC_APP_NAME=Jarvis Voice Assistant   # App display name
NEXT_PUBLIC_APP_VERSION=2.0.0                 # Version number
```

## Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for all new code
3. Test components with different states
4. Ensure responsive design works on all screen sizes
5. Add animations that enhance UX without being distracting

## License

Part of the Jarvis Voice Assistant v2.0 project.