import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Jarvis Voice Assistant v2.0',
  description: 'Futuristic AI voice assistant with advanced capabilities',
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-900 text-white min-h-screen overflow-hidden`}>
        <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900/20 to-gray-900">
          {/* Matrix Rain Background - This will be replaced by our Matrix component */}
          <div className="absolute inset-0 opacity-30 bg-gradient-to-b from-transparent via-green-900/5 to-transparent" />
          
          {/* Animated Background Grid */}
          <div 
            className="absolute inset-0 opacity-5"
            style={{
              backgroundImage: `
                linear-gradient(rgba(6, 182, 212, 0.5) 1px, transparent 1px),
                linear-gradient(90deg, rgba(6, 182, 212, 0.5) 1px, transparent 1px)
              `,
              backgroundSize: '50px 50px',
              animation: 'grid-shift 20s linear infinite'
            }}
          />
          
          {/* Floating Particles */}
          <div className="absolute inset-0">
            {[...Array(30)].map((_, i) => (
              <div
                key={i}
                className="absolute w-1 h-1 bg-cyan-400 rounded-full opacity-20"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animation: `float ${8 + Math.random() * 15}s ease-in-out infinite`,
                  animationDelay: `${Math.random() * 10}s`
                }}
              />
            ))}
          </div>
        </div>
        
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  )
}