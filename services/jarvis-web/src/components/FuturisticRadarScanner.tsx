'use client';

import React, { useEffect, useState } from 'react';
import { VoiceState } from '@/types/jarvis';

interface RadarScannerProps {
  voiceState: VoiceState;
  isActive?: boolean;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

const RadarScanner: React.FC<RadarScannerProps> = ({ 
  voiceState, 
  isActive = false,
  size = 'lg' 
}) => {
  const [scanAngle, setScanAngle] = useState(0);
  const [pulseIntensity, setPulseIntensity] = useState(0);

  const sizeClasses = {
    sm: 'w-32 h-32',
    md: 'w-48 h-48', 
    lg: 'w-64 h-64',
    xl: 'w-80 h-80'
  };

  const ringSize = {
    sm: { outer: 128, middle: 96, inner: 64 },
    md: { outer: 192, middle: 144, inner: 96 },
    lg: { outer: 256, middle: 192, inner: 128 },
    xl: { outer: 320, middle: 240, inner: 160 }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setScanAngle(prev => (prev + 2) % 360);
      if (voiceState.isListening) {
        setPulseIntensity(prev => (prev + 0.1) % 1);
      }
    }, 50);

    return () => clearInterval(interval);
  }, [voiceState.isListening]);

  const getStatusColor = () => {
    switch (voiceState.status) {
      case 'listening': return 'from-cyan-500 to-blue-500';
      case 'processing': return 'from-yellow-500 to-orange-500';
      case 'speaking': return 'from-green-500 to-emerald-500';
      case 'error': return 'from-red-500 to-pink-500';
      default: return 'from-gray-500 to-slate-500';
    }
  };

  const getGlowIntensity = () => {
    if (voiceState.isListening) return 'shadow-2xl shadow-cyan-500/50';
    if (voiceState.isProcessing) return 'shadow-2xl shadow-yellow-500/50';
    return 'shadow-lg shadow-blue-500/30';
  };

  return (
    <div className={`relative ${sizeClasses[size]} mx-auto`}>
      {/* Outer Ring - Static */}
      <div className={`absolute inset-0 rounded-full border-2 border-cyan-500/30 ${getGlowIntensity()}`}>
        {/* Decorative Corner Brackets */}
        <div className="absolute -top-2 -left-2 w-4 h-4 border-l-2 border-t-2 border-cyan-400"></div>
        <div className="absolute -top-2 -right-2 w-4 h-4 border-r-2 border-t-2 border-cyan-400"></div>
        <div className="absolute -bottom-2 -left-2 w-4 h-4 border-l-2 border-b-2 border-cyan-400"></div>
        <div className="absolute -bottom-2 -right-2 w-4 h-4 border-r-2 border-b-2 border-cyan-400"></div>
      </div>

      {/* Middle Ring - Pulsing */}
      <div 
        className={`absolute inset-4 rounded-full border border-cyan-400/50 bg-gradient-to-r ${getStatusColor()} opacity-20`}
        style={{ 
          animation: voiceState.isListening ? 'pulse 2s ease-in-out infinite' : 'none',
          transform: `scale(${1 + pulseIntensity * 0.1})`
        }}
      ></div>

      {/* Inner Core - Voice Responsive */}
      <div className={`absolute inset-8 rounded-full bg-gradient-to-r ${getStatusColor()} opacity-60 flex items-center justify-center`}>
        {/* Central Icon */}
        <div className="w-8 h-8 text-white">
          {voiceState.status === 'listening' && (
            <svg className="w-full h-full animate-pulse" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
            </svg>
          )}
          {voiceState.status === 'processing' && (
            <div className="w-full h-full border-2 border-current border-t-transparent rounded-full animate-spin"></div>
          )}
          {voiceState.status === 'speaking' && (
            <svg className="w-full h-full" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.617.764L4.69 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.69l3.693-3.764zM12 8a1 1 0 012 0v4a1 1 0 11-2 0V8z" clipRule="evenodd" />
              <path d="M16.293 4.293a1 1 0 011.414 0 9.001 9.001 0 010 12.728 1 1 0 11-1.414-1.414 7.001 7.001 0 000-9.9 1 1 0 010-1.414z" />
            </svg>
          )}
          {voiceState.status === 'idle' && (
            <div className="w-2 h-2 bg-current rounded-full animate-pulse"></div>
          )}
        </div>
      </div>

      {/* Scanning Line */}
      <div 
        className="absolute inset-0 rounded-full overflow-hidden"
        style={{
          background: `conic-gradient(from ${scanAngle}deg, transparent 0deg, rgba(6, 182, 212, 0.6) 10deg, transparent 20deg)`
        }}
      ></div>

      {/* Grid Overlay */}
      <div className="absolute inset-0 rounded-full" style={{
        background: `
          radial-gradient(circle at center, transparent 48%, rgba(6, 182, 212, 0.1) 49%, rgba(6, 182, 212, 0.1) 51%, transparent 52%),
          radial-gradient(circle at center, transparent 68%, rgba(6, 182, 212, 0.1) 69%, rgba(6, 182, 212, 0.1) 71%, transparent 72%),
          linear-gradient(0deg, transparent 49%, rgba(6, 182, 212, 0.1) 50%, transparent 51%),
          linear-gradient(90deg, transparent 49%, rgba(6, 182, 212, 0.1) 50%, transparent 51%),
          linear-gradient(45deg, transparent 49%, rgba(6, 182, 212, 0.05) 50%, transparent 51%),
          linear-gradient(-45deg, transparent 49%, rgba(6, 182, 212, 0.05) 50%, transparent 51%)
        `
      }}></div>

      {/* Voice Volume Indicators */}
      {voiceState.isListening && (
        <div className="absolute inset-0 flex items-center justify-center">
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 bg-cyan-400 rounded-full"
              style={{
                height: `${Math.max(4, voiceState.volume * 100 * (0.5 + Math.random() * 0.5))}px`,
                transform: `rotate(${i * 45}deg) translateY(-${ringSize[size].inner / 2 + 20}px)`,
                opacity: voiceState.volume > 0.1 ? 0.8 : 0.3,
                animation: voiceState.volume > 0.1 ? 'voice-wave 0.3s ease-in-out infinite' : 'none'
              }}
            />
          ))}
        </div>
      )}

      {/* Status Text */}
      <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-cyan-400 text-sm font-mono uppercase tracking-wider">
        {voiceState.status}
        {voiceState.error && (
          <div className="text-red-400 text-xs mt-1 normal-case">
            {voiceState.error}
          </div>
        )}
      </div>
    </div>
  );
};

export default RadarScanner;