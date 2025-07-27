'use client';

import React, { useEffect, useState } from 'react';

interface VoiceWaveformProps {
  isActive: boolean;
  volume: number;
  className?: string;
  barCount?: number;
  color?: 'cyan' | 'green' | 'yellow' | 'red';
  height?: 'sm' | 'md' | 'lg';
}

const VoiceWaveform: React.FC<VoiceWaveformProps> = ({
  isActive,
  volume,
  className = '',
  barCount = 12,
  color = 'cyan',
  height = 'md'
}) => {
  const [heights, setHeights] = useState<number[]>(Array(barCount).fill(0));

  const colorClasses = {
    cyan: 'bg-cyan-400',
    green: 'bg-green-400',
    yellow: 'bg-yellow-400',
    red: 'bg-red-400'
  };

  const heightClasses = {
    sm: 'h-8',
    md: 'h-16',
    lg: 'h-24'
  };

  useEffect(() => {
    if (!isActive) {
      setHeights(Array(barCount).fill(2));
      return;
    }

    const interval = setInterval(() => {
      setHeights(prev => 
        prev.map((_, index) => {
          const baseHeight = volume * 100;
          const randomMultiplier = 0.3 + Math.random() * 0.7;
          const positionMultiplier = Math.sin((index / barCount) * Math.PI) * 0.5 + 0.5;
          return Math.max(2, baseHeight * randomMultiplier * positionMultiplier);
        })
      );
    }, 100);

    return () => clearInterval(interval);
  }, [isActive, volume, barCount]);

  return (
    <div className={`flex items-end justify-center space-x-1 ${heightClasses[height]} ${className}`}>
      {heights.map((height, index) => (
        <div
          key={index}
          className={`w-1 ${colorClasses[color]} rounded-t-sm transition-all duration-100 ease-out ${
            isActive ? 'opacity-80' : 'opacity-30'
          }`}
          style={{
            height: `${Math.min(height, 100)}%`,
            animation: isActive ? `voice-wave ${0.8 + Math.random() * 0.4}s ease-in-out infinite alternate` : 'none',
            animationDelay: `${index * 0.05}s`
          }}
        />
      ))}
    </div>
  );
};

export default VoiceWaveform;