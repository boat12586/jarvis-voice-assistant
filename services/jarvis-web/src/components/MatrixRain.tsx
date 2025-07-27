'use client';

import React, { useEffect, useRef } from 'react';

interface MatrixRainProps {
  className?: string;
  intensity?: 'low' | 'medium' | 'high';
  color?: 'green' | 'cyan' | 'blue' | 'purple';
  speed?: number;
}

const MatrixRain: React.FC<MatrixRainProps> = ({
  className = '',
  intensity = 'medium',
  color = 'green',
  speed = 1
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const setCanvasSize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    // Matrix characters (Japanese Katakana + Latin + Numbers)
    const characters = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const charactersArray = characters.split('');

    // Column settings based on intensity
    const intensitySettings = {
      low: { columns: 15, speed: 0.5, opacity: 0.6 },
      medium: { columns: 25, speed: 1, opacity: 0.8 },
      high: { columns: 40, speed: 1.5, opacity: 1 }
    };

    const settings = intensitySettings[intensity];
    const columnWidth = canvas.offsetWidth / settings.columns;
    
    // Color settings
    const colorMap = {
      green: { primary: '#00ff00', secondary: '#008f00', shadow: '#004f00' },
      cyan: { primary: '#00ffff', secondary: '#008fff', shadow: '#004f8f' },
      blue: { primary: '#0080ff', secondary: '#0040ff', shadow: '#002080' },
      purple: { primary: '#8000ff', secondary: '#4000ff', shadow: '#200080' }
    };

    const colors = colorMap[color];

    // Initialize columns
    const columns: Array<{
      x: number;
      y: number;
      chars: string[];
      speed: number;
      opacity: number;
      lastUpdate: number;
    }> = [];

    for (let i = 0; i < settings.columns; i++) {
      columns.push({
        x: i * columnWidth,
        y: Math.random() * -canvas.offsetHeight,
        chars: [],
        speed: (Math.random() * 2 + 1) * settings.speed * speed,
        opacity: Math.random() * settings.opacity + 0.2,
        lastUpdate: 0
      });
    }

    let lastTime = 0;
    const targetFPS = 30;
    const frameInterval = 1000 / targetFPS;

    const animate = (currentTime: number) => {
      if (currentTime - lastTime < frameInterval) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }

      lastTime = currentTime;

      // Clear canvas with slight trailing effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

      // Update and draw columns
      columns.forEach((column, index) => {
        // Update character array
        if (currentTime - column.lastUpdate > 100 / column.speed) {
          if (Math.random() < 0.3) {
            column.chars.unshift(
              charactersArray[Math.floor(Math.random() * charactersArray.length)]
            );
          }
          
          // Remove characters that are off screen
          if (column.chars.length > 50) {
            column.chars.pop();
          }
          
          column.lastUpdate = currentTime;
        }

        // Move column down
        column.y += column.speed;

        // Reset column when it goes off screen
        if (column.y > canvas.offsetHeight + 200) {
          column.y = Math.random() * -500 - 100;
          column.chars = [];
          column.speed = (Math.random() * 2 + 1) * settings.speed * speed;
          column.opacity = Math.random() * settings.opacity + 0.2;
        }

        // Draw characters
        ctx.font = '16px monospace';
        ctx.textAlign = 'center';

        column.chars.forEach((char, charIndex) => {
          const charY = column.y + charIndex * 20;
          
          if (charY > -20 && charY < canvas.offsetHeight + 20) {
            // Calculate opacity based on position in column
            const fadeOpacity = charIndex === 0 ? 1 : Math.max(0.1, 1 - (charIndex / column.chars.length));
            const finalOpacity = column.opacity * fadeOpacity;
            
            // Draw character shadow
            ctx.fillStyle = `rgba(0, 0, 0, ${finalOpacity * 0.8})`;
            ctx.fillText(char, column.x + 1, charY + 1);
            
            // Draw main character with gradient effect
            if (charIndex < 3) {
              // Bright head of the trail
              ctx.fillStyle = `${colors.primary}${Math.floor(finalOpacity * 255).toString(16).padStart(2, '0')}`;
            } else if (charIndex < 8) {
              // Medium brightness
              ctx.fillStyle = `${colors.secondary}${Math.floor(finalOpacity * 200).toString(16).padStart(2, '0')}`;
            } else {
              // Dim tail
              ctx.fillStyle = `${colors.shadow}${Math.floor(finalOpacity * 150).toString(16).padStart(2, '0')}`;
            }
            
            ctx.fillText(char, column.x, charY);

            // Add glow effect for first character
            if (charIndex === 0 && finalOpacity > 0.7) {
              ctx.shadowColor = colors.primary;
              ctx.shadowBlur = 10;
              ctx.fillStyle = colors.primary;
              ctx.fillText(char, column.x, charY);
              ctx.shadowBlur = 0;
            }
          }
        });
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', setCanvasSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [intensity, color, speed]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{
        width: '100%',
        height: '100%',
        zIndex: -1
      }}
    />
  );
};

export default MatrixRain;