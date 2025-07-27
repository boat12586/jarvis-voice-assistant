'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface HolographicPanelProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  glowColor?: 'cyan' | 'blue' | 'green' | 'yellow' | 'red' | 'purple';
  variant?: 'solid' | 'glass' | 'outline';
  animated?: boolean;
}

const HolographicPanel: React.FC<HolographicPanelProps> = ({
  children,
  className = '',
  title,
  glowColor = 'cyan',
  variant = 'glass',
  animated = true
}) => {
  const glowColors = {
    cyan: 'shadow-cyan-500/20 border-cyan-500/30 bg-cyan-500/5',
    blue: 'shadow-blue-500/20 border-blue-500/30 bg-blue-500/5',
    green: 'shadow-green-500/20 border-green-500/30 bg-green-500/5',
    yellow: 'shadow-yellow-500/20 border-yellow-500/30 bg-yellow-500/5',
    red: 'shadow-red-500/20 border-red-500/30 bg-red-500/5',
    purple: 'shadow-purple-500/20 border-purple-500/30 bg-purple-500/5'
  };

  const borderGlow = {
    cyan: 'border-cyan-400/50 shadow-lg shadow-cyan-400/25',
    blue: 'border-blue-400/50 shadow-lg shadow-blue-400/25',
    green: 'border-green-400/50 shadow-lg shadow-green-400/25',
    yellow: 'border-yellow-400/50 shadow-lg shadow-yellow-400/25',
    red: 'border-red-400/50 shadow-lg shadow-red-400/25',
    purple: 'border-purple-400/50 shadow-lg shadow-purple-400/25'
  };

  const getVariantClasses = () => {
    switch (variant) {
      case 'solid':
        return `bg-gray-900/90 border-2 ${borderGlow[glowColor]}`;
      case 'glass':
        return `backdrop-blur-md bg-gray-900/20 border ${glowColors[glowColor]}`;
      case 'outline':
        return `bg-transparent border-2 ${borderGlow[glowColor]}`;
      default:
        return `backdrop-blur-md bg-gray-900/20 border ${glowColors[glowColor]}`;
    }
  };

  const panelContent = (
    <div className={`relative rounded-lg overflow-hidden ${getVariantClasses()} ${className}`}>
      {/* Corner Decorations */}
      <div className="absolute top-0 left-0 w-4 h-4 border-l-2 border-t-2 border-cyan-400 opacity-60"></div>
      <div className="absolute top-0 right-0 w-4 h-4 border-r-2 border-t-2 border-cyan-400 opacity-60"></div>
      <div className="absolute bottom-0 left-0 w-4 h-4 border-l-2 border-b-2 border-cyan-400 opacity-60"></div>
      <div className="absolute bottom-0 right-0 w-4 h-4 border-r-2 border-b-2 border-cyan-400 opacity-60"></div>

      {/* Scanning Line Effect */}
      <div className="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent opacity-30 animate-pulse"></div>

      {/* Title Bar */}
      {title && (
        <div className="relative border-b border-cyan-500/20 px-4 py-2 bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent">
          <div className="flex items-center justify-between">
            <h3 className="text-cyan-400 font-mono text-sm uppercase tracking-wider">{title}</h3>
            <div className="flex space-x-1">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
              <div className="w-2 h-2 rounded-full bg-yellow-400 opacity-60"></div>
              <div className="w-2 h-2 rounded-full bg-red-400 opacity-40"></div>
            </div>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="relative p-4">
        {children}
      </div>

      {/* Holographic Grid Overlay */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(6, 182, 212, 0.5) 1px, transparent 1px),
            linear-gradient(90deg, rgba(6, 182, 212, 0.5) 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px'
        }}
      ></div>
    </div>
  );

  if (animated) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.95 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        whileHover={{ scale: 1.02 }}
        className="relative"
      >
        {panelContent}
      </motion.div>
    );
  }

  return panelContent;
};

export default HolographicPanel;