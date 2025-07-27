'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Settings, Maximize2, Minimize2, Wifi, WifiOff } from 'lucide-react';
import { useJarvisConnection } from '@/hooks/useJarvisConnection';
import RadarScanner from '@/components/FuturisticRadarScanner';
import HolographicPanel from '@/components/HolographicPanel';
import ChatInterface from '@/components/ChatInterface';
import SystemMonitor from '@/components/SystemMonitor';
import VoiceWaveform from '@/components/VoiceWaveform';

export default function Dashboard() {
  const [userId] = useState('user_' + Math.random().toString(36).substr(2, 9));
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  const {
    isConnected,
    isLoading,
    error,
    messages,
    isTyping,
    voiceState,
    healthStatus,
    systemStats,
    sendMessage,
    toggleVoice,
    clearMessages,
    refreshSystemStatus
  } = useJarvisConnection(userId);

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <div className="text-cyan-400 font-mono text-lg">Initializing Jarvis...</div>
          <div className="text-gray-400 text-sm mt-2">Connecting to neural network</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen overflow-hidden p-4">
      {/* Top Status Bar */}
      <div className="flex items-center justify-between mb-6">
        {/* Left Status */}
        <div className="flex items-center space-x-4">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-2"
          >
            {isConnected ? (
              <Wifi className="w-5 h-5 text-green-400" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-400" />
            )}
            <span className="text-sm font-mono text-gray-300">
              {isConnected ? 'ONLINE' : 'OFFLINE'}
            </span>
          </motion.div>
          
          {error && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-red-500/20 border border-red-500/30 rounded px-3 py-1"
            >
              <span className="text-red-400 text-sm">{error}</span>
            </motion.div>
          )}
        </div>

        {/* Center Title */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
            J.A.R.V.I.S v2.0
          </h1>
          <div className="text-sm text-gray-400 font-mono">
            {currentTime.toLocaleTimeString()}
          </div>
        </motion.div>

        {/* Right Controls */}
        <div className="flex items-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={refreshSystemStatus}
            className="p-2 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 text-cyan-400 transition-all duration-200"
          >
            <Settings className="w-5 h-5" />
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleFullscreen}
            className="p-2 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 text-cyan-400 transition-all duration-200"
          >
            {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </motion.button>
        </div>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-12 grid-rows-12 gap-4 h-[calc(100vh-120px)]">
        {/* Left Panel - System Monitor */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="col-span-3 row-span-12"
        >
          <SystemMonitor healthStatus={healthStatus} systemStats={systemStats} />
        </motion.div>

        {/* Center Panel - Main Interface */}
        <div className="col-span-6 row-span-12 flex flex-col items-center justify-center space-y-6">
          {/* Central Radar Scanner */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="flex-shrink-0"
          >
            <RadarScanner 
              voiceState={voiceState} 
              isActive={isConnected}
              size="xl"
            />
          </motion.div>

          {/* Voice Controls */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex-shrink-0"
          >
            <HolographicPanel title="Voice Interface" glowColor="green" variant="glass">
              <div className="flex flex-col items-center space-y-4">
                <VoiceWaveform 
                  isActive={voiceState.isListening} 
                  volume={voiceState.volume}
                  height="lg"
                  barCount={16}
                  color={voiceState.status === 'error' ? 'red' : 'cyan'}
                />
                
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={toggleVoice}
                  disabled={!isConnected}
                  className={`w-16 h-16 rounded-full border-2 transition-all duration-300 ${
                    voiceState.isListening
                      ? 'bg-red-500/20 border-red-400 shadow-lg shadow-red-400/25'
                      : 'bg-cyan-500/20 border-cyan-400 shadow-lg shadow-cyan-400/25'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <div className={`w-6 h-6 mx-auto rounded-full ${
                    voiceState.isListening ? 'bg-red-400' : 'bg-cyan-400'
                  }`} />
                </motion.button>
                
                <div className="text-center">
                  <div className="text-cyan-400 font-mono text-sm uppercase">
                    {voiceState.status}
                  </div>
                  {voiceState.error && (
                    <div className="text-red-400 text-xs mt-1">
                      {voiceState.error}
                    </div>
                  )}
                </div>
              </div>
            </HolographicPanel>
          </motion.div>

          {/* Quick Stats */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="flex-shrink-0 w-full max-w-md"
          >
            <HolographicPanel glowColor="blue" variant="glass">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-cyan-400 text-lg font-mono">{messages.length}</div>
                  <div className="text-gray-400 text-xs">Messages</div>
                </div>
                <div>
                  <div className="text-green-400 text-lg font-mono">
                    {systemStats?.active_connections || 0}
                  </div>
                  <div className="text-gray-400 text-xs">Active</div>
                </div>
                <div>
                  <div className="text-yellow-400 text-lg font-mono">
                    {healthStatus?.status === 'healthy' ? '100' : 
                     healthStatus?.status === 'degraded' ? '75' : '0'}%
                  </div>
                  <div className="text-gray-400 text-xs">Health</div>
                </div>
              </div>
            </HolographicPanel>
          </motion.div>
        </div>

        {/* Right Panel - Chat Interface */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="col-span-3 row-span-12"
        >
          <ChatInterface
            messages={messages}
            onSendMessage={sendMessage}
            onVoiceToggle={toggleVoice}
            voiceState={voiceState}
            isConnected={isConnected}
          />
        </motion.div>
      </div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="absolute bottom-4 left-1/2 transform -translate-x-1/2"
      >
        <div className="text-center text-xs text-gray-500 font-mono">
          Jarvis Voice Assistant v2.0 | Neural Network Active | User: {userId}
        </div>
      </motion.div>
    </div>
  );
}