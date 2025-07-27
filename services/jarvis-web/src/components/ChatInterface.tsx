'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Mic, MicOff, Volume2, VolumeX } from 'lucide-react';
import { ChatMessage, VoiceState } from '@/types/jarvis';
import HolographicPanel from './HolographicPanel';
import VoiceWaveform from './VoiceWaveform';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  onVoiceToggle: () => void;
  voiceState: VoiceState;
  isConnected: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  onVoiceToggle,
  voiceState,
  isConnected
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (inputMessage.trim() && isConnected) {
      onSendMessage(inputMessage.trim());
      setInputMessage('');
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputMessage(e.target.value);
    setIsTyping(e.target.value.length > 0);
  };

  const MessageBubble = ({ message }: { message: ChatMessage }) => {
    const isUser = message.type === 'user';
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.9 }}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div
          className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
            isUser
              ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white ml-4'
              : 'bg-gray-800/80 backdrop-blur-sm border border-cyan-500/20 text-gray-100 mr-4'
          } shadow-lg`}
        >
          <div className="text-sm break-words">{message.text}</div>
          <div className={`text-xs mt-1 ${isUser ? 'text-cyan-200' : 'text-gray-400'} flex justify-between items-center`}>
            <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
            {message.confidence && (
              <span className="ml-2">
                {(message.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>
      </motion.div>
    );
  };

  const TypingIndicator = () => (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex justify-start mb-4"
    >
      <div className="bg-gray-800/80 backdrop-blur-sm border border-cyan-500/20 rounded-lg px-4 py-2 mr-4">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </motion.div>
  );

  return (
    <HolographicPanel title="Communication Interface" glowColor="cyan" className="h-full flex flex-col">
      {/* Connection Status */}
      <div className="flex items-center justify-between mb-4 p-2 bg-gray-800/50 rounded">
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
          <span className="text-xs font-mono text-gray-300">
            {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <VoiceWaveform 
            isActive={voiceState.isListening} 
            volume={voiceState.volume} 
            height="sm"
            barCount={8}
          />
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2 min-h-0">
        <AnimatePresence>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {voiceState.isProcessing && <TypingIndicator />}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-cyan-500/20 pt-4 mt-4">
        {/* Voice Status */}
        {voiceState.status !== 'idle' && (
          <div className="mb-3 p-2 bg-gray-800/30 rounded text-center">
            <span className="text-cyan-400 text-sm font-mono uppercase">
              {voiceState.status}
            </span>
            {voiceState.error && (
              <div className="text-red-400 text-xs mt-1">{voiceState.error}</div>
            )}
          </div>
        )}

        {/* Input Controls */}
        <div className="flex items-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onVoiceToggle}
            className={`p-3 rounded-full transition-all duration-200 ${
              voiceState.isListening
                ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/25'
                : 'bg-cyan-600 hover:bg-cyan-700 shadow-lg shadow-cyan-500/25'
            }`}
            disabled={!isConnected}
          >
            {voiceState.isListening ? (
              <MicOff className="w-5 h-5 text-white" />
            ) : (
              <Mic className="w-5 h-5 text-white" />
            )}
          </motion.button>

          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder={isConnected ? "Type your message..." : "Not connected"}
              disabled={!isConnected}
              className="w-full px-4 py-3 bg-gray-800/50 backdrop-blur-sm border border-cyan-500/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all duration-200"
            />
            {isTyping && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute right-3 top-1/2 transform -translate-y-1/2"
              >
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
              </motion.div>
            )}
          </div>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || !isConnected}
            className="p-3 rounded-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed shadow-lg shadow-cyan-500/25 transition-all duration-200"
          >
            <Send className="w-5 h-5 text-white" />
          </motion.button>
        </div>

        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2 mt-3">
          {[
            'What can you do?',
            'System status',
            'Clear conversation',
            'Settings'
          ].map((action, index) => (
            <motion.button
              key={action}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onSendMessage(action)}
              disabled={!isConnected}
              className="px-3 py-1 text-xs bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 rounded-full text-cyan-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {action}
            </motion.button>
          ))}
        </div>
      </div>
    </HolographicPanel>
  );
};

export default ChatInterface;