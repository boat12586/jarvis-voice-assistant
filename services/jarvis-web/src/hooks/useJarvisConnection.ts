'use client';

import { useState, useEffect, useCallback } from 'react';
import { ChatMessage, VoiceState, HealthStatus, SystemStats } from '@/types/jarvis';
import { apiClient } from '@/lib/api';
import { wsManager } from '@/lib/websocket';

interface UseJarvisConnectionReturn {
  // Connection state
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Chat state
  messages: ChatMessage[];
  isTyping: boolean;
  
  // Voice state
  voiceState: VoiceState;
  
  // System state
  healthStatus: HealthStatus | null;
  systemStats: SystemStats | null;
  
  // Actions
  sendMessage: (message: string) => Promise<void>;
  toggleVoice: () => void;
  clearMessages: () => void;
  connect: () => void;
  disconnect: () => void;
  refreshSystemStatus: () => Promise<void>;
}

export const useJarvisConnection = (userId: string): UseJarvisConnectionReturn => {
  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  
  // Voice state
  const [voiceState, setVoiceState] = useState<VoiceState>({
    isListening: false,
    isProcessing: false,
    volume: 0,
    status: 'idle'
  });
  
  // System state
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);

  // Generate message ID
  const generateMessageId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  // Connect to WebSocket
  const connect = useCallback(() => {
    setIsLoading(true);
    setError(null);
    
    try {
      const socket = wsManager.connect(userId);
      
      if (socket) {
        // Handle connection events
        socket.on('connect', () => {
          setIsConnected(true);
          setIsLoading(false);
          setError(null);
          console.log('Connected to Jarvis');
        });

        socket.on('disconnect', () => {
          setIsConnected(false);
          setError('Disconnected from server');
        });

        socket.on('connect_error', (err) => {
          setIsConnected(false);
          setError(`Connection failed: ${err.message}`);
          setIsLoading(false);
        });

        // Handle chat responses
        wsManager.onChatResponse((data) => {
          setIsTyping(false);
          
          const message: ChatMessage = {
            id: generateMessageId(),
            text: data.response,
            type: 'assistant',
            timestamp: new Date(),
            confidence: data.confidence,
            session_id: data.session_id
          };
          
          setMessages(prev => [...prev, message]);
          setCurrentSessionId(data.session_id);
        });

        // Handle voice responses
        wsManager.onVoiceResponse((data) => {
          setVoiceState(prev => ({
            ...prev,
            isProcessing: false,
            status: 'idle'
          }));
          
          if (data.text) {
            const message: ChatMessage = {
              id: generateMessageId(),
              text: data.text,
              type: 'assistant',
              timestamp: new Date(),
              confidence: data.confidence,
              session_id: data.session_id
            };
            
            setMessages(prev => [...prev, message]);
          }
        });

        // Handle status updates
        wsManager.onStatusUpdate((data) => {
          setVoiceState(prev => ({
            ...prev,
            ...data
          }));
        });

        // Handle general messages
        wsManager.onMessage((message) => {
          switch (message.type) {
            case 'error':
              setError(message.data || 'Unknown error occurred');
              break;
            case 'status':
              if (message.data?.voice) {
                setVoiceState(prev => ({ ...prev, ...message.data.voice }));
              }
              break;
          }
        });
      }
    } catch (err) {
      setError('Failed to initialize connection');
      setIsLoading(false);
    }
  }, [userId]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    wsManager.disconnect();
    setIsConnected(false);
    setIsLoading(false);
  }, []);

  // Send chat message
  const sendMessage = useCallback(async (messageText: string) => {
    if (!isConnected || !messageText.trim()) return;

    // Add user message to chat
    const userMessage: ChatMessage = {
      id: generateMessageId(),
      text: messageText,
      type: 'user',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    try {
      // Send via API
      const response = await apiClient.sendChatMessage({
        message: messageText,
        user_id: userId,
        session_id: currentSessionId || undefined,
        language: 'en'
      });

      // If WebSocket is not working, handle response directly
      if (!wsManager.isConnected()) {
        setIsTyping(false);
        
        const assistantMessage: ChatMessage = {
          id: generateMessageId(),
          text: response.response,
          type: 'assistant',
          timestamp: new Date(),
          confidence: response.confidence,
          session_id: response.session_id
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        setCurrentSessionId(response.session_id);
      }
    } catch (err) {
      setIsTyping(false);
      setError('Failed to send message');
      console.error('Send message error:', err);
    }
  }, [isConnected, userId, currentSessionId]);

  // Toggle voice recording
  const toggleVoice = useCallback(() => {
    if (!isConnected) return;

    if (voiceState.isListening) {
      // Stop listening
      setVoiceState(prev => ({
        ...prev,
        isListening: false,
        isProcessing: true,
        status: 'processing'
      }));
      
      // Send stop voice command via WebSocket
      wsManager.sendMessage({
        type: 'voice',
        data: { action: 'stop' }
      });
    } else {
      // Start listening
      setVoiceState(prev => ({
        ...prev,
        isListening: true,
        status: 'listening',
        error: undefined
      }));
      
      // Send start voice command via WebSocket
      wsManager.sendMessage({
        type: 'voice',
        data: { action: 'start' }
      });
    }
  }, [isConnected, voiceState.isListening]);

  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentSessionId(null);
  }, []);

  // Refresh system status
  const refreshSystemStatus = useCallback(async () => {
    try {
      const [health, stats] = await Promise.all([
        apiClient.getHealth(),
        apiClient.getSystemStats()
      ]);
      
      setHealthStatus(health);
      setSystemStats(stats);
    } catch (err) {
      console.error('Failed to refresh system status:', err);
    }
  }, []);

  // Initialize connection on mount
  useEffect(() => {
    connect();
    
    // Refresh system status periodically
    const statusInterval = setInterval(refreshSystemStatus, 30000); // Every 30 seconds
    
    return () => {
      clearInterval(statusInterval);
      disconnect();
    };
  }, [connect, disconnect, refreshSystemStatus]);

  // Initial system status fetch
  useEffect(() => {
    refreshSystemStatus();
  }, [refreshSystemStatus]);

  // Simulate volume changes when listening
  useEffect(() => {
    if (!voiceState.isListening) return;

    const interval = setInterval(() => {
      setVoiceState(prev => ({
        ...prev,
        volume: Math.random() * 0.8 + 0.1 // Random volume between 0.1 and 0.9
      }));
    }, 100);

    return () => clearInterval(interval);
  }, [voiceState.isListening]);

  return {
    // Connection state
    isConnected,
    isLoading,
    error,
    
    // Chat state
    messages,
    isTyping,
    
    // Voice state
    voiceState,
    
    // System state
    healthStatus,
    systemStats,
    
    // Actions
    sendMessage,
    toggleVoice,
    clearMessages,
    connect,
    disconnect,
    refreshSystemStatus
  };
};