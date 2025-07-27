export interface ChatMessage {
  id: string;
  text: string;
  type: 'user' | 'assistant';
  timestamp: Date;
  confidence?: number;
  language?: string;
  session_id?: string;
}

export interface VoiceState {
  isListening: boolean;
  isProcessing: boolean;
  volume: number;
  status: 'idle' | 'listening' | 'processing' | 'speaking' | 'error';
  error?: string;
}

export interface SessionInfo {
  session_id: string;
  user_id: string;
  created_at: Date;
  last_activity: Date;
  status: string;
  context_length: number;
}

export interface SystemStats {
  active_connections: number;
  redis_connected: boolean;
  uptime: string;
  total_sessions: number;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    core: string;
    redis: string;
    audio: string;
    ai: string;
    web: string;
  };
  timestamp: Date;
  version: string;
}

export interface WebSocketMessage {
  type: 'chat' | 'voice' | 'ping' | 'pong' | 'status' | 'error';
  data?: any;
  timestamp?: string;
  session_id?: string;
  message?: string;
}

export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: number;
}

export interface ChatRequest {
  message: string;
  user_id: string;
  session_id?: string;
  language?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  timestamp: Date;
  confidence: number;
  processing_time: number;
}

export interface UserSettings {
  theme: 'light' | 'dark' | 'auto';
  language: 'en' | 'th';
  voice_enabled: boolean;
  notifications_enabled: boolean;
  auto_scroll: boolean;
  wake_word_enabled: boolean;
  wake_words: string[];
}