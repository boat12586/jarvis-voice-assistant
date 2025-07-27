import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import { PermissionsAndroid, Platform } from 'react-native';
import { v4 as uuidv4 } from 'uuid';
import 'react-native-get-random-values';

// MARK: - Types

export interface JarvisConfig {
  baseURL: string;
  apiKey: string;
  timeout?: number;
  retryAttempts?: number;
  enableLogging?: boolean;
}

export interface DeviceInfo {
  deviceID: string;
  platform: 'ios' | 'android';
  appVersion: string;
  osVersion: string;
  deviceModel?: string;
  timezone?: string;
}

export interface AuthenticationResult {
  success: boolean;
  userID: string;
  token: string;
  sessionID: string;
  expiresAt: Date;
  userInfo: Record<string, any>;
}

export interface VoiceSettings {
  voice?: string;
  language?: string;
  sampleRate?: number;
  enableNoiseReduction?: boolean;
  enableVAD?: boolean;
}

export interface VoiceResponse {
  success: boolean;
  text?: string;
  audioData?: string;
  processingTime: number;
  confidence: number;
  language: string;
}

export interface CommandResponse {
  success: boolean;
  command: string;
  response: string;
  confidence: number;
  processingTime: number;
  metadata?: Record<string, any>;
}

export interface SessionInfo {
  sessionID: string;
  userID: string;
  deviceID: string;
  created: Date;
  lastActivity: Date;
  isActive: boolean;
}

// MARK: - Errors

export class JarvisError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: Record<string, any>
  ) {
    super(message);
    this.name = 'JarvisError';
  }
}

export class AuthenticationError extends JarvisError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'AUTHENTICATION_FAILED', details);
    this.name = 'AuthenticationError';
  }
}

export class NetworkError extends JarvisError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'NETWORK_ERROR', details);
    this.name = 'NetworkError';
  }
}

export class VoiceProcessingError extends JarvisError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'VOICE_PROCESSING_ERROR', details);
    this.name = 'VoiceProcessingError';
  }
}

export class ConfigurationError extends JarvisError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'CONFIGURATION_ERROR', details);
    this.name = 'ConfigurationError';
  }
}

export class PermissionError extends JarvisError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'PERMISSION_DENIED', details);
    this.name = 'PermissionError';
  }
}

// MARK: - Event Emitter

class EventEmitter {
  private listeners: Record<string, Function[]> = {};

  on(event: string, callback: Function) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  off(event: string, callback: Function) {
    if (!this.listeners[event]) return;
    this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
  }

  emit(event: string, ...args: any[]) {
    if (!this.listeners[event]) return;
    this.listeners[event].forEach(callback => callback(...args));
  }
}

// MARK: - Main SDK Class

export class JarvisClient extends EventEmitter {
  private config?: JarvisConfig;
  private authToken?: string;
  private sessionID?: string;
  private userID?: string;
  private audioRecorderPlayer: AudioRecorderPlayer;
  private isRecording = false;
  private voiceSessionActive = false;
  private networkConnected = true;

  constructor() {
    super();
    this.audioRecorderPlayer = new AudioRecorderPlayer();
    this.setupNetworkMonitoring();
  }

  // MARK: - Configuration

  configure(config: JarvisConfig): void {
    this.config = config;
    if (config.enableLogging) {
      console.log('Jarvis SDK configured with base URL:', config.baseURL);
    }
  }

  // MARK: - Authentication

  async authenticate(deviceInfo: DeviceInfo): Promise<AuthenticationResult> {
    if (!this.config) {
      throw new ConfigurationError('SDK not configured');
    }

    const url = `${this.config.baseURL}/api/v2/mobile/auth/login`;
    const requestBody = {
      device_info: {
        device_id: deviceInfo.deviceID,
        platform: deviceInfo.platform,
        app_version: deviceInfo.appVersion,
        os_version: deviceInfo.osVersion,
        device_model: deviceInfo.deviceModel || 'Unknown',
        timezone: deviceInfo.timezone || 'UTC'
      }
    };

    try {
      const response = await this.makeRequest(url, requestBody, this.config.apiKey);

      if (response.success) {
        // Store authentication data
        this.authToken = response.mobile_token;
        this.sessionID = response.session_id;
        this.userID = response.user_id;

        // Store in secure storage
        await AsyncStorage.setItem('jarvis_auth_token', response.mobile_token);
        await AsyncStorage.setItem('jarvis_session_id', response.session_id);
        await AsyncStorage.setItem('jarvis_user_id', response.user_id);

        const result: AuthenticationResult = {
          success: true,
          userID: response.user_id,
          token: response.mobile_token,
          sessionID: response.session_id,
          expiresAt: new Date(response.expires_at),
          userInfo: response.user_info
        };

        this.emit('authenticated', result);
        return result;
      } else {
        throw new AuthenticationError('Authentication failed');
      }
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async refreshToken(): Promise<AuthenticationResult> {
    if (!this.config || !this.authToken || !this.sessionID) {
      throw new ConfigurationError('SDK not properly configured or not authenticated');
    }

    const url = `${this.config.baseURL}/api/v2/mobile/auth/refresh`;
    const requestBody = {
      current_token: this.authToken,
      session_id: this.sessionID,
      device_id: await this.getDeviceID()
    };

    try {
      const response = await this.makeRequest(url, requestBody, this.config.apiKey);

      if (response.success) {
        // Update stored tokens
        this.authToken = response.mobile_token;
        await AsyncStorage.setItem('jarvis_auth_token', response.mobile_token);

        const result: AuthenticationResult = {
          success: true,
          userID: response.user_id,
          token: response.mobile_token,
          sessionID: response.session_id,
          expiresAt: new Date(response.expires_at),
          userInfo: response.user_info
        };

        this.emit('tokenRefreshed', result);
        return result;
      } else {
        throw new AuthenticationError('Token refresh failed');
      }
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  // MARK: - Voice Processing

  async startVoiceSession(): Promise<VoiceResponse> {
    if (!this.authToken) {
      throw new AuthenticationError('Not authenticated');
    }

    // Check microphone permission
    const hasPermission = await this.requestMicrophonePermission();
    if (!hasPermission) {
      throw new PermissionError('Microphone access denied');
    }

    try {
      // Start recording
      await this.startRecording();
      
      this.voiceSessionActive = true;
      this.emit('voiceStarted');

      return {
        success: true,
        text: 'Voice session started',
        processingTime: 0.0,
        confidence: 1.0,
        language: 'en'
      };
    } catch (error) {
      this.emit('error', error);
      throw new VoiceProcessingError('Failed to start voice session');
    }
  }

  private async requestMicrophonePermission(): Promise<boolean> {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (error) {
        console.error('Permission request error:', error);
        return false;
      }
    }
    return true; // iOS handles permissions automatically
  }

  private async startRecording(): Promise<void> {
    if (this.isRecording) return;

    try {
      const audioSet = {
        AudioEncoderAndroid: AudioRecorderPlayer.AudioEncoderAndroidType.AAC,
        AudioSourceAndroid: AudioRecorderPlayer.AudioSourceAndroidType.MIC,
        AVEncoderAudioQualityKeyIOS: AudioRecorderPlayer.AVEncoderAudioQualityIOSType.high,
        AVNumberOfChannelsKeyIOS: 1,
        AVFormatIDKeyIOS: AudioRecorderPlayer.AVFormatIDIOSType.mp4,
      };

      const uri = await this.audioRecorderPlayer.startRecorder(
        undefined,
        audioSet,
        true
      );

      this.isRecording = true;

      // Set up recording listeners
      this.audioRecorderPlayer.addRecordBackListener((e) => {
        // Process audio data
        this.processAudioData(e);
      });

      if (this.config?.enableLogging) {
        console.log('Voice recording started');
      }
    } catch (error) {
      throw new VoiceProcessingError('Failed to start recording');
    }
  }

  private async processAudioData(recordingData: any): Promise<void> {
    // Convert audio data to base64 and send to server
    try {
      const response = await this.processVoiceData(recordingData);
      if (response.success && response.text) {
        this.emit('speechRecognized', response.text);
      }
    } catch (error) {
      console.error('Audio processing error:', error);
    }
  }

  private async processVoiceData(audioData: any): Promise<VoiceResponse> {
    if (!this.config || !this.authToken || !this.sessionID) {
      throw new ConfigurationError('SDK not properly configured');
    }

    const url = `${this.config.baseURL}/api/v2/mobile/voice/process`;
    const requestBody = {
      session_id: this.sessionID,
      audio_data: audioData.base64 || audioData,
      language: 'en',
      format: 'wav'
    };

    try {
      const response = await this.makeAuthenticatedRequest(url, requestBody);
      return response as VoiceResponse;
    } catch (error) {
      throw new VoiceProcessingError('Voice processing failed');
    }
  }

  async stopVoiceSession(): Promise<void> {
    if (this.isRecording) {
      try {
        await this.audioRecorderPlayer.stopRecorder();
        this.audioRecorderPlayer.removeRecordBackListener();
        this.isRecording = false;
        this.voiceSessionActive = false;

        this.emit('voiceEnded');

        if (this.config?.enableLogging) {
          console.log('Voice recording stopped');
        }
      } catch (error) {
        console.error('Error stopping voice session:', error);
      }
    }
  }

  // MARK: - Command Processing

  async executeCommand(command: string): Promise<CommandResponse> {
    if (!this.config || !this.authToken || !this.sessionID) {
      throw new ConfigurationError('SDK not properly configured');
    }

    const url = `${this.config.baseURL}/api/v2/mobile/command`;
    const requestBody = {
      session_id: this.sessionID,
      command: command,
      language: 'en',
      send_notification: false
    };

    try {
      const response = await this.makeAuthenticatedRequest(url, requestBody);
      this.emit('commandExecuted', response);
      return response as CommandResponse;
    } catch (error) {
      this.emit('error', error);
      throw new NetworkError('Command execution failed');
    }
  }

  // MARK: - Text-to-Speech

  async synthesizeSpeech(text: string, voice: string = 'en-US-AriaNeural'): Promise<VoiceResponse> {
    if (!this.config || !this.authToken || !this.sessionID) {
      throw new ConfigurationError('SDK not properly configured');
    }

    const url = `${this.config.baseURL}/api/v2/mobile/voice/process`;
    const requestBody = {
      session_id: this.sessionID,
      text: text,
      voice_settings: {
        voice: voice
      },
      language: 'en'
    };

    try {
      const response = await this.makeAuthenticatedRequest(url, requestBody);
      
      // Play audio if available
      if (response.response_audio) {
        await this.playAudio(response.response_audio);
      }

      return response as VoiceResponse;
    } catch (error) {
      this.emit('error', error);
      throw new VoiceProcessingError('TTS processing failed');
    }
  }

  private async playAudio(base64Audio: string): Promise<void> {
    try {
      // Convert base64 to file and play
      const audioPath = `${AudioRecorderPlayer.directories.CacheDir}/jarvis_tts.mp3`;
      // Implementation depends on your audio playing strategy
      // This is a simplified version
      
      await this.audioRecorderPlayer.startPlayer(audioPath);
      
      this.audioRecorderPlayer.addPlayBackListener((e) => {
        if (e.currentPosition === e.duration) {
          this.audioRecorderPlayer.stopPlayer();
        }
      });
    } catch (error) {
      console.error('Audio playback error:', error);
    }
  }

  // MARK: - Session Management

  async createSession(sessionType: string = 'chat'): Promise<SessionInfo> {
    if (!this.config || !this.authToken) {
      throw new ConfigurationError('SDK not properly configured');
    }

    const url = `${this.config.baseURL}/api/v2/mobile/session`;
    const requestBody = {
      session_type: sessionType,
      preferences: {}
    };

    try {
      const response = await this.makeAuthenticatedRequest(url, requestBody);
      
      if (response.success) {
        this.sessionID = response.session_id;
        await AsyncStorage.setItem('jarvis_session_id', response.session_id);
        
        const sessionInfo: SessionInfo = {
          sessionID: response.session_id,
          userID: this.userID!,
          deviceID: await this.getDeviceID(),
          created: new Date(),
          lastActivity: new Date(),
          isActive: true
        };

        this.emit('sessionCreated', sessionInfo);
        return sessionInfo;
      } else {
        throw new Error('Session creation failed');
      }
    } catch (error) {
      this.emit('error', error);
      throw new NetworkError('Session creation failed');
    }
  }

  // MARK: - HTTP Utilities

  private async makeRequest(url: string, body: any, apiKey: string): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), (this.config?.timeout || 30) * 1000);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(body),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new NetworkError(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new NetworkError('Request timeout');
      }
      
      throw new NetworkError(error.message || 'Network request failed');
    }
  }

  private async makeAuthenticatedRequest(url: string, body: any): Promise<any> {
    if (!this.authToken) {
      throw new AuthenticationError('Not authenticated');
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), (this.config?.timeout || 30) * 1000);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.authToken}`
        },
        body: JSON.stringify(body),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new NetworkError(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new NetworkError('Request timeout');
      }
      
      throw new NetworkError(error.message || 'Network request failed');
    }
  }

  // MARK: - Utility Methods

  private async getDeviceID(): Promise<string> {
    let deviceID = await AsyncStorage.getItem('jarvis_device_id');
    if (!deviceID) {
      deviceID = uuidv4();
      await AsyncStorage.setItem('jarvis_device_id', deviceID);
    }
    return deviceID;
  }

  private setupNetworkMonitoring(): void {
    NetInfo.addEventListener(state => {
      const wasConnected = this.networkConnected;
      this.networkConnected = state.isConnected || false;
      
      if (wasConnected !== this.networkConnected) {
        this.emit(this.networkConnected ? 'connected' : 'disconnected');
      }
    });
  }

  // MARK: - Public Utility Methods

  isAuthenticated(): boolean {
    return !!this.authToken && !!this.userID;
  }

  getCurrentUserID(): string | undefined {
    return this.userID;
  }

  getCurrentSessionID(): string | undefined {
    return this.sessionID;
  }

  isNetworkConnected(): boolean {
    return this.networkConnected;
  }

  async logout(): Promise<void> {
    // Clear authentication data
    this.authToken = undefined;
    this.sessionID = undefined;
    this.userID = undefined;

    // Clear stored data
    await AsyncStorage.multiRemove([
      'jarvis_auth_token',
      'jarvis_session_id',
      'jarvis_user_id'
    ]);

    // Stop any active sessions
    await this.stopVoiceSession();

    this.emit('logout');
  }

  // MARK: - Cleanup

  async cleanup(): Promise<void> {
    await this.stopVoiceSession();
    await this.audioRecorderPlayer.stopPlayer();
    this.removeAllListeners();
  }
}

// MARK: - Default Export

export default JarvisClient;