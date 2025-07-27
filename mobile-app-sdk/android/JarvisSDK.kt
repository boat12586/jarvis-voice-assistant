package com.jarvis.sdk

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.content.PermissionChecker
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

// MARK: - Models

data class JarvisConfig(
    val baseURL: String,
    val apiKey: String,
    val timeout: Long = 30L,
    val retryAttempts: Int = 3,
    val enableLogging: Boolean = false
)

data class DeviceInfo(
    @SerializedName("device_id") val deviceID: String,
    val platform: String,
    @SerializedName("app_version") val appVersion: String,
    @SerializedName("os_version") val osVersion: String,
    @SerializedName("device_model") val deviceModel: String,
    val timezone: String
)

data class AuthRequest(
    @SerializedName("device_info") val deviceInfo: DeviceInfo
)

data class AuthResponse(
    val success: Boolean,
    @SerializedName("user_id") val userID: String,
    @SerializedName("mobile_token") val token: String,
    @SerializedName("session_id") val sessionID: String,
    @SerializedName("expires_at") val expiresAt: String,
    @SerializedName("user_info") val userInfo: Map<String, Any>
)

data class VoiceRequest(
    @SerializedName("session_id") val sessionID: String,
    @SerializedName("audio_data") val audioData: String? = null,
    val text: String? = null,
    @SerializedName("voice_settings") val voiceSettings: VoiceSettings? = null,
    val language: String = "en",
    val format: String = "wav"
)

data class VoiceSettings(
    val voice: String = "en-US-AriaNeural"
)

data class VoiceResponse(
    val success: Boolean,
    @SerializedName("response_text") val text: String?,
    @SerializedName("response_audio") val audioData: String?,
    @SerializedName("processing_time") val processingTime: Double,
    val confidence: Double,
    val language: String
)

data class CommandRequest(
    @SerializedName("session_id") val sessionID: String,
    val command: String,
    val language: String = "en",
    @SerializedName("send_notification") val sendNotification: Boolean = false
)

data class CommandResponse(
    val success: Boolean,
    val command: String,
    val response: String,
    val confidence: Double,
    @SerializedName("processing_time") val processingTime: Double
)

// MARK: - Exceptions

sealed class JarvisException(message: String) : Exception(message) {
    class AuthenticationFailed(message: String) : JarvisException("Authentication failed: $message")
    class NetworkError(message: String) : JarvisException("Network error: $message")
    class VoiceProcessingError(message: String) : JarvisException("Voice processing error: $message")
    class ConfigurationError(message: String) : JarvisException("Configuration error: $message")
    class PermissionDenied(message: String) : JarvisException("Permission denied: $message")
    class SessionExpired : JarvisException("Session expired")
    class InvalidResponse : JarvisException("Invalid response from server")
}

// MARK: - Result Wrapper

sealed class Result<T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error<T>(val error: JarvisException) : Result<T>()
}

// MARK: - Main SDK Class

class JarvisClient(private val context: Context) {
    private var config: JarvisConfig? = null
    private var authToken: String? = null
    private var sessionID: String? = null
    private var userID: String? = null
    
    private val httpClient: OkHttpClient
    private val gson = Gson()
    private val mainHandler = Handler(Looper.getMainLooper())
    
    // Audio recording
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    private var voiceSessionActive = false
    
    // Audio configuration
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    
    companion object {
        private const val TAG = "JarvisSDK"
        private const val PERMISSION_RECORD_AUDIO = android.Manifest.permission.RECORD_AUDIO
    }
    
    init {
        httpClient = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }
    
    // MARK: - Configuration
    
    fun configure(config: JarvisConfig) {
        this.config = config
        if (config.enableLogging) {
            Log.d(TAG, "Jarvis SDK configured with base URL: ${config.baseURL}")
        }
    }
    
    // MARK: - Authentication
    
    suspend fun authenticate(deviceInfo: DeviceInfo): Result<AuthResponse> {
        val config = this.config ?: return Result.Error(JarvisException.ConfigurationError("SDK not configured"))
        
        val url = "${config.baseURL}/api/v2/mobile/auth/login"
        val request = AuthRequest(deviceInfo)
        
        return try {
            val response = makeRequest<AuthResponse>(url, request, config.apiKey)
            
            response.data?.let { authResponse ->
                if (authResponse.success) {
                    // Store authentication data
                    authToken = authResponse.token
                    sessionID = authResponse.sessionID
                    userID = authResponse.userID
                    
                    Result.Success(authResponse)
                } else {
                    Result.Error(JarvisException.AuthenticationFailed("Authentication failed"))
                }
            } ?: Result.Error(JarvisException.InvalidResponse())
            
        } catch (e: Exception) {
            Result.Error(JarvisException.NetworkError(e.message ?: "Unknown error"))
        }
    }
    
    // MARK: - Voice Processing
    
    suspend fun startVoiceSession(): Result<VoiceResponse> {
        if (authToken == null) {
            return Result.Error(JarvisException.AuthenticationFailed("Not authenticated"))
        }
        
        // Check microphone permission
        if (!hasMicrophonePermission()) {
            return Result.Error(JarvisException.PermissionDenied("Microphone access denied"))
        }
        
        return try {
            startRecording()
            Result.Success(VoiceResponse(
                success = true,
                text = "Voice session started",
                audioData = null,
                processingTime = 0.0,
                confidence = 1.0,
                language = "en"
            ))
        } catch (e: Exception) {
            Result.Error(JarvisException.VoiceProcessingError(e.message ?: "Failed to start voice session"))
        }
    }
    
    private fun hasMicrophonePermission(): Boolean {
        return ContextCompat.checkSelfPermission(context, PERMISSION_RECORD_AUDIO) == PermissionChecker.PERMISSION_GRANTED
    }
    
    private fun startRecording() {
        if (isRecording) return
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )
        
        audioRecord?.startRecording()
        isRecording = true
        voiceSessionActive = true
        
        recordingThread = Thread {
            val buffer = ByteArray(bufferSize)
            
            while (isRecording) {
                val bytesRead = audioRecord?.read(buffer, 0, bufferSize) ?: 0
                if (bytesRead > 0) {
                    // Process audio buffer
                    processAudioBuffer(buffer.copyOf(bytesRead))
                }
            }
        }
        
        recordingThread?.start()
        
        if (config?.enableLogging == true) {
            Log.d(TAG, "Voice recording started")
        }
    }
    
    private fun processAudioBuffer(buffer: ByteArray) {
        // Convert audio buffer to base64
        val base64Audio = Base64.encodeToString(buffer, Base64.DEFAULT)
        
        // Send to server for processing
        GlobalScope.launch {
            processVoiceData(base64Audio)
        }
    }
    
    private suspend fun processVoiceData(audioData: String): Result<VoiceResponse> {
        val config = this.config ?: return Result.Error(JarvisException.ConfigurationError("SDK not configured"))
        val sessionID = this.sessionID ?: return Result.Error(JarvisException.SessionExpired())
        
        val url = "${config.baseURL}/api/v2/mobile/voice/process"
        val request = VoiceRequest(
            sessionID = sessionID,
            audioData = audioData,
            language = "en",
            format = "wav"
        )
        
        return try {
            val response = makeAuthenticatedRequest<VoiceResponse>(url, request)
            response
        } catch (e: Exception) {
            Result.Error(JarvisException.VoiceProcessingError(e.message ?: "Voice processing failed"))
        }
    }
    
    fun stopVoiceSession() {
        if (isRecording) {
            isRecording = false
            voiceSessionActive = false
            
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
            
            recordingThread?.join()
            recordingThread = null
            
            if (config?.enableLogging == true) {
                Log.d(TAG, "Voice recording stopped")
            }
        }
    }
    
    // MARK: - Command Processing
    
    suspend fun executeCommand(command: String): Result<CommandResponse> {
        val config = this.config ?: return Result.Error(JarvisException.ConfigurationError("SDK not configured"))
        val sessionID = this.sessionID ?: return Result.Error(JarvisException.SessionExpired())
        
        val url = "${config.baseURL}/api/v2/mobile/command"
        val request = CommandRequest(
            sessionID = sessionID,
            command = command,
            language = "en",
            sendNotification = false
        )
        
        return try {
            val response = makeAuthenticatedRequest<CommandResponse>(url, request)
            response
        } catch (e: Exception) {
            Result.Error(JarvisException.NetworkError(e.message ?: "Command execution failed"))
        }
    }
    
    // MARK: - Text-to-Speech
    
    suspend fun synthesizeSpeech(text: String, voice: String = "en-US-AriaNeural"): Result<VoiceResponse> {
        val config = this.config ?: return Result.Error(JarvisException.ConfigurationError("SDK not configured"))
        val sessionID = this.sessionID ?: return Result.Error(JarvisException.SessionExpired())
        
        val url = "${config.baseURL}/api/v2/mobile/voice/process"
        val request = VoiceRequest(
            sessionID = sessionID,
            text = text,
            voiceSettings = VoiceSettings(voice = voice),
            language = "en"
        )
        
        return try {
            val response = makeAuthenticatedRequest<VoiceResponse>(url, request)
            response
        } catch (e: Exception) {
            Result.Error(JarvisException.VoiceProcessingError(e.message ?: "TTS processing failed"))
        }
    }
    
    // MARK: - HTTP Utilities
    
    private suspend inline fun <reified T> makeRequest(url: String, requestBody: Any, apiKey: String): Result<T> {
        return withContext(Dispatchers.IO) {
            try {
                val json = gson.toJson(requestBody)
                val body = json.toRequestBody("application/json".toMediaType())
                
                val request = Request.Builder()
                    .url(url)
                    .post(body)
                    .addHeader("Content-Type", "application/json")
                    .addHeader("Authorization", "Bearer $apiKey")
                    .build()
                
                val response = httpClient.newCall(request).execute()
                
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    if (responseBody != null) {
                        val result = gson.fromJson(responseBody, T::class.java)
                        Result.Success(result)
                    } else {
                        Result.Error(JarvisException.InvalidResponse())
                    }
                } else {
                    Result.Error(JarvisException.NetworkError("HTTP ${response.code}"))
                }
            } catch (e: IOException) {
                Result.Error(JarvisException.NetworkError(e.message ?: "Network request failed"))
            } catch (e: Exception) {
                Result.Error(JarvisException.NetworkError(e.message ?: "Unknown error"))
            }
        }
    }
    
    private suspend inline fun <reified T> makeAuthenticatedRequest(url: String, requestBody: Any): Result<T> {
        val authToken = this.authToken ?: return Result.Error(JarvisException.AuthenticationFailed("Not authenticated"))
        
        return withContext(Dispatchers.IO) {
            try {
                val json = gson.toJson(requestBody)
                val body = json.toRequestBody("application/json".toMediaType())
                
                val request = Request.Builder()
                    .url(url)
                    .post(body)
                    .addHeader("Content-Type", "application/json")
                    .addHeader("Authorization", "Bearer $authToken")
                    .build()
                
                val response = httpClient.newCall(request).execute()
                
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    if (responseBody != null) {
                        val result = gson.fromJson(responseBody, T::class.java)
                        Result.Success(result)
                    } else {
                        Result.Error(JarvisException.InvalidResponse())
                    }
                } else {
                    Result.Error(JarvisException.NetworkError("HTTP ${response.code}"))
                }
            } catch (e: IOException) {
                Result.Error(JarvisException.NetworkError(e.message ?: "Network request failed"))
            } catch (e: Exception) {
                Result.Error(JarvisException.NetworkError(e.message ?: "Unknown error"))
            }
        }
    }
    
    // MARK: - Utility Methods
    
    fun isAuthenticated(): Boolean {
        return authToken != null && userID != null
    }
    
    fun getCurrentUserID(): String? {
        return userID
    }
    
    fun getCurrentSessionID(): String? {
        return sessionID
    }
    
    fun logout() {
        authToken = null
        sessionID = null
        userID = null
        stopVoiceSession()
    }
    
    // MARK: - Cleanup
    
    fun cleanup() {
        stopVoiceSession()
        httpClient.dispatcher.executorService.shutdown()
    }
}

// MARK: - Extensions

fun JarvisClient.executeCommandAsync(command: String, callback: (Result<CommandResponse>) -> Unit) {
    GlobalScope.launch {
        val result = executeCommand(command)
        withContext(Dispatchers.Main) {
            callback(result)
        }
    }
}

fun JarvisClient.synthesizeSpeechAsync(text: String, voice: String = "en-US-AriaNeural", callback: (Result<VoiceResponse>) -> Unit) {
    GlobalScope.launch {
        val result = synthesizeSpeech(text, voice)
        withContext(Dispatchers.Main) {
            callback(result)
        }
    }
}

fun JarvisClient.authenticateAsync(deviceInfo: DeviceInfo, callback: (Result<AuthResponse>) -> Unit) {
    GlobalScope.launch {
        val result = authenticate(deviceInfo)
        withContext(Dispatchers.Main) {
            callback(result)
        }
    }
}