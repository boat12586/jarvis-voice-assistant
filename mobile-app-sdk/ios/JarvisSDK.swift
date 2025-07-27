import Foundation
import AVFoundation
import Network

// MARK: - Models

public struct JarvisConfig {
    public let baseURL: String
    public let apiKey: String
    public let timeout: TimeInterval
    public let retryAttempts: Int
    public let enableLogging: Bool
    
    public init(
        baseURL: String,
        apiKey: String,
        timeout: TimeInterval = 30.0,
        retryAttempts: Int = 3,
        enableLogging: Bool = false
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.timeout = timeout
        self.retryAttempts = retryAttempts
        self.enableLogging = enableLogging
    }
}

public struct DeviceInfo {
    public let deviceID: String
    public let platform: Platform
    public let appVersion: String
    public let osVersion: String
    public let deviceModel: String
    public let timezone: String
    
    public init(
        deviceID: String,
        platform: Platform,
        appVersion: String,
        osVersion: String,
        deviceModel: String = UIDevice.current.model,
        timezone: String = TimeZone.current.identifier
    ) {
        self.deviceID = deviceID
        self.platform = platform
        self.appVersion = appVersion
        self.osVersion = osVersion
        self.deviceModel = deviceModel
        self.timezone = timezone
    }
}

public enum Platform: String, CaseIterable {
    case iOS = "ios"
    case android = "android"
}

public struct AuthenticationResult {
    public let success: Bool
    public let userID: String
    public let token: String
    public let sessionID: String
    public let expiresAt: Date
    public let userInfo: [String: Any]
}

public struct VoiceResponse {
    public let success: Bool
    public let text: String?
    public let audioData: Data?
    public let processingTime: TimeInterval
    public let confidence: Double
    public let language: String
}

public struct CommandResponse {
    public let success: Bool
    public let command: String
    public let response: String
    public let confidence: Double
    public let processingTime: TimeInterval
}

// MARK: - Errors

public enum JarvisError: Error, LocalizedError {
    case authenticationFailed(String)
    case networkError(String)
    case voiceProcessingError(String)
    case configurationError(String)
    case permissionDenied(String)
    case sessionExpired
    case invalidResponse
    
    public var errorDescription: String? {
        switch self {
        case .authenticationFailed(let message):
            return "Authentication failed: \(message)"
        case .networkError(let message):
            return "Network error: \(message)"
        case .voiceProcessingError(let message):
            return "Voice processing error: \(message)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .permissionDenied(let message):
            return "Permission denied: \(message)"
        case .sessionExpired:
            return "Session expired"
        case .invalidResponse:
            return "Invalid response from server"
        }
    }
}

// MARK: - Main SDK Class

public class JarvisClient {
    private var config: JarvisConfig?
    private var authToken: String?
    private var sessionID: String?
    private var userID: String?
    
    private let urlSession: URLSession
    private let audioEngine = AVAudioEngine()
    private let speechRecognizer = SFSpeechRecognizer()
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    private var isRecording = false
    private var voiceSessionActive = false
    
    // Network monitoring
    private let networkMonitor = NWPathMonitor()
    private let networkQueue = DispatchQueue(label: "NetworkMonitor")
    
    public init() {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30.0
        configuration.timeoutIntervalForResource = 60.0
        self.urlSession = URLSession(configuration: configuration)
        
        setupNetworkMonitoring()
    }
    
    // MARK: - Configuration
    
    public func configure(config: JarvisConfig) {
        self.config = config
        if config.enableLogging {
            print("Jarvis SDK configured with base URL: \(config.baseURL)")
        }
    }
    
    // MARK: - Authentication
    
    public func authenticate(deviceInfo: DeviceInfo, completion: @escaping (Result<AuthenticationResult, JarvisError>) -> Void) {
        guard let config = config else {
            completion(.failure(.configurationError("SDK not configured")))
            return
        }
        
        let url = URL(string: "\(config.baseURL)/api/v2/mobile/auth/login")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        
        let requestBody = [
            "device_info": [
                "device_id": deviceInfo.deviceID,
                "platform": deviceInfo.platform.rawValue,
                "app_version": deviceInfo.appVersion,
                "os_version": deviceInfo.osVersion,
                "device_model": deviceInfo.deviceModel,
                "timezone": deviceInfo.timezone
            ]
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(.configurationError("Failed to serialize request")))
            return
        }
        
        urlSession.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.handleAuthResponse(data: data, response: response, error: error, completion: completion)
            }
        }.resume()
    }
    
    private func handleAuthResponse(data: Data?, response: URLResponse?, error: Error?, completion: @escaping (Result<AuthenticationResult, JarvisError>) -> Void) {
        if let error = error {
            completion(.failure(.networkError(error.localizedDescription)))
            return
        }
        
        guard let httpResponse = response as? HTTPURLResponse else {
            completion(.failure(.invalidResponse))
            return
        }
        
        guard httpResponse.statusCode == 200 else {
            completion(.failure(.authenticationFailed("HTTP \(httpResponse.statusCode)")))
            return
        }
        
        guard let data = data else {
            completion(.failure(.invalidResponse))
            return
        }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            guard let success = json?["success"] as? Bool, success,
                  let userID = json?["user_id"] as? String,
                  let token = json?["mobile_token"] as? String,
                  let sessionID = json?["session_id"] as? String else {
                completion(.failure(.authenticationFailed("Invalid response format")))
                return
            }
            
            // Store authentication data
            self.authToken = token
            self.sessionID = sessionID
            self.userID = userID
            
            let expiresAt = Date().addingTimeInterval(30 * 24 * 60 * 60) // 30 days
            let userInfo = json?["user_info"] as? [String: Any] ?? [:]
            
            let result = AuthenticationResult(
                success: true,
                userID: userID,
                token: token,
                sessionID: sessionID,
                expiresAt: expiresAt,
                userInfo: userInfo
            )
            
            completion(.success(result))
            
        } catch {
            completion(.failure(.invalidResponse))
        }
    }
    
    // MARK: - Voice Processing
    
    public func startVoiceSession(completion: @escaping (Result<VoiceResponse, JarvisError>) -> Void) {
        guard authToken != nil else {
            completion(.failure(.authenticationFailed("Not authenticated")))
            return
        }
        
        requestMicrophonePermission { [weak self] granted in
            if granted {
                self?.startRecording(completion: completion)
            } else {
                completion(.failure(.permissionDenied("Microphone access denied")))
            }
        }
    }
    
    private func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        switch AVAudioSession.sharedInstance().recordPermission {
        case .granted:
            completion(true)
        case .denied:
            completion(false)
        case .undetermined:
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        @unknown default:
            completion(false)
        }
    }
    
    private func startRecording(completion: @escaping (Result<VoiceResponse, JarvisError>) -> Void) {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                self?.processAudioBuffer(buffer, completion: completion)
            }
            
            audioEngine.prepare()
            try audioEngine.start()
            
            isRecording = true
            voiceSessionActive = true
            
            if config?.enableLogging == true {
                print("Voice recording started")
            }
            
        } catch {
            completion(.failure(.voiceProcessingError("Failed to start recording: \(error.localizedDescription)")))
        }
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, completion: @escaping (Result<VoiceResponse, JarvisError>) -> Void) {
        // Convert audio buffer to data
        guard let audioData = bufferToData(buffer) else {
            completion(.failure(.voiceProcessingError("Failed to convert audio buffer")))
            return
        }
        
        // Send to server for processing
        sendVoiceData(audioData, completion: completion)
    }
    
    private func bufferToData(_ buffer: AVAudioPCMBuffer) -> Data? {
        let audioBuffer = buffer.audioBufferList.pointee.mBuffers
        let data = Data(bytes: audioBuffer.mData!, count: Int(audioBuffer.mDataByteSize))
        return data
    }
    
    private func sendVoiceData(_ audioData: Data, completion: @escaping (Result<VoiceResponse, JarvisError>) -> Void) {
        guard let config = config,
              let authToken = authToken,
              let sessionID = sessionID else {
            completion(.failure(.configurationError("SDK not properly configured")))
            return
        }
        
        let url = URL(string: "\(config.baseURL)/api/v2/mobile/voice/process")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        let base64Audio = audioData.base64EncodedString()
        let requestBody = [
            "session_id": sessionID,
            "audio_data": base64Audio,
            "language": "en",
            "format": "wav"
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(.voiceProcessingError("Failed to serialize request")))
            return
        }
        
        urlSession.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.handleVoiceResponse(data: data, response: response, error: error, completion: completion)
            }
        }.resume()
    }
    
    private func handleVoiceResponse(data: Data?, response: URLResponse?, error: Error?, completion: @escaping (Result<VoiceResponse, JarvisError>) -> Void) {
        if let error = error {
            completion(.failure(.networkError(error.localizedDescription)))
            return
        }
        
        guard let httpResponse = response as? HTTPURLResponse else {
            completion(.failure(.invalidResponse))
            return
        }
        
        guard httpResponse.statusCode == 200 else {
            completion(.failure(.voiceProcessingError("HTTP \(httpResponse.statusCode)")))
            return
        }
        
        guard let data = data else {
            completion(.failure(.invalidResponse))
            return
        }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            guard let success = json?["success"] as? Bool, success else {
                completion(.failure(.voiceProcessingError("Voice processing failed")))
                return
            }
            
            let text = json?["response_text"] as? String
            let processingTime = json?["processing_time"] as? Double ?? 0.0
            let confidence = json?["confidence"] as? Double ?? 0.0
            let language = json?["language"] as? String ?? "en"
            
            let voiceResponse = VoiceResponse(
                success: true,
                text: text,
                audioData: nil,
                processingTime: processingTime,
                confidence: confidence,
                language: language
            )
            
            completion(.success(voiceResponse))
            
        } catch {
            completion(.failure(.invalidResponse))
        }
    }
    
    public func stopVoiceSession() {
        if isRecording {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
            isRecording = false
            voiceSessionActive = false
            
            if config?.enableLogging == true {
                print("Voice recording stopped")
            }
        }
    }
    
    // MARK: - Command Processing
    
    public func executeCommand(_ command: String, completion: @escaping (Result<CommandResponse, JarvisError>) -> Void) {
        guard let config = config,
              let authToken = authToken,
              let sessionID = sessionID else {
            completion(.failure(.configurationError("SDK not properly configured")))
            return
        }
        
        let url = URL(string: "\(config.baseURL)/api/v2/mobile/command")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        let requestBody = [
            "session_id": sessionID,
            "command": command,
            "language": "en",
            "send_notification": false
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(.configurationError("Failed to serialize request")))
            return
        }
        
        urlSession.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.handleCommandResponse(data: data, response: response, error: error, completion: completion)
            }
        }.resume()
    }
    
    private func handleCommandResponse(data: Data?, response: URLResponse?, error: Error?, completion: @escaping (Result<CommandResponse, JarvisError>) -> Void) {
        if let error = error {
            completion(.failure(.networkError(error.localizedDescription)))
            return
        }
        
        guard let httpResponse = response as? HTTPURLResponse else {
            completion(.failure(.invalidResponse))
            return
        }
        
        guard httpResponse.statusCode == 200 else {
            completion(.failure(.networkError("HTTP \(httpResponse.statusCode)")))
            return
        }
        
        guard let data = data else {
            completion(.failure(.invalidResponse))
            return
        }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            guard let success = json?["success"] as? Bool, success else {
                completion(.failure(.voiceProcessingError("Command processing failed")))
                return
            }
            
            let command = json?["command"] as? String ?? ""
            let response = json?["response"] as? String ?? ""
            let confidence = json?["confidence"] as? Double ?? 0.0
            let processingTime = json?["processing_time"] as? Double ?? 0.0
            
            let commandResponse = CommandResponse(
                success: true,
                command: command,
                response: response,
                confidence: confidence,
                processingTime: processingTime
            )
            
            completion(.success(commandResponse))
            
        } catch {
            completion(.failure(.invalidResponse))
        }
    }
    
    // MARK: - Text-to-Speech
    
    public func synthesizeSpeech(text: String, voice: String = "en-US-AriaNeural", completion: @escaping (Result<VoiceResponse, JarvisError>) -> Void) {
        guard let config = config,
              let authToken = authToken,
              let sessionID = sessionID else {
            completion(.failure(.configurationError("SDK not properly configured")))
            return
        }
        
        let url = URL(string: "\(config.baseURL)/api/v2/mobile/voice/process")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        let requestBody = [
            "session_id": sessionID,
            "text": text,
            "voice_settings": [
                "voice": voice
            ],
            "language": "en"
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(.configurationError("Failed to serialize request")))
            return
        }
        
        urlSession.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.handleTTSResponse(data: data, response: response, error: error, completion: completion)
            }
        }.resume()
    }
    
    private func handleTTSResponse(data: Data?, response: URLResponse?, error: Error?, completion: @escaping (Result<VoiceResponse, JarvisError>) -> Void) {
        if let error = error {
            completion(.failure(.networkError(error.localizedDescription)))
            return
        }
        
        guard let httpResponse = response as? HTTPURLResponse else {
            completion(.failure(.invalidResponse))
            return
        }
        
        guard httpResponse.statusCode == 200 else {
            completion(.failure(.voiceProcessingError("HTTP \(httpResponse.statusCode)")))
            return
        }
        
        guard let data = data else {
            completion(.failure(.invalidResponse))
            return
        }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            guard let success = json?["success"] as? Bool, success else {
                completion(.failure(.voiceProcessingError("TTS processing failed")))
                return
            }
            
            let text = json?["response_text"] as? String
            let processingTime = json?["processing_time"] as? Double ?? 0.0
            let confidence = json?["confidence"] as? Double ?? 1.0
            let language = json?["language"] as? String ?? "en"
            
            var audioData: Data?
            if let base64Audio = json?["response_audio"] as? String {
                audioData = Data(base64Encoded: base64Audio)
            }
            
            let voiceResponse = VoiceResponse(
                success: true,
                text: text,
                audioData: audioData,
                processingTime: processingTime,
                confidence: confidence,
                language: language
            )
            
            completion(.success(voiceResponse))
            
        } catch {
            completion(.failure(.invalidResponse))
        }
    }
    
    // MARK: - Network Monitoring
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            if path.status == .satisfied {
                if self?.config?.enableLogging == true {
                    print("Network connection available")
                }
            } else {
                if self?.config?.enableLogging == true {
                    print("Network connection lost")
                }
            }
        }
        
        networkMonitor.start(queue: networkQueue)
    }
    
    // MARK: - Cleanup
    
    deinit {
        stopVoiceSession()
        networkMonitor.cancel()
    }
}

// MARK: - Extensions

extension JarvisClient {
    public func isAuthenticated() -> Bool {
        return authToken != nil && userID != nil
    }
    
    public func getCurrentUserID() -> String? {
        return userID
    }
    
    public func getCurrentSessionID() -> String? {
        return sessionID
    }
    
    public func logout() {
        authToken = nil
        sessionID = nil
        userID = nil
        stopVoiceSession()
    }
}