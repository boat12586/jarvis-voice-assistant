# Jarvis Mobile SDK v2.0

## Overview

The Jarvis Mobile SDK provides a comprehensive interface for integrating Jarvis Voice Assistant capabilities into iOS and Android applications. This SDK handles authentication, voice processing, command execution, and real-time communication with the Jarvis backend services.

## Features

- **Voice Processing**: Real-time speech-to-text and text-to-speech
- **Command Execution**: Natural language command processing
- **Authentication**: Secure token-based authentication
- **Push Notifications**: Real-time notifications for responses
- **Offline Support**: Basic offline functionality
- **Session Management**: Persistent user sessions
- **Multi-language Support**: Support for multiple languages
- **Biometric Authentication**: Fingerprint and face recognition

## Installation

### iOS (Swift)

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/your-org/jarvis-ios-sdk.git", from: "2.0.0")
]
```

### Android (Kotlin)

Add to your `build.gradle`:

```kotlin
dependencies {
    implementation 'com.jarvis:mobile-sdk:2.0.0'
}
```

### React Native

```bash
npm install @jarvis/react-native-sdk
```

## Quick Start

### iOS Implementation

```swift
import JarvisSDK

class ViewController: UIViewController {
    private let jarvis = JarvisClient()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupJarvis()
    }
    
    private func setupJarvis() {
        // Configure Jarvis
        let config = JarvisConfig(
            baseURL: "https://your-domain.com",
            apiKey: "your-api-key"
        )
        
        jarvis.configure(config: config)
        
        // Authenticate
        jarvis.authenticate(deviceInfo: getDeviceInfo()) { result in
            switch result {
            case .success(let auth):
                print("Authenticated: \(auth.userID)")
                self.startVoiceProcessing()
            case .failure(let error):
                print("Authentication failed: \(error)")
            }
        }
    }
    
    private func startVoiceProcessing() {
        jarvis.startVoiceSession { [weak self] result in
            switch result {
            case .success(let response):
                print("Voice response: \(response.text)")
            case .failure(let error):
                print("Voice processing error: \(error)")
            }
        }
    }
    
    private func getDeviceInfo() -> DeviceInfo {
        return DeviceInfo(
            deviceID: UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString,
            platform: .iOS,
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0",
            osVersion: UIDevice.current.systemVersion
        )
    }
}
```

### Android Implementation

```kotlin
import com.jarvis.sdk.JarvisClient
import com.jarvis.sdk.models.*

class MainActivity : AppCompatActivity() {
    private lateinit var jarvis: JarvisClient
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        setupJarvis()
    }
    
    private fun setupJarvis() {
        // Configure Jarvis
        val config = JarvisConfig(
            baseURL = "https://your-domain.com",
            apiKey = "your-api-key"
        )
        
        jarvis = JarvisClient(config)
        
        // Authenticate
        jarvis.authenticate(getDeviceInfo()) { result ->
            when (result) {
                is Result.Success -> {
                    println("Authenticated: ${result.data.userID}")
                    startVoiceProcessing()
                }
                is Result.Error -> {
                    println("Authentication failed: ${result.error}")
                }
            }
        }
    }
    
    private fun startVoiceProcessing() {
        jarvis.startVoiceSession { result ->
            when (result) {
                is Result.Success -> {
                    println("Voice response: ${result.data.text}")
                }
                is Result.Error -> {
                    println("Voice processing error: ${result.error}")
                }
            }
        }
    }
    
    private fun getDeviceInfo(): DeviceInfo {
        return DeviceInfo(
            deviceID = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID),
            platform = Platform.ANDROID,
            appVersion = BuildConfig.VERSION_NAME,
            osVersion = Build.VERSION.RELEASE
        )
    }
}
```

### React Native Implementation

```javascript
import { JarvisClient } from '@jarvis/react-native-sdk';

class App extends React.Component {
    constructor(props) {
        super(props);
        this.jarvis = new JarvisClient({
            baseURL: 'https://your-domain.com',
            apiKey: 'your-api-key'
        });
    }
    
    componentDidMount() {
        this.setupJarvis();
    }
    
    async setupJarvis() {
        try {
            // Authenticate
            const auth = await this.jarvis.authenticate({
                deviceID: await this.getDeviceID(),
                platform: Platform.OS,
                appVersion: '1.0.0',
                osVersion: Platform.Version
            });
            
            console.log('Authenticated:', auth.userID);
            
            // Start voice processing
            this.startVoiceProcessing();
        } catch (error) {
            console.error('Setup failed:', error);
        }
    }
    
    async startVoiceProcessing() {
        try {
            const response = await this.jarvis.startVoiceSession();
            console.log('Voice response:', response.text);
        } catch (error) {
            console.error('Voice processing error:', error);
        }
    }
    
    async getDeviceID() {
        // Implementation depends on your device ID strategy
        return 'unique-device-id';
    }
}
```

## API Reference

### Authentication

```javascript
// Authenticate with device info
await jarvis.authenticate({
    deviceID: 'unique-device-id',
    platform: 'ios' | 'android',
    appVersion: '1.0.0',
    osVersion: '14.0'
});

// Refresh authentication token
await jarvis.refreshToken();

// Logout
await jarvis.logout();
```

### Voice Processing

```javascript
// Start voice session
const session = await jarvis.startVoiceSession();

// Process voice input
const response = await jarvis.processVoice({
    audioData: base64AudioData,
    language: 'en',
    streaming: true
});

// Text-to-speech
const audioResponse = await jarvis.synthesizeSpeech({
    text: 'Hello, this is Jarvis!',
    voice: 'en-US-AriaNeural'
});

// Stop voice session
await jarvis.stopVoiceSession();
```

### Command Processing

```javascript
// Execute command
const response = await jarvis.executeCommand({
    command: '/weather Bangkok',
    language: 'en',
    sendNotification: true
});

// Get available commands
const commands = await jarvis.getAvailableCommands();
```

### Session Management

```javascript
// Create session
const session = await jarvis.createSession({
    sessionType: 'voice',
    preferences: {
        language: 'en',
        theme: 'dark'
    }
});

// Get session info
const sessionInfo = await jarvis.getSession(sessionId);

// End session
await jarvis.endSession(sessionId);
```

### Push Notifications

```javascript
// Register for notifications
await jarvis.registerForNotifications({
    platform: 'ios',
    pushToken: 'device-push-token'
});

// Handle notification
jarvis.onNotification((notification) => {
    console.log('Received notification:', notification);
});
```

### Offline Support

```javascript
// Enable offline mode
await jarvis.enableOfflineMode();

// Queue requests for later sync
await jarvis.queueRequest({
    type: 'command',
    data: { command: '/time' }
});

// Sync queued requests
await jarvis.syncOfflineRequests();
```

## Configuration

### Basic Configuration

```javascript
const config = {
    baseURL: 'https://your-domain.com',
    apiKey: 'your-api-key',
    timeout: 30000,
    retryAttempts: 3,
    enableLogging: true
};
```

### Advanced Configuration

```javascript
const config = {
    baseURL: 'https://your-domain.com',
    apiKey: 'your-api-key',
    
    // Audio settings
    audio: {
        sampleRate: 16000,
        channels: 1,
        enableNoiseReduction: true,
        enableVAD: true
    },
    
    // Voice settings
    voice: {
        defaultVoice: 'en-US-AriaNeural',
        language: 'en',
        autoDetectLanguage: true
    },
    
    // Network settings
    network: {
        timeout: 30000,
        retryAttempts: 3,
        retryDelay: 1000
    },
    
    // Features
    features: {
        pushNotifications: true,
        offlineMode: true,
        biometricAuth: true,
        voiceCommands: true
    }
};
```

## Error Handling

```javascript
try {
    const response = await jarvis.executeCommand({
        command: '/weather Bangkok'
    });
} catch (error) {
    switch (error.code) {
        case 'AUTHENTICATION_FAILED':
            // Handle authentication error
            await jarvis.refreshToken();
            break;
        case 'NETWORK_ERROR':
            // Handle network error
            await jarvis.retryRequest();
            break;
        case 'VOICE_PROCESSING_ERROR':
            // Handle voice processing error
            console.error('Voice processing failed:', error.message);
            break;
        default:
            console.error('Unknown error:', error);
    }
}
```

## Event Handling

```javascript
// Voice events
jarvis.on('voiceStarted', () => {
    console.log('Voice processing started');
});

jarvis.on('voiceEnded', (result) => {
    console.log('Voice processing ended:', result);
});

jarvis.on('speechRecognized', (text) => {
    console.log('Speech recognized:', text);
});

// Connection events
jarvis.on('connected', () => {
    console.log('Connected to Jarvis');
});

jarvis.on('disconnected', () => {
    console.log('Disconnected from Jarvis');
});

// Error events
jarvis.on('error', (error) => {
    console.error('Jarvis error:', error);
});
```

## Best Practices

### 1. Error Handling
Always implement proper error handling for network requests and voice processing:

```javascript
const handleJarvisError = (error) => {
    switch (error.code) {
        case 'AUTHENTICATION_FAILED':
            // Redirect to login
            break;
        case 'NETWORK_ERROR':
            // Show offline message
            break;
        case 'VOICE_PROCESSING_ERROR':
            // Show voice error message
            break;
    }
};
```

### 2. Resource Management
Properly manage audio resources and sessions:

```javascript
// Clean up resources
componentWillUnmount() {
    jarvis.stopVoiceSession();
    jarvis.disconnect();
}
```

### 3. Performance Optimization
Optimize for mobile performance:

```javascript
// Use streaming for real-time processing
const config = {
    audio: {
        streaming: true,
        chunkSize: 1024
    }
};

// Implement caching
jarvis.enableCaching({
    maxCacheSize: 50 * 1024 * 1024, // 50MB
    ttl: 3600 // 1 hour
});
```

### 4. Security
Implement proper security measures:

```javascript
// Enable biometric authentication
jarvis.enableBiometricAuth({
    title: 'Authenticate with Jarvis',
    subtitle: 'Use your fingerprint or face ID'
});

// Secure token storage
jarvis.setSecureStorage(true);
```

## Testing

### Unit Tests
```javascript
describe('Jarvis SDK', () => {
    let jarvis;
    
    beforeEach(() => {
        jarvis = new JarvisClient(mockConfig);
    });
    
    it('should authenticate successfully', async () => {
        const auth = await jarvis.authenticate(mockDeviceInfo);
        expect(auth.success).toBe(true);
    });
    
    it('should process voice input', async () => {
        const response = await jarvis.processVoice(mockAudioData);
        expect(response.text).toBeDefined();
    });
});
```

### Integration Tests
```javascript
describe('Jarvis Integration', () => {
    it('should complete full voice workflow', async () => {
        // Authenticate
        await jarvis.authenticate(deviceInfo);
        
        // Start voice session
        const session = await jarvis.startVoiceSession();
        
        // Process voice
        const response = await jarvis.processVoice(audioData);
        
        // Verify response
        expect(response.success).toBe(true);
        expect(response.text).toBeDefined();
    });
});
```

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Check API key and base URL
   - Verify device info is correct
   - Ensure network connectivity

2. **Voice Processing Issues**
   - Check microphone permissions
   - Verify audio format compatibility
   - Test with different sample rates

3. **Network Connectivity**
   - Implement retry logic
   - Check firewall settings
   - Verify SSL certificates

### Debug Mode
Enable debug mode for detailed logging:

```javascript
const config = {
    enableLogging: true,
    logLevel: 'debug'
};
```

## Support

- **Documentation**: [docs.jarvis.com](https://docs.jarvis.com)
- **API Reference**: [api.jarvis.com](https://api.jarvis.com)
- **Support**: [support@jarvis.com](mailto:support@jarvis.com)
- **GitHub Issues**: [github.com/jarvis/mobile-sdk/issues](https://github.com/jarvis/mobile-sdk/issues)

## License

This SDK is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Version**: 2.0.0  
**Last Updated**: July 18, 2025  
**Compatibility**: iOS 13+, Android 8+, React Native 0.68+