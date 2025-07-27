// MongoDB initialization script for Jarvis v2.0
// This script sets up the database structure and initial data

db = db.getSiblingDB('jarvis_v2');

// Create collections
db.createCollection('users');
db.createCollection('sessions');
db.createCollection('conversations');
db.createCollection('plugins');
db.createCollection('audio_sessions');
db.createCollection('system_logs');

// Create indexes for performance
db.users.createIndex({ "user_id": 1 }, { unique: true });
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true, sparse: true });

db.sessions.createIndex({ "session_id": 1 }, { unique: true });
db.sessions.createIndex({ "user_id": 1 });
db.sessions.createIndex({ "created_at": 1 });

db.conversations.createIndex({ "session_id": 1 });
db.conversations.createIndex({ "user_id": 1 });
db.conversations.createIndex({ "timestamp": 1 });

db.plugins.createIndex({ "plugin_name": 1 }, { unique: true });
db.plugins.createIndex({ "plugin_type": 1 });

db.audio_sessions.createIndex({ "session_id": 1 }, { unique: true });
db.audio_sessions.createIndex({ "user_id": 1 });
db.audio_sessions.createIndex({ "created_at": 1 });

db.system_logs.createIndex({ "timestamp": 1 });
db.system_logs.createIndex({ "level": 1 });
db.system_logs.createIndex({ "service": 1 });

// Insert default admin user
db.users.insertOne({
  user_id: "admin",
  username: "admin",
  email: "admin@jarvis.local",
  role: "admin",
  status: "active",
  created_at: new Date(),
  updated_at: new Date(),
  preferences: {
    language: "en",
    theme: "dark",
    voice_settings: {
      voice: "en-US-AriaNeural",
      rate: "+0%",
      volume: "+0%"
    }
  },
  password_hash: "$2b$12$dummy_hash_for_admin_user", // Change in production
  last_login: null,
  login_count: 0
});

// Insert default plugins configuration
db.plugins.insertMany([
  {
    plugin_name: "weather",
    plugin_type: "command",
    version: "1.0.0",
    enabled: true,
    auto_load: true,
    settings: {
      api_key: "your_openweathermap_api_key",
      default_city: "Bangkok",
      units: "metric"
    },
    permissions: ["network"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    plugin_name: "time",
    plugin_type: "command",
    version: "1.0.0",
    enabled: true,
    auto_load: true,
    settings: {
      default_timezone: "UTC",
      date_format: "%Y-%m-%d",
      time_format: "%H:%M:%S"
    },
    permissions: [],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    plugin_name: "calculator",
    plugin_type: "command",
    version: "1.0.0",
    enabled: true,
    auto_load: true,
    settings: {
      precision: 10,
      allow_functions: true
    },
    permissions: [],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    plugin_name: "greetings",
    plugin_type: "middleware",
    version: "1.0.0",
    enabled: true,
    auto_load: true,
    settings: {
      enabled: true,
      personalized: true,
      time_based: true
    },
    permissions: [],
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Insert system configuration
db.system_config.insertOne({
  config_id: "global",
  version: "2.0.0",
  settings: {
    max_sessions_per_user: 10,
    session_timeout: 3600,
    max_audio_duration: 300,
    supported_languages: ["en", "th", "es", "fr", "de", "ja", "ko", "zh"],
    default_language: "en",
    audio_settings: {
      sample_rate: 16000,
      channels: 1,
      chunk_size: 1024,
      supported_formats: ["wav", "mp3", "flac"]
    },
    security: {
      jwt_expiry: 86400,
      max_login_attempts: 5,
      password_min_length: 8,
      enable_2fa: false
    }
  },
  created_at: new Date(),
  updated_at: new Date()
});

print("Jarvis v2.0 MongoDB initialization completed successfully!");
print("Collections created: users, sessions, conversations, plugins, audio_sessions, system_logs");
print("Default admin user created with username: admin");
print("Default plugins configuration loaded");
print("System configuration initialized");