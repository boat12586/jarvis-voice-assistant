"""
Greetings Plugin for Jarvis v2.0
Smart greeting and courtesy responses based on user context
"""

import random
from datetime import datetime
from typing import List, Dict, Any
from plugin_system import BasePlugin, PluginMetadata, PluginType, PluginPriority, PluginContext, PluginResponse, PluginConfig

class GreetingsPlugin(BasePlugin):
    """Intelligent greeting and courtesy plugin"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="greetings",
            version="1.0.0",
            description="Provides intelligent greetings and courtesy responses",
            author="Jarvis Team",
            plugin_type=PluginType.MIDDLEWARE,
            priority=PluginPriority.HIGH,
            dependencies=[],
            permissions=[],
            config_schema={
                "enabled": {
                    "type": "boolean",
                    "description": "Enable greeting responses",
                    "default": True
                },
                "personalized": {
                    "type": "boolean",
                    "description": "Use personalized greetings",
                    "default": True
                },
                "time_based": {
                    "type": "boolean",
                    "description": "Use time-based greetings",
                    "default": True
                }
            },
            tags=["greetings", "courtesy", "social", "middleware"]
        )
    
    def __init__(self, plugin_manager):
        super().__init__(plugin_manager)
        
        # Greeting patterns and responses
        self.greetings = {
            'hello': ['hello', 'hi', 'hey', 'greetings', 'good day'],
            'good_morning': ['good morning', 'morning'],
            'good_afternoon': ['good afternoon', 'afternoon'],
            'good_evening': ['good evening', 'evening'],
            'good_night': ['good night', 'goodnight', 'night'],
            'how_are_you': ['how are you', 'how do you do', 'how\'s it going', 'what\'s up'],
            'thank_you': ['thank you', 'thanks', 'thank u', 'thx', 'appreciated'],
            'goodbye': ['goodbye', 'bye', 'see you later', 'farewell', 'catch you later'],
            'please': ['please', 'pls'],
            'sorry': ['sorry', 'apologize', 'my bad']
        }
        
        # Response templates
        self.responses = {
            'hello': [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
                "Greetings! I'm here to help.",
                "Hello! Ready to assist you.",
            ],
            'good_morning': [
                "Good morning! Hope you're having a great start to your day!",
                "Morning! What can I help you with today?",
                "Good morning! Ready to tackle the day together?",
            ],
            'good_afternoon': [
                "Good afternoon! How's your day going?",
                "Afternoon! What can I assist you with?",
                "Good afternoon! Hope you're having a productive day!",
            ],
            'good_evening': [
                "Good evening! How can I help you tonight?",
                "Evening! What brings you here?",
                "Good evening! Ready to assist you.",
            ],
            'good_night': [
                "Good night! Sleep well!",
                "Sweet dreams! See you tomorrow.",
                "Good night! Rest well!",
            ],
            'how_are_you': [
                "I'm doing great, thank you for asking! How are you?",
                "I'm functioning perfectly! How about you?",
                "All systems running smoothly! How are you doing?",
                "I'm excellent! How are things with you?",
            ],
            'thank_you': [
                "You're welcome! Happy to help!",
                "My pleasure! Anything else I can do for you?",
                "Glad I could help!",
                "You're very welcome!",
            ],
            'goodbye': [
                "Goodbye! Take care!",
                "See you later! Have a great day!",
                "Farewell! Don't hesitate to come back if you need anything.",
                "Bye! It was great helping you!",
            ],
            'please': [
                "Of course! I'd be happy to help.",
                "Certainly! What do you need?",
                "Absolutely! How can I assist?",
            ],
            'sorry': [
                "No worries at all!",
                "That's perfectly fine!",
                "No problem whatsoever!",
                "Don't worry about it!",
            ]
        }
        
        # User interaction tracking
        self.user_interactions = {}
    
    async def initialize(self, config: PluginConfig) -> bool:
        """Initialize greetings plugin"""
        try:
            self.config = config
            self.enabled = self.get_config_value("enabled", True)
            self.personalized = self.get_config_value("personalized", True)
            self.time_based = self.get_config_value("time_based", True)
            
            if not self.enabled:
                self.logger.info("Greetings plugin disabled by configuration")
                return True
            
            self.logger.info("Greetings plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize greetings plugin: {e}")
            return False
    
    async def execute(self, context: PluginContext) -> PluginResponse:
        """Main execution method (not used for middleware)"""
        return PluginResponse(success=True, should_continue=True)
    
    async def on_message_received(self, context: PluginContext) -> PluginResponse:
        """Process incoming messages for greetings"""
        if not self.enabled:
            return PluginResponse(success=True, should_continue=True)
        
        message = context.message.lower().strip()
        user_id = context.user_id
        
        # Update user interaction tracking
        self.update_user_interaction(user_id, message)
        
        # Check for greeting patterns
        greeting_type = self.detect_greeting(message)
        
        if greeting_type:
            response = self.generate_greeting_response(greeting_type, context)
            
            return PluginResponse(
                success=True,
                result=response,
                data={
                    "greeting_type": greeting_type,
                    "personalized": self.personalized,
                    "time_based": self.time_based
                },
                should_continue=False  # Greeting should be the response
            )
        
        return PluginResponse(success=True, should_continue=True)
    
    async def on_user_connected(self, user_id: str, session_id: str):
        """Welcome new user connections"""
        if not self.enabled:
            return
        
        # Initialize user interaction tracking
        self.user_interactions[user_id] = {
            'first_interaction': datetime.now(),
            'last_interaction': datetime.now(),
            'interaction_count': 0,
            'session_id': session_id
        }
        
        self.logger.info(f"User {user_id} connected - greetings initialized")
    
    async def on_user_disconnected(self, user_id: str, session_id: str):
        """Handle user disconnection"""
        if user_id in self.user_interactions:
            # Keep some interaction history but remove detailed tracking
            del self.user_interactions[user_id]
        
        self.logger.info(f"User {user_id} disconnected - greetings cleaned up")
    
    def detect_greeting(self, message: str) -> str:
        """Detect greeting type in message"""
        message = message.lower().strip()
        
        # Remove punctuation for better matching
        cleaned_message = ''.join(char for char in message if char.isalnum() or char.isspace())
        
        # Check each greeting type
        for greeting_type, patterns in self.greetings.items():
            for pattern in patterns:
                if pattern in cleaned_message:
                    return greeting_type
        
        return None
    
    def generate_greeting_response(self, greeting_type: str, context: PluginContext) -> str:
        """Generate appropriate greeting response"""
        base_responses = self.responses.get(greeting_type, ["Hello!"])
        
        # Add time-based context if enabled
        if self.time_based and greeting_type == 'hello':
            hour = datetime.now().hour
            if 5 <= hour < 12:
                base_responses = self.responses['good_morning']
            elif 12 <= hour < 17:
                base_responses = self.responses['good_afternoon']
            elif 17 <= hour < 22:
                base_responses = self.responses['good_evening']
            else:
                base_responses = self.responses['good_night']
        
        # Select random response
        response = random.choice(base_responses)
        
        # Add personalization if enabled
        if self.personalized and context.user_id in self.user_interactions:
            user_data = self.user_interactions[context.user_id]
            interaction_count = user_data['interaction_count']
            
            if interaction_count == 0:
                # First interaction
                response = f"Welcome! {response}"
            elif interaction_count < 5:
                # Early interactions
                response = f"Good to see you again! {response}"
            else:
                # Regular user
                response = f"Hello again! {response}"
        
        # Add emoji for friendliness
        emoji_map = {
            'hello': '👋',
            'good_morning': '🌅',
            'good_afternoon': '☀️',
            'good_evening': '🌆',
            'good_night': '🌙',
            'how_are_you': '😊',
            'thank_you': '😊',
            'goodbye': '👋',
            'please': '😊',
            'sorry': '😊'
        }
        
        emoji = emoji_map.get(greeting_type, '😊')
        return f"{emoji} {response}"
    
    def update_user_interaction(self, user_id: str, message: str):
        """Update user interaction tracking"""
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = {
                'first_interaction': datetime.now(),
                'last_interaction': datetime.now(),
                'interaction_count': 0
            }
        
        user_data = self.user_interactions[user_id]
        user_data['last_interaction'] = datetime.now()
        user_data['interaction_count'] += 1
        
        # Keep interaction history manageable
        if user_data['interaction_count'] > 1000:
            user_data['interaction_count'] = 100  # Reset to reasonable number
    
    async def cleanup(self) -> bool:
        """Cleanup greetings plugin resources"""
        self.user_interactions.clear()
        self.logger.info("Greetings plugin cleaned up")
        return True