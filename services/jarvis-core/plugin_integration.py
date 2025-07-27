"""
Plugin Integration for Jarvis v2.0 Core Service
Integrates the plugin system with FastAPI core service
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from datetime import datetime

# Add plugin system to path
plugin_path = Path(__file__).parent.parent / "jarvis-plugins"
sys.path.insert(0, str(plugin_path))

from plugin_system import PluginManager, PluginContext, PluginResponse, PluginType, PluginStatus
from models import ChatRequest, ChatResponse, WebSocketResponse

logger = logging.getLogger(__name__)

class JarvisPluginManager:
    """Enhanced plugin manager integrated with Jarvis core service"""
    
    def __init__(self, user_manager, connection_manager, redis_client):
        self.user_manager = user_manager
        self.connection_manager = connection_manager
        self.redis_client = redis_client
        
        # Initialize plugin manager
        plugins_dir = plugin_path / "plugins"
        config_dir = Path("config/plugins")
        
        self.plugin_manager = PluginManager(
            plugin_dir=str(plugins_dir),
            config_dir=str(config_dir)
        )
        
        # Set up API client for plugins
        self.plugin_manager.set_api_client(self)
        
        self.logger = logging.getLogger(__name__)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize plugin system"""
        try:
            if self._initialized:
                return True
            
            self.logger.info("Initializing Jarvis plugin system...")
            
            # Load all plugins
            results = await self.plugin_manager.load_all_plugins()
            
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            self.logger.info(f"Plugin system initialized: {successful}/{total} plugins loaded")
            
            if successful > 0:
                # Log loaded plugins
                for plugin_name, success in results.items():
                    if success:
                        plugin_info = self.plugin_manager.get_plugin_info(plugin_name)
                        metadata = plugin_info.get('metadata', {})
                        self.logger.info(f"  ✅ {plugin_name} v{metadata.get('version', '?')} - {metadata.get('description', '')}")
                    else:
                        self.logger.warning(f"  ❌ {plugin_name} - Failed to load")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin system: {e}")
            return False
    
    async def process_chat_message(self, request: ChatRequest, base_response: str) -> ChatResponse:
        """Process chat message through plugins"""
        try:
            if not self._initialized:
                self.logger.warning("Plugin system not initialized, skipping plugin processing")
                return None
            
            # Create plugin context
            context = PluginContext(
                user_id=request.user_id,
                session_id=request.session_id or "",
                message=request.message,
                user_data=await self.get_user_context(request.user_id),
                session_data=await self.get_session_context(request.session_id),
                plugin_data={}
            )
            
            # Process through middleware plugins first
            context = await self.plugin_manager.process_message_middleware(context)
            
            # Check if message is a command
            if request.message.strip().startswith('/'):
                return await self.handle_command(request, context)
            
            # Execute relevant plugins
            responses = await self.plugin_manager.execute_plugins(context, PluginType.AI)
            
            # Check if any plugin provided a response
            for response in responses:
                if response.success and response.result:
                    # Process response through middleware
                    final_response = await self.plugin_manager.process_response_middleware(
                        context, response.result
                    )
                    
                    # Create enhanced chat response
                    return ChatResponse(
                        response=final_response,
                        session_id=request.session_id or context.session_id,
                        user_id=request.user_id,
                        timestamp=datetime.now(),
                        confidence=0.95,
                        processing_time=0.1,
                        language=request.language or "en",
                        metadata={
                            "plugin_processed": True,
                            "plugin_data": response.data,
                            "processing_plugins": len(responses)
                        }
                    )
            
            # No plugin handled the message, return base response
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing message through plugins: {e}")
            return None
    
    async def handle_command(self, request: ChatRequest, context: PluginContext) -> ChatResponse:
        """Handle command execution through plugins"""
        try:
            message = request.message.strip()
            if not message.startswith('/'):
                return None
            
            # Parse command
            parts = message[1:].split()
            if not parts:
                return None
            
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Execute command through plugin system
            response = await self.plugin_manager.execute_command(command, args, context)
            
            if response.success and response.result:
                # Process response through middleware
                final_response = await self.plugin_manager.process_response_middleware(
                    context, response.result
                )
                
                return ChatResponse(
                    response=final_response,
                    session_id=request.session_id or context.session_id,
                    user_id=request.user_id,
                    timestamp=datetime.now(),
                    confidence=1.0,  # Commands have high confidence
                    processing_time=0.1,
                    language=request.language or "en",
                    metadata={
                        "command": command,
                        "args": args,
                        "plugin_data": response.data,
                        "command_response": True
                    }
                )
            elif not response.success:
                # Command failed
                return ChatResponse(
                    response=f"❌ {response.error}",
                    session_id=request.session_id or context.session_id,
                    user_id=request.user_id,
                    timestamp=datetime.now(),
                    confidence=1.0,
                    processing_time=0.1,
                    language=request.language or "en",
                    metadata={
                        "command": command,
                        "args": args,
                        "error": response.error,
                        "command_response": True
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error handling command: {e}")
            return ChatResponse(
                response=f"❌ Command execution failed: {str(e)}",
                session_id=request.session_id or context.session_id,
                user_id=request.user_id,
                timestamp=datetime.now(),
                confidence=1.0,
                processing_time=0.1,
                language=request.language or "en",
                metadata={
                    "error": str(e),
                    "command_response": True
                }
            )
    
    async def notify_user_connected(self, user_id: str, session_id: str):
        """Notify plugins of user connection"""
        try:
            if self._initialized:
                await self.plugin_manager.notify_user_connected(user_id, session_id)
        except Exception as e:
            self.logger.error(f"Error notifying plugins of user connection: {e}")
    
    async def notify_user_disconnected(self, user_id: str, session_id: str):
        """Notify plugins of user disconnection"""
        try:
            if self._initialized:
                await self.plugin_manager.notify_user_disconnected(user_id, session_id)
        except Exception as e:
            self.logger.error(f"Error notifying plugins of user disconnection: {e}")
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context for plugins"""
        try:
            user = self.user_manager.get_user(user_id) if user_id else None
            if user:
                return {
                    "username": user.username,
                    "role": user.role.value,
                    "preferences": user.preferences,
                    "created_at": user.created_at.isoformat()
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting user context: {e}")
            return {}
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context for plugins"""
        try:
            if not session_id:
                return {}
            
            session = self.user_manager.get_session(session_id)
            if session:
                return {
                    "session_type": session.session_type.value,
                    "created_at": session.created_at.isoformat(),
                    "conversation_history": session.conversation_history[-10:],  # Last 10 messages
                    "voice_settings": session.voice_settings,
                    "ui_preferences": session.ui_preferences
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting session context: {e}")
            return {}
    
    # Plugin Management API Methods
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins"""
        return self.plugin_manager.list_plugins()
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get plugin information"""
        return self.plugin_manager.get_plugin_info(plugin_name)
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin"""
        return await self.plugin_manager.load_plugin(plugin_name)
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin"""
        return await self.plugin_manager.unload_plugin(plugin_name)
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin"""
        return await self.plugin_manager.reload_plugin(plugin_name)
    
    async def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Update plugin configuration"""
        try:
            from plugin_system import PluginConfig
            plugin_config = PluginConfig(**config)
            return await self.plugin_manager.save_plugin_config(plugin_name, plugin_config)
        except Exception as e:
            self.logger.error(f"Error updating plugin config: {e}")
            return False
    
    def get_available_commands(self) -> Dict[str, str]:
        """Get all available commands"""
        return self.plugin_manager.get_commands()
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        plugins = self.list_plugins()
        
        stats = {
            "total_plugins": len(plugins),
            "active_plugins": len([p for p in plugins if p.get("status") == "active"]),
            "inactive_plugins": len([p for p in plugins if p.get("status") == "inactive"]),
            "error_plugins": len([p for p in plugins if p.get("status") == "error"]),
            "total_commands": len(self.get_available_commands()),
            "plugin_types": {}
        }
        
        # Count by type
        for plugin in plugins:
            plugin_type = plugin.get("metadata", {}).get("plugin_type", "unknown")
            stats["plugin_types"][plugin_type] = stats["plugin_types"].get(plugin_type, 0) + 1
        
        return stats
    
    # API Client Implementation for Plugins
    async def call_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """API client for plugins to call Jarvis services"""
        try:
            # This would implement actual API calls to other Jarvis services
            # For now, return mock data
            
            if endpoint.startswith("/api/v2/users/"):
                # User info request
                user_id = endpoint.split("/")[-1]
                user = self.user_manager.get_user(user_id)
                if user:
                    return {
                        "user_id": user.user_id,
                        "username": user.username,
                        "role": user.role.value,
                        "status": user.status.value
                    }
            
            elif endpoint.startswith("/api/v2/sessions/"):
                # Session info request
                session_id = endpoint.split("/")[-1]
                session = self.user_manager.get_session(session_id)
                if session:
                    return {
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "status": session.status.value,
                        "created_at": session.created_at.isoformat()
                    }
            
            return {"error": "Endpoint not found"}
            
        except Exception as e:
            self.logger.error(f"Plugin API call error: {e}")
            return {"error": str(e)}