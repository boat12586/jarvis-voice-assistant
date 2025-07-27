"""
Jarvis v2.0 - Extensible Plugin System
Dynamic plugin loading, management, and execution framework
"""

import os
import sys
import json
import importlib
import inspect
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import hashlib
import uuid

logger = logging.getLogger(__name__)

class PluginStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    LOADING = "loading"
    DISABLED = "disabled"

class PluginType(str, Enum):
    COMMAND = "command"          # Chat commands (/weather, /time, etc.)
    VOICE = "voice"              # Voice processing plugins
    AI = "ai"                    # AI processing/enhancement plugins
    INTEGRATION = "integration"  # External service integrations
    UTILITY = "utility"          # General utility plugins
    MIDDLEWARE = "middleware"    # Request/response processing

class PluginPriority(int, Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20

@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    priority: PluginPriority = PluginPriority.NORMAL
    dependencies: List[str] = None
    permissions: List[str] = None
    config_schema: Dict[str, Any] = None
    min_jarvis_version: str = "2.0.0"
    homepage: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.permissions is None:
            self.permissions = []
        if self.config_schema is None:
            self.config_schema = {}
        if self.tags is None:
            self.tags = []

@dataclass
class PluginConfig:
    """Plugin configuration and settings"""
    enabled: bool = True
    auto_load: bool = True
    settings: Dict[str, Any] = None
    user_permissions: List[str] = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}
        if self.user_permissions is None:
            self.user_permissions = []

@dataclass
class PluginContext:
    """Context passed to plugin methods"""
    user_id: str
    session_id: str
    message: str
    user_data: Dict[str, Any] = None
    session_data: Dict[str, Any] = None
    plugin_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_data is None:
            self.user_data = {}
        if self.session_data is None:
            self.session_data = {}
        if self.plugin_data is None:
            self.plugin_data = {}

@dataclass
class PluginResponse:
    """Response from plugin execution"""
    success: bool
    result: Any = None
    error: str = None
    data: Dict[str, Any] = None
    should_continue: bool = True  # Whether to continue processing other plugins
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class BasePlugin(ABC):
    """Base class for all Jarvis plugins"""
    
    def __init__(self, plugin_manager: 'PluginManager'):
        self.plugin_manager = plugin_manager
        self.logger = logging.getLogger(f"plugin.{self.get_metadata().name}")
        self.config = PluginConfig()
        self._is_initialized = False
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, config: PluginConfig) -> bool:
        """Initialize the plugin with configuration"""
        pass
    
    @abstractmethod
    async def execute(self, context: PluginContext) -> PluginResponse:
        """Execute the plugin's main functionality"""
        pass
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        return True
    
    async def on_user_connected(self, user_id: str, session_id: str):
        """Called when a user connects"""
        pass
    
    async def on_user_disconnected(self, user_id: str, session_id: str):
        """Called when a user disconnects"""
        pass
    
    async def on_message_received(self, context: PluginContext) -> PluginResponse:
        """Called when a message is received (before main processing)"""
        return PluginResponse(success=True, should_continue=True)
    
    async def on_message_processed(self, context: PluginContext, response: str) -> str:
        """Called after message is processed (can modify response)"""
        return response
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.settings.get(key, default)
    
    def set_config_value(self, key: str, value: Any):
        """Set configuration value"""
        self.config.settings[key] = value
    
    def has_permission(self, permission: str) -> bool:
        """Check if plugin has permission"""
        return permission in self.config.user_permissions
    
    async def call_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Call Jarvis API endpoints"""
        return await self.plugin_manager.call_api(endpoint, method, data)
    
    def get_plugin_data_path(self) -> Path:
        """Get path for plugin data storage"""
        plugin_name = self.get_metadata().name
        data_path = Path(f"data/plugins/{plugin_name}")
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path

class CommandPlugin(BasePlugin):
    """Base class for command plugins (e.g., /weather, /time)"""
    
    @abstractmethod
    def get_commands(self) -> List[str]:
        """Return list of commands this plugin handles"""
        pass
    
    @abstractmethod
    async def handle_command(self, command: str, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle a specific command"""
        pass
    
    async def execute(self, context: PluginContext) -> PluginResponse:
        """Parse command and execute appropriate handler"""
        message = context.message.strip()
        if not message.startswith('/'):
            return PluginResponse(success=True, should_continue=True)
        
        parts = message[1:].split()
        if not parts:
            return PluginResponse(success=True, should_continue=True)
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in [cmd.lower() for cmd in self.get_commands()]:
            return await self.handle_command(command, args, context)
        
        return PluginResponse(success=True, should_continue=True)

class PluginManager:
    """Manages loading, execution, and lifecycle of plugins"""
    
    def __init__(self, plugin_dir: str = "plugins", config_dir: str = "config/plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.config_dir = Path(config_dir)
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.plugin_status: Dict[str, PluginStatus] = {}
        self.command_registry: Dict[str, str] = {}  # command -> plugin_name
        self.api_client = None
        
        # Create directories
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Plugin manager initialized: {self.plugin_dir}")
    
    def set_api_client(self, api_client):
        """Set API client for plugin communication"""
        self.api_client = api_client
    
    async def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directory"""
        discovered = []
        
        for plugin_path in self.plugin_dir.iterdir():
            if plugin_path.is_dir() and not plugin_path.name.startswith('_'):
                # Check for main plugin file
                init_file = plugin_path / "__init__.py"
                main_file = plugin_path / "main.py"
                
                if init_file.exists() or main_file.exists():
                    discovered.append(plugin_path.name)
                    logger.debug(f"Discovered plugin: {plugin_path.name}")
        
        return discovered
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin"""
        try:
            self.plugin_status[plugin_name] = PluginStatus.LOADING
            logger.info(f"Loading plugin: {plugin_name}")
            
            # Import plugin module
            plugin_path = self.plugin_dir / plugin_name
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}",
                plugin_path / "main.py" if (plugin_path / "main.py").exists() 
                else plugin_path / "__init__.py"
            )
            
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load plugin spec for {plugin_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and
                    not inspect.isabstract(obj)):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                raise ImportError(f"No valid plugin class found in {plugin_name}")
            
            # Create plugin instance
            plugin_instance = plugin_class(self)
            metadata = plugin_instance.get_metadata()
            
            # Validate metadata
            if not metadata.name:
                raise ValueError(f"Plugin {plugin_name} has no name in metadata")
            
            # Load configuration
            config = await self.load_plugin_config(plugin_name)
            
            # Check if plugin should be loaded
            if not config.enabled:
                logger.info(f"Plugin {plugin_name} is disabled")
                self.plugin_status[plugin_name] = PluginStatus.DISABLED
                return True
            
            # Initialize plugin
            success = await plugin_instance.initialize(config)
            if not success:
                raise RuntimeError(f"Plugin {plugin_name} initialization failed")
            
            # Store plugin
            self.plugins[plugin_name] = plugin_instance
            self.plugin_metadata[plugin_name] = metadata
            self.plugin_configs[plugin_name] = config
            self.plugin_status[plugin_name] = PluginStatus.ACTIVE
            
            # Register commands if it's a command plugin
            if isinstance(plugin_instance, CommandPlugin):
                for command in plugin_instance.get_commands():
                    self.command_registry[command.lower()] = plugin_name
                    logger.debug(f"Registered command: /{command} -> {plugin_name}")
            
            logger.info(f"Plugin loaded successfully: {plugin_name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            logger.debug(traceback.format_exc())
            self.plugin_status[plugin_name] = PluginStatus.ERROR
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin"""
        try:
            if plugin_name not in self.plugins:
                return True
            
            logger.info(f"Unloading plugin: {plugin_name}")
            
            plugin = self.plugins[plugin_name]
            
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from registries
            if isinstance(plugin, CommandPlugin):
                commands_to_remove = []
                for command, registered_plugin in self.command_registry.items():
                    if registered_plugin == plugin_name:
                        commands_to_remove.append(command)
                
                for command in commands_to_remove:
                    del self.command_registry[command]
            
            # Remove plugin
            del self.plugins[plugin_name]
            del self.plugin_metadata[plugin_name]
            del self.plugin_configs[plugin_name]
            self.plugin_status[plugin_name] = PluginStatus.INACTIVE
            
            logger.info(f"Plugin unloaded: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin"""
        await self.unload_plugin(plugin_name)
        return await self.load_plugin(plugin_name)
    
    async def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins"""
        discovered = await self.discover_plugins()
        results = {}
        
        for plugin_name in discovered:
            results[plugin_name] = await self.load_plugin(plugin_name)
        
        logger.info(f"Loaded {sum(results.values())}/{len(results)} plugins")
        return results
    
    async def load_plugin_config(self, plugin_name: str) -> PluginConfig:
        """Load plugin configuration"""
        config_file = self.config_dir / f"{plugin_name}.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                return PluginConfig(**config_data)
            except Exception as e:
                logger.error(f"Failed to load config for {plugin_name}: {e}")
        
        # Return default config
        return PluginConfig()
    
    async def save_plugin_config(self, plugin_name: str, config: PluginConfig) -> bool:
        """Save plugin configuration"""
        try:
            config_file = self.config_dir / f"{plugin_name}.json"
            with open(config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            # Update runtime config
            if plugin_name in self.plugin_configs:
                self.plugin_configs[plugin_name] = config
            
            return True
        except Exception as e:
            logger.error(f"Failed to save config for {plugin_name}: {e}")
            return False
    
    async def execute_plugins(self, context: PluginContext, plugin_type: PluginType = None) -> List[PluginResponse]:
        """Execute plugins for given context"""
        responses = []
        
        # Get plugins to execute
        plugins_to_execute = []
        for plugin_name, plugin in self.plugins.items():
            if self.plugin_status[plugin_name] != PluginStatus.ACTIVE:
                continue
                
            metadata = self.plugin_metadata[plugin_name]
            if plugin_type is None or metadata.plugin_type == plugin_type:
                plugins_to_execute.append((plugin_name, plugin, metadata))
        
        # Sort by priority (highest first)
        plugins_to_execute.sort(key=lambda x: x[2].priority.value, reverse=True)
        
        # Execute plugins
        for plugin_name, plugin, metadata in plugins_to_execute:
            try:
                response = await plugin.execute(context)
                responses.append(response)
                
                # Stop if plugin says not to continue
                if not response.should_continue:
                    logger.debug(f"Plugin {plugin_name} stopped further processing")
                    break
                    
            except Exception as e:
                logger.error(f"Error executing plugin {plugin_name}: {e}")
                responses.append(PluginResponse(
                    success=False,
                    error=str(e),
                    should_continue=True
                ))
        
        return responses
    
    async def execute_command(self, command: str, args: List[str], context: PluginContext) -> PluginResponse:
        """Execute a specific command"""
        command_lower = command.lower()
        
        if command_lower not in self.command_registry:
            return PluginResponse(
                success=False,
                error=f"Unknown command: /{command}"
            )
        
        plugin_name = self.command_registry[command_lower]
        
        if plugin_name not in self.plugins:
            return PluginResponse(
                success=False,
                error=f"Plugin {plugin_name} not loaded"
            )
        
        if self.plugin_status[plugin_name] != PluginStatus.ACTIVE:
            return PluginResponse(
                success=False,
                error=f"Plugin {plugin_name} is not active"
            )
        
        plugin = self.plugins[plugin_name]
        
        if not isinstance(plugin, CommandPlugin):
            return PluginResponse(
                success=False,
                error=f"Plugin {plugin_name} is not a command plugin"
            )
        
        try:
            return await plugin.handle_command(command, args, context)
        except Exception as e:
            logger.error(f"Error executing command /{command}: {e}")
            return PluginResponse(
                success=False,
                error=f"Command execution failed: {str(e)}"
            )
    
    async def call_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Call Jarvis API endpoints for plugins"""
        if self.api_client is None:
            raise RuntimeError("API client not configured")
        
        # TODO: Implement API client calls
        # This would integrate with the FastAPI core service
        return {}
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get information about a plugin"""
        if plugin_name not in self.plugin_metadata:
            return {}
        
        metadata = self.plugin_metadata[plugin_name]
        config = self.plugin_configs.get(plugin_name, PluginConfig())
        status = self.plugin_status.get(plugin_name, PluginStatus.INACTIVE)
        
        return {
            "name": plugin_name,
            "metadata": asdict(metadata),
            "config": asdict(config),
            "status": status.value,
            "loaded": plugin_name in self.plugins
        }
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their info"""
        all_plugins = set(self.plugin_metadata.keys()) | set(self.plugin_status.keys())
        return [self.get_plugin_info(name) for name in sorted(all_plugins)]
    
    def get_commands(self) -> Dict[str, str]:
        """Get all registered commands"""
        return dict(self.command_registry)
    
    async def notify_user_connected(self, user_id: str, session_id: str):
        """Notify all plugins of user connection"""
        for plugin_name, plugin in self.plugins.items():
            if self.plugin_status[plugin_name] == PluginStatus.ACTIVE:
                try:
                    await plugin.on_user_connected(user_id, session_id)
                except Exception as e:
                    logger.error(f"Error in {plugin_name}.on_user_connected: {e}")
    
    async def notify_user_disconnected(self, user_id: str, session_id: str):
        """Notify all plugins of user disconnection"""
        for plugin_name, plugin in self.plugins.items():
            if self.plugin_status[plugin_name] == PluginStatus.ACTIVE:
                try:
                    await plugin.on_user_disconnected(user_id, session_id)
                except Exception as e:
                    logger.error(f"Error in {plugin_name}.on_user_disconnected: {e}")
    
    async def process_message_middleware(self, context: PluginContext) -> PluginContext:
        """Process message through middleware plugins"""
        for plugin_name, plugin in self.plugins.items():
            if (self.plugin_status[plugin_name] == PluginStatus.ACTIVE and
                self.plugin_metadata[plugin_name].plugin_type == PluginType.MIDDLEWARE):
                try:
                    response = await plugin.on_message_received(context)
                    if not response.should_continue:
                        break
                except Exception as e:
                    logger.error(f"Error in {plugin_name} middleware: {e}")
        
        return context
    
    async def process_response_middleware(self, context: PluginContext, response: str) -> str:
        """Process response through middleware plugins"""
        for plugin_name, plugin in self.plugins.items():
            if (self.plugin_status[plugin_name] == PluginStatus.ACTIVE and
                self.plugin_metadata[plugin_name].plugin_type == PluginType.MIDDLEWARE):
                try:
                    response = await plugin.on_message_processed(context, response)
                except Exception as e:
                    logger.error(f"Error in {plugin_name} response middleware: {e}")
        
        return response