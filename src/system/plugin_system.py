"""
Plugin System Architecture for JARVIS Voice Assistant
Extensible plugin framework for adding new features and capabilities
"""

import os
import sys
import json
import importlib
import inspect
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Union, Type
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio

# Enhanced logging support
try:
    from src.system.enhanced_logger import ComponentLogger
    ENHANCED_LOGGING = True
except ImportError:
    ENHANCED_LOGGING = False


@dataclass
class PluginMetadata:
    """Plugin metadata structure"""
    name: str
    version: str
    description: str
    author: str
    category: str
    dependencies: List[str]
    permissions: List[str]
    api_version: str = "1.0"
    enabled: bool = True
    priority: int = 5  # 1 = highest priority


@dataclass
class PluginEvent:
    """Plugin event structure"""
    event_type: str
    data: Dict[str, Any]
    source: str
    timestamp: float
    handled: bool = False


class PluginInterface(ABC):
    """Base interface for all JARVIS plugins"""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger(f"plugin.{metadata.name}")
        self.is_active = False
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    async def initialize(self, jarvis_api: 'JarvisAPI') -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        pass
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def get_event_handlers(self, event_type: str) -> List[Callable]:
        """Get handlers for an event type"""
        return self._event_handlers.get(event_type, [])
    
    async def handle_event(self, event: PluginEvent) -> bool:
        """Handle an event"""
        handlers = self.get_event_handlers(event.event_type)
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                event.handled = True
            except Exception as e:
                self.logger.error(f"Event handler failed: {e}")
        
        return event.handled
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities"""
        return []
    
    def get_commands(self) -> Dict[str, Callable]:
        """Get voice commands handled by this plugin"""
        return {}
    
    def get_web_routes(self) -> Dict[str, Callable]:
        """Get web API routes provided by this plugin"""
        return {}


class VoicePluginInterface(PluginInterface):
    """Interface for voice-enabled plugins"""
    
    @abstractmethod
    async def handle_voice_command(self, command: str, context: Dict[str, Any]) -> Optional[str]:
        """Handle a voice command"""
        pass
    
    def get_wake_phrases(self) -> List[str]:
        """Get additional wake phrases this plugin recognizes"""
        return []
    
    def get_voice_responses(self) -> Dict[str, str]:
        """Get predefined voice responses"""
        return {}


class AIPluginInterface(PluginInterface):
    """Interface for AI-enhanced plugins"""
    
    @abstractmethod
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an AI query"""
        pass
    
    def get_model_requirements(self) -> Dict[str, str]:
        """Get required AI models"""
        return {}
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Get AI prompt templates"""
        return {}


class JarvisAPI:
    """API interface for plugins to interact with JARVIS"""
    
    def __init__(self, core_system):
        self.core = core_system
        self.logger = logging.getLogger("jarvis.api")
    
    # Voice API
    async def speak(self, text: str, priority: int = 5) -> bool:
        """Make JARVIS speak"""
        try:
            if hasattr(self.core, 'tts_engine'):
                return await self.core.tts_engine.speak(text)
            return False
        except Exception as e:
            self.logger.error(f"Speak API failed: {e}")
            return False
    
    async def listen(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for voice input"""
        try:
            if hasattr(self.core, 'voice_controller'):
                return await self.core.voice_controller.listen(timeout)
            return None
        except Exception as e:
            self.logger.error(f"Listen API failed: {e}")
            return None
    
    # AI API
    async def query_ai(self, prompt: str, context: Optional[Dict] = None) -> Optional[str]:
        """Query the AI engine"""
        try:
            if hasattr(self.core, 'ai_engine'):
                return await self.core.ai_engine.process_query(prompt, context or {})
            return None
        except Exception as e:
            self.logger.error(f"AI query failed: {e}")
            return None
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add knowledge to the RAG system"""
        try:
            if hasattr(self.core, 'rag_system'):
                return self.core.rag_system.add_document(content, metadata)
            return False
        except Exception as e:
            self.logger.error(f"Add knowledge failed: {e}")
            return False
    
    # Event API
    def emit_event(self, event_type: str, data: Dict[str, Any], source: str = "plugin"):
        """Emit an event"""
        event = PluginEvent(
            event_type=event_type,
            data=data,
            source=source,
            timestamp=time.time()
        )
        
        if hasattr(self.core, 'plugin_manager'):
            asyncio.create_task(self.core.plugin_manager.handle_event(event))
    
    # Configuration API
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            if hasattr(self.core, 'config'):
                return self.core.config.get(key, default)
            return default
        except Exception:
            return default
    
    def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            if hasattr(self.core, 'config'):
                self.core.config[key] = value
                return True
            return False
        except Exception:
            return False
    
    # Storage API
    def get_plugin_data_dir(self, plugin_name: str) -> Path:
        """Get plugin data directory"""
        data_dir = Path("data/plugins") / plugin_name
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def save_plugin_data(self, plugin_name: str, filename: str, data: Any) -> bool:
        """Save plugin data"""
        try:
            data_dir = self.get_plugin_data_dir(plugin_name)
            file_path = data_dir / filename
            
            if isinstance(data, (dict, list)):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            
            return True
        except Exception as e:
            self.logger.error(f"Save plugin data failed: {e}")
            return False
    
    def load_plugin_data(self, plugin_name: str, filename: str) -> Optional[Any]:
        """Load plugin data"""
        try:
            data_dir = self.get_plugin_data_dir(plugin_name)
            file_path = data_dir / filename
            
            if not file_path.exists():
                return None
            
            if filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            self.logger.error(f"Load plugin data failed: {e}")
            return None


class PluginLoader:
    """Plugin loading and management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_modules: Dict[str, Any] = {}
        
    def discover_plugins(self, plugin_dirs: List[str]) -> List[Dict[str, Any]]:
        """Discover available plugins"""
        discovered = []
        
        for plugin_dir in plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
            
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                if os.path.isdir(item_path):
                    # Check for plugin.json
                    manifest_path = os.path.join(item_path, 'plugin.json')
                    if os.path.exists(manifest_path):
                        try:
                            with open(manifest_path, 'r', encoding='utf-8') as f:
                                manifest = json.load(f)
                            
                            manifest['path'] = item_path
                            discovered.append(manifest)
                            
                        except Exception as e:
                            self.logger.error(f"Failed to read plugin manifest {manifest_path}: {e}")
        
        return discovered
    
    async def load_plugin(self, plugin_path: str, manifest: Dict[str, Any]) -> Optional[PluginInterface]:
        """Load a single plugin"""
        try:
            plugin_name = manifest['name']
            self.logger.info(f"Loading plugin: {plugin_name}")
            
            # Create metadata
            metadata = PluginMetadata(**{
                k: v for k, v in manifest.items() 
                if k in PluginMetadata.__annotations__
            })
            
            # Add plugin path to Python path
            if plugin_path not in sys.path:
                sys.path.insert(0, plugin_path)
            
            # Import plugin module
            main_module = manifest.get('main', 'main.py')
            module_name = os.path.splitext(main_module)[0]
            
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_name}", 
                os.path.join(plugin_path, main_module)
            )
            
            if not spec or not spec.loader:
                raise ImportError(f"Could not load plugin module: {main_module}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj is not PluginInterface):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                raise ImportError(f"No plugin class found in {main_module}")
            
            # Instantiate plugin
            plugin_instance = plugin_class(metadata)
            
            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_modules[plugin_name] = module
            
            self.logger.info(f"✅ Plugin loaded: {plugin_name}")
            return plugin_instance
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        try:
            if plugin_name in self.loaded_plugins:
                plugin = self.loaded_plugins[plugin_name]
                
                # Cleanup plugin
                if asyncio.iscoroutinefunction(plugin.cleanup):
                    asyncio.create_task(plugin.cleanup())
                else:
                    plugin.cleanup()
                
                # Remove from loaded plugins
                del self.loaded_plugins[plugin_name]
                
                # Remove module
                if plugin_name in self.plugin_modules:
                    del self.plugin_modules[plugin_name]
                
                self.logger.info(f"Unloaded plugin: {plugin_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False


class PluginManager:
    """Main plugin management system"""
    
    def __init__(self, core_system, config: Optional[Dict[str, Any]] = None):
        self.core = core_system
        self.config = config or {}
        
        # Setup logging
        if ENHANCED_LOGGING:
            self.logger = ComponentLogger("plugin_manager", self.config)
        else:
            self.logger = logging.getLogger(__name__)
        
        # Components
        self.api = JarvisAPI(core_system)
        self.loader = PluginLoader()
        
        # Plugin management
        self.plugins: Dict[str, PluginInterface] = {}
        self.enabled_plugins: Dict[str, bool] = {}
        self.plugin_categories: Dict[str, List[str]] = {}
        
        # Event system
        self.event_queue = asyncio.Queue()
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._event_processor_task = None
        
        # Plugin directories
        self.plugin_dirs = [
            "plugins",
            "src/plugins", 
            os.path.expanduser("~/.jarvis/plugins")
        ]
        
        # Performance tracking
        self.plugin_stats: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize plugin manager"""
        try:
            operation_id = None
            if ENHANCED_LOGGING:
                operation_id = self.logger.operation_start("initialize_plugin_manager")
            
            self.logger.info("🔌 Initializing Plugin Manager...")
            
            # Create plugin directories
            for plugin_dir in self.plugin_dirs:
                Path(plugin_dir).mkdir(parents=True, exist_ok=True)
            
            # Discover plugins
            discovered_plugins = self.loader.discover_plugins(self.plugin_dirs)
            self.logger.info(f"Discovered {len(discovered_plugins)} plugins")
            
            # Load plugins
            loaded_count = 0
            for plugin_manifest in discovered_plugins:
                plugin_name = plugin_manifest['name']
                
                if not plugin_manifest.get('enabled', True):
                    self.logger.info(f"Skipping disabled plugin: {plugin_name}")
                    continue
                
                plugin = await self.loader.load_plugin(
                    plugin_manifest['path'], 
                    plugin_manifest
                )
                
                if plugin:
                    # Initialize plugin
                    if await plugin.initialize(self.api):
                        self.plugins[plugin_name] = plugin
                        self.enabled_plugins[plugin_name] = True
                        
                        # Categorize plugin
                        category = plugin.metadata.category
                        if category not in self.plugin_categories:
                            self.plugin_categories[category] = []
                        self.plugin_categories[category].append(plugin_name)
                        
                        loaded_count += 1
                        self.logger.info(f"✅ Plugin initialized: {plugin_name}")
                    else:
                        self.logger.error(f"Plugin initialization failed: {plugin_name}")
            
            # Start event processor
            self._event_processor_task = asyncio.create_task(self._process_events())
            
            self.logger.info(f"✅ Plugin Manager initialized with {loaded_count} plugins")
            
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, True, {
                    "plugins_discovered": len(discovered_plugins),
                    "plugins_loaded": loaded_count,
                    "categories": len(self.plugin_categories)
                })
                
        except Exception as e:
            self.logger.error(f"Plugin manager initialization failed: {e}", exception=e)
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, False)
    
    async def _process_events(self):
        """Process plugin events"""
        while True:
            try:
                event = await self.event_queue.get()
                await self.handle_event(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event processing failed: {e}")
    
    async def handle_event(self, event: PluginEvent):
        """Handle an event across all plugins"""
        handlers_called = 0
        
        for plugin_name, plugin in self.plugins.items():
            if not self.enabled_plugins.get(plugin_name, False):
                continue
            
            try:
                if await plugin.handle_event(event):
                    handlers_called += 1
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} event handling failed: {e}")
        
        self.logger.debug(f"Event '{event.event_type}' handled by {handlers_called} plugins")
    
    def emit_event(self, event_type: str, data: Dict[str, Any], source: str = "system"):
        """Emit an event to all plugins"""
        event = PluginEvent(
            event_type=event_type,
            data=data,
            source=source,
            timestamp=time.time()
        )
        
        asyncio.create_task(self.event_queue.put(event))
    
    async def handle_voice_command(self, command: str, context: Dict[str, Any]) -> Optional[str]:
        """Handle voice command through plugins"""
        for plugin_name, plugin in self.plugins.items():
            if not self.enabled_plugins.get(plugin_name, False):
                continue
            
            if isinstance(plugin, VoicePluginInterface):
                try:
                    result = await plugin.handle_voice_command(command, context)
                    if result:
                        self.logger.info(f"Voice command handled by plugin: {plugin_name}")
                        return result
                except Exception as e:
                    self.logger.error(f"Plugin {plugin_name} voice command failed: {e}")
        
        return None
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        stats = {
            "total_plugins": len(self.plugins),
            "enabled_plugins": sum(self.enabled_plugins.values()),
            "categories": dict(self.plugin_categories),
            "plugins": {}
        }
        
        for plugin_name, plugin in self.plugins.items():
            stats["plugins"][plugin_name] = {
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "category": plugin.metadata.category,
                "enabled": self.enabled_plugins.get(plugin_name, False),
                "active": plugin.is_active
            }
        
        if ENHANCED_LOGGING:
            component_stats = self.logger.get_component_stats()
            stats.update({"performance": component_stats})
        
        return stats
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        if plugin_name in self.plugins:
            self.enabled_plugins[plugin_name] = True
            self.logger.info(f"Plugin enabled: {plugin_name}")
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name in self.plugins:
            self.enabled_plugins[plugin_name] = False
            self.logger.info(f"Plugin disabled: {plugin_name}")
            return True
        return False
    
    async def cleanup(self):
        """Cleanup plugin manager"""
        try:
            self.logger.info("🧹 Cleaning up Plugin Manager...")
            
            # Cancel event processor
            if self._event_processor_task:
                self._event_processor_task.cancel()
                try:
                    await self._event_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup all plugins
            for plugin_name, plugin in self.plugins.items():
                try:
                    await plugin.cleanup()
                    self.logger.info(f"Plugin cleaned up: {plugin_name}")
                except Exception as e:
                    self.logger.error(f"Plugin cleanup failed {plugin_name}: {e}")
            
            self.plugins.clear()
            self.enabled_plugins.clear()
            
            self.logger.info("✅ Plugin Manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Plugin manager cleanup failed: {e}")


# Sample plugin example
class WeatherPlugin(VoicePluginInterface):
    """Example weather plugin"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.api_key = None
    
    async def initialize(self, jarvis_api: JarvisAPI) -> bool:
        self.logger.info("Initializing Weather Plugin")
        
        # Get API key from config
        self.api_key = jarvis_api.get_config("weather_api_key")
        if not self.api_key:
            self.logger.warning("Weather API key not configured")
        
        # Register event handlers
        self.register_event_handler("weather_request", self._handle_weather_request)
        
        return True
    
    async def cleanup(self) -> bool:
        self.logger.info("Cleaning up Weather Plugin")
        return True
    
    async def handle_voice_command(self, command: str, context: Dict[str, Any]) -> Optional[str]:
        command_lower = command.lower()
        
        if "weather" in command_lower:
            location = self._extract_location(command)
            return await self._get_weather(location)
        
        return None
    
    def _extract_location(self, command: str) -> str:
        # Simple location extraction
        if "in" in command:
            parts = command.split("in")
            if len(parts) > 1:
                return parts[-1].strip()
        return "current location"
    
    async def _get_weather(self, location: str) -> str:
        # Mock weather response
        return f"The weather in {location} is sunny with a temperature of 25°C."
    
    async def _handle_weather_request(self, event: PluginEvent):
        location = event.data.get("location", "current location")
        weather_info = await self._get_weather(location)
        self.logger.info(f"Weather request handled for {location}")
    
    def get_capabilities(self) -> List[str]:
        return ["weather_query", "location_weather"]
    
    def get_commands(self) -> Dict[str, Callable]:
        return {
            "weather": self.handle_voice_command,
            "forecast": self.handle_voice_command
        }


# Global plugin manager
_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager(core_system=None, config: Optional[Dict[str, Any]] = None) -> PluginManager:
    """Get global plugin manager instance"""
    global _plugin_manager
    
    if _plugin_manager is None and core_system:
        _plugin_manager = PluginManager(core_system, config)
    
    return _plugin_manager