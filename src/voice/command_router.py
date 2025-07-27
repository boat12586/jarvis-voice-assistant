"""
Voice Command Router for JARVIS Voice Assistant
Routes parsed commands to appropriate handlers and manages command execution
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from dataclasses import dataclass
from enum import Enum
import time
import json

from .command_parser import ParsedCommand, CommandType, CommandPriority


class CommandStatus(Enum):
    """Command execution status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CommandExecution:
    """Command execution tracking"""
    command: ParsedCommand
    handler_name: str
    start_time: float
    end_time: Optional[float] = None
    status: CommandStatus = CommandStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_id: str = ""


class CommandHandler:
    """Base class for command handlers"""
    
    def __init__(self, name: str, supported_intents: List[str]):
        self.name = name
        self.supported_intents = supported_intents
        self.logger = logging.getLogger(f"handler.{name}")
    
    async def handle(self, command: ParsedCommand) -> Dict[str, Any]:
        """Handle command execution - override in subclasses"""
        raise NotImplementedError("Handlers must implement handle method")
    
    def can_handle(self, command: ParsedCommand) -> bool:
        """Check if this handler can process the command"""
        return command.intent in self.supported_intents
    
    def get_priority(self, command: ParsedCommand) -> int:
        """Get handler priority for this command (higher = more preferred)"""
        return 50  # Default priority


class VoiceCommandRouter(QObject):
    """Routes and executes voice commands"""
    
    # Signals
    command_received = pyqtSignal(dict)
    command_started = pyqtSignal(str, str)  # execution_id, handler_name
    command_completed = pyqtSignal(str, dict)  # execution_id, result
    command_failed = pyqtSignal(str, str)  # execution_id, error
    handler_registered = pyqtSignal(str)  # handler_name
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("command_router", {})
        
        # Command handlers
        self.handlers: Dict[str, CommandHandler] = {}
        self.intent_handlers: Dict[str, List[CommandHandler]] = {}
        
        # Execution tracking
        self.active_executions: Dict[str, CommandExecution] = {}
        self.execution_history: List[CommandExecution] = []
        self.max_history = self.config.get("max_history", 100)
        
        # Queue management
        self.command_queue: List[CommandExecution] = []
        self.max_queue_size = self.config.get("max_queue_size", 10)
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self._process_queue)
        self.processing_timer.start(100)  # Process queue every 100ms
        
        # Statistics
        self.stats = {
            "commands_processed": 0,
            "commands_successful": 0,
            "commands_failed": 0,
            "average_execution_time": 0.0,
            "handler_usage": {}
        }
        
        # Initialize built-in handlers
        self._initialize_builtin_handlers()
    
    def _initialize_builtin_handlers(self):
        """Initialize built-in command handlers"""
        try:
            # Information handler
            self.register_handler(InformationHandler())
            
            # Action handler
            self.register_handler(ActionHandler())
            
            # System handler
            self.register_handler(SystemHandler())
            
            # Conversation handler
            self.register_handler(ConversationHandler())
            
            self.logger.info("Built-in command handlers initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize built-in handlers: {e}")
    
    def register_handler(self, handler: CommandHandler) -> bool:
        """Register a command handler"""
        try:
            if handler.name in self.handlers:
                self.logger.warning(f"Handler {handler.name} already registered, replacing")
            
            self.handlers[handler.name] = handler
            
            # Update intent mapping
            for intent in handler.supported_intents:
                if intent not in self.intent_handlers:
                    self.intent_handlers[intent] = []
                
                self.intent_handlers[intent].append(handler)
                
                # Sort by priority (highest first)
                self.intent_handlers[intent].sort(
                    key=lambda h: h.get_priority(None), reverse=True
                )
            
            self.handler_registered.emit(handler.name)
            self.logger.info(f"Registered command handler: {handler.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register handler {handler.name}: {e}")
            self.error_occurred.emit(f"Handler registration failed: {e}")
            return False
    
    def unregister_handler(self, handler_name: str) -> bool:
        """Unregister a command handler"""
        try:
            if handler_name not in self.handlers:
                self.logger.warning(f"Handler {handler_name} not found")
                return False
            
            handler = self.handlers[handler_name]
            
            # Remove from intent mapping
            for intent in handler.supported_intents:
                if intent in self.intent_handlers:
                    self.intent_handlers[intent] = [
                        h for h in self.intent_handlers[intent] 
                        if h.name != handler_name
                    ]
                    
                    if not self.intent_handlers[intent]:
                        del self.intent_handlers[intent]
            
            # Remove from handlers
            del self.handlers[handler_name]
            
            self.logger.info(f"Unregistered command handler: {handler_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister handler {handler_name}: {e}")
            return False
    
    def route_command(self, command: ParsedCommand) -> str:
        """Route command for execution"""
        try:
            self.logger.info(f"Routing command: {command.intent} - '{command.cleaned_text}'")
            
            # Find appropriate handler
            handler = self._find_best_handler(command)
            
            if not handler:
                self.logger.warning(f"No handler found for intent: {command.intent}")
                self.error_occurred.emit(f"No handler available for: {command.intent}")
                return ""
            
            # Create execution
            execution = CommandExecution(
                command=command,
                handler_name=handler.name,
                start_time=time.time(),
                execution_id=self._generate_execution_id()
            )
            
            # Add to queue
            if len(self.command_queue) >= self.max_queue_size:
                self.logger.warning("Command queue full, dropping oldest command")
                self.command_queue.pop(0)
            
            self.command_queue.append(execution)
            self.command_received.emit(command.__dict__)
            
            self.logger.info(f"Command queued with execution ID: {execution.execution_id}")
            return execution.execution_id
            
        except Exception as e:
            self.logger.error(f"Command routing failed: {e}")
            self.error_occurred.emit(f"Command routing failed: {e}")
            return ""
    
    def _find_best_handler(self, command: ParsedCommand) -> Optional[CommandHandler]:
        """Find the best handler for a command"""
        try:
            # Get handlers for this intent
            if command.intent not in self.intent_handlers:
                return None
            
            handlers = self.intent_handlers[command.intent]
            
            # Find the best handler
            best_handler = None
            best_priority = -1
            
            for handler in handlers:
                if handler.can_handle(command):
                    priority = handler.get_priority(command)
                    if priority > best_priority:
                        best_priority = priority
                        best_handler = handler
            
            return best_handler
            
        except Exception as e:
            self.logger.error(f"Handler selection failed: {e}")
            return None
    
    def _process_queue(self):
        """Process command queue"""
        try:
            if not self.command_queue:
                return
            
            # Get next command (respect priority)
            execution = self._get_next_execution()
            
            if not execution:
                return
            
            # Start execution
            self._execute_command(execution)
            
        except Exception as e:
            self.logger.error(f"Queue processing failed: {e}")
    
    def _get_next_execution(self) -> Optional[CommandExecution]:
        """Get next command to execute (priority-based)"""
        try:
            if not self.command_queue:
                return None
            
            # Sort by command priority
            self.command_queue.sort(
                key=lambda e: e.command.priority.value, reverse=True
            )
            
            return self.command_queue.pop(0)
            
        except Exception as e:
            self.logger.error(f"Execution selection failed: {e}")
            return None
    
    def _execute_command(self, execution: CommandExecution):
        """Execute a command"""
        try:
            # Get handler
            handler = self.handlers.get(execution.handler_name)
            if not handler:
                execution.status = CommandStatus.FAILED
                execution.error = f"Handler {execution.handler_name} not found"
                self._complete_execution(execution)
                return
            
            # Start execution
            execution.status = CommandStatus.PROCESSING
            self.active_executions[execution.execution_id] = execution
            
            self.command_started.emit(execution.execution_id, execution.handler_name)
            self.logger.info(f"Starting execution {execution.execution_id} with {execution.handler_name}")
            
            # Execute asynchronously
            QTimer.singleShot(0, lambda: self._execute_async(execution, handler))
            
        except Exception as e:
            execution.status = CommandStatus.FAILED
            execution.error = str(e)
            self._complete_execution(execution)
    
    def _execute_async(self, execution: CommandExecution, handler: CommandHandler):
        """Execute command asynchronously"""
        try:
            # Use QTimer for simple async execution without asyncio complexity
            def run_handler():
                try:
                    # Run handler synchronously for now (can be improved later)
                    import asyncio
                    result = asyncio.run(handler.handle(execution.command))
                    execution.result = result
                    execution.status = CommandStatus.COMPLETED
                except Exception as e:
                    execution.error = str(e)
                    execution.status = CommandStatus.FAILED
                    self.logger.error(f"Handler execution failed: {e}")
                
                # Complete execution
                self._complete_execution(execution)
            
            # Execute on the next event loop iteration
            QTimer.singleShot(10, run_handler)
            
        except Exception as e:
            execution.status = CommandStatus.FAILED
            execution.error = str(e)
            self._complete_execution(execution)
    
    def _complete_execution(self, execution: CommandExecution):
        """Complete command execution"""
        try:
            execution.end_time = time.time()
            
            # Remove from active executions
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
            
            # Add to history
            self.execution_history.append(execution)
            if len(self.execution_history) > self.max_history:
                self.execution_history.pop(0)
            
            # Update statistics
            self._update_stats(execution)
            
            # Emit signals
            if execution.status == CommandStatus.COMPLETED:
                self.command_completed.emit(execution.execution_id, execution.result or {})
                self.logger.info(f"Command {execution.execution_id} completed successfully")
            else:
                self.command_failed.emit(execution.execution_id, execution.error or "Unknown error")
                self.logger.error(f"Command {execution.execution_id} failed: {execution.error}")
            
        except Exception as e:
            self.logger.error(f"Execution completion failed: {e}")
    
    def _update_stats(self, execution: CommandExecution):
        """Update router statistics"""
        try:
            self.stats["commands_processed"] += 1
            
            if execution.status == CommandStatus.COMPLETED:
                self.stats["commands_successful"] += 1
            else:
                self.stats["commands_failed"] += 1
            
            # Update execution time
            if execution.end_time:
                exec_time = execution.end_time - execution.start_time
                total_time = self.stats["average_execution_time"] * (self.stats["commands_processed"] - 1)
                self.stats["average_execution_time"] = (total_time + exec_time) / self.stats["commands_processed"]
            
            # Update handler usage
            handler_name = execution.handler_name
            if handler_name not in self.stats["handler_usage"]:
                self.stats["handler_usage"][handler_name] = 0
            self.stats["handler_usage"][handler_name] += 1
            
        except Exception as e:
            self.logger.error(f"Stats update failed: {e}")
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        import uuid
        return f"exec_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        try:
            # Check active executions
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                return {
                    "execution_id": execution_id,
                    "status": execution.status.value,
                    "handler": execution.handler_name,
                    "start_time": execution.start_time,
                    "elapsed_time": time.time() - execution.start_time
                }
            
            # Check history
            for execution in self.execution_history:
                if execution.execution_id == execution_id:
                    return {
                        "execution_id": execution_id,
                        "status": execution.status.value,
                        "handler": execution.handler_name,
                        "start_time": execution.start_time,
                        "end_time": execution.end_time,
                        "execution_time": (execution.end_time or time.time()) - execution.start_time,
                        "result": execution.result,
                        "error": execution.error
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return None
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel command execution"""
        try:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = CommandStatus.CANCELLED
                execution.error = "Cancelled by user"
                self._complete_execution(execution)
                return True
            
            # Remove from queue if pending
            self.command_queue = [
                e for e in self.command_queue 
                if e.execution_id != execution_id
            ]
            
            return False
            
        except Exception as e:
            self.logger.error(f"Execution cancellation failed: {e}")
            return False
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "registered_handlers": len(self.handlers),
            "active_executions": len(self.active_executions),
            "queue_size": len(self.command_queue),
            "supported_intents": list(self.intent_handlers.keys()),
            "statistics": self.stats.copy()
        }
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")


# Built-in Command Handlers

class InformationHandler(CommandHandler):
    """Handles information requests"""
    
    def __init__(self):
        super().__init__(
            name="information",
            supported_intents=["information_request", "how_to_request", "explanation_request"]
        )
    
    async def handle(self, command: ParsedCommand) -> Dict[str, Any]:
        """Handle information requests"""
        try:
            topic = command.parameters.get("topic", command.cleaned_text)
            
            # Simulate information lookup
            await asyncio.sleep(0.1)
            
            return {
                "type": "information_response",
                "topic": topic,
                "message": f"Here's information about: {topic}",
                "requires_ai_response": True,
                "context_needed": True
            }
            
        except Exception as e:
            raise Exception(f"Information lookup failed: {e}")


class ActionHandler(CommandHandler):
    """Handles action requests"""
    
    def __init__(self):
        super().__init__(
            name="action",
            supported_intents=["action_request"]
        )
    
    async def handle(self, command: ParsedCommand) -> Dict[str, Any]:
        """Handle action requests"""
        try:
            action = command.parameters.get("action", "unknown")
            target = command.parameters.get("target", "")
            
            # Simulate action execution
            await asyncio.sleep(0.05)
            
            return {
                "type": "action_response",
                "action": action,
                "target": target,
                "message": f"Executed {action} on {target}",
                "success": True
            }
            
        except Exception as e:
            raise Exception(f"Action execution failed: {e}")


class SystemHandler(CommandHandler):
    """Handles system control commands"""
    
    def __init__(self):
        super().__init__(
            name="system",
            supported_intents=["system_control"]
        )
    
    def get_priority(self, command: ParsedCommand) -> int:
        return 90  # High priority for system commands
    
    async def handle(self, command: ParsedCommand) -> Dict[str, Any]:
        """Handle system commands"""
        try:
            system_type = command.parameters.get("system_type", "general")
            
            # Handle different system commands
            if system_type == "audio":
                direction = command.parameters.get("direction", "none")
                return {
                    "type": "system_response",
                    "system_type": "audio",
                    "action": f"volume_{direction}",
                    "message": f"Audio volume {direction}d",
                    "success": True
                }
            
            elif system_type == "settings":
                return {
                    "type": "system_response",
                    "system_type": "settings",
                    "action": "open_settings",
                    "message": "Opening system settings",
                    "success": True
                }
            
            else:
                return {
                    "type": "system_response",
                    "message": "System command executed",
                    "success": True
                }
            
        except Exception as e:
            raise Exception(f"System command failed: {e}")


class ConversationHandler(CommandHandler):
    """Handles conversational commands"""
    
    def __init__(self):
        super().__init__(
            name="conversation",
            supported_intents=["greeting"]
        )
    
    async def handle(self, command: ParsedCommand) -> Dict[str, Any]:
        """Handle conversational commands"""
        try:
            # Determine response based on language
            if command.language == "th":
                message = "สวัสดีครับ! ผมคือ JARVIS มีอะไรให้ช่วยไหมครับ?"
            else:
                message = "Hello! I'm JARVIS, your AI assistant. How can I help you today?"
            
            return {
                "type": "conversation_response",
                "message": message,
                "language": command.language,
                "friendly": True,
                "requires_response": True
            }
            
        except Exception as e:
            raise Exception(f"Conversation handling failed: {e}")