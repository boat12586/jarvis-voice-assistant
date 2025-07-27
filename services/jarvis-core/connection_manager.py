"""
Enhanced Connection Manager for Jarvis v2.0
Supports multi-user sessions with isolation and real-time communication
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from models import WebSocketMessage, WebSocketResponse, SessionStatus
from user_manager import UserManager

logger = logging.getLogger(__name__)

class UserConnection:
    """Represents a user's WebSocket connection with session context"""
    
    def __init__(self, websocket: WebSocket, user_id: str, session_id: str):
        self.websocket = websocket
        self.user_id = user_id
        self.session_id = session_id
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        self.connection_id = f"{user_id}_{session_id}_{int(self.connected_at.timestamp())}"
        
    async def send_message(self, message: WebSocketResponse):
        """Send message to this connection"""
        try:
            await self.websocket.send_text(message.json())
            self.last_activity = datetime.now()
        except Exception as e:
            logger.error(f"Failed to send message to {self.connection_id}: {e}")
            raise
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

class ConnectionManager:
    """Enhanced connection manager with multi-user support and session isolation"""
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
        
        # Connection mappings
        self.connections: Dict[str, UserConnection] = {}  # connection_id -> UserConnection
        self.user_connections: Dict[str, Set[str]] = {}   # user_id -> set of connection_ids
        self.session_connections: Dict[str, Set[str]] = {} # session_id -> set of connection_ids
        
        # Message queues for offline users
        self.pending_messages: Dict[str, List[WebSocketResponse]] = {}  # user_id -> messages
        
        # Rate limiting
        self.message_counts: Dict[str, Dict[str, int]] = {}  # user_id -> {minute: count}
        self.max_messages_per_minute = 60
        
    async def connect(self, websocket: WebSocket, user_id: str, session_id: Optional[str] = None) -> bool:
        """Connect a user with session isolation"""
        try:
            # Validate user
            user = self.user_manager.get_user(user_id)
            if not user:
                logger.warning(f"Connection attempt by non-existent user: {user_id}")
                await websocket.close(code=4004, reason="User not found")
                return False
            
            if user.status.value != "active":
                logger.warning(f"Connection attempt by inactive user: {user_id}")
                await websocket.close(code=4003, reason="User account inactive")
                return False
            
            # Create or validate session
            if session_id:
                session = self.user_manager.get_session(session_id)
                if not session or session.user_id != user_id:
                    logger.warning(f"Invalid session {session_id} for user {user_id}")
                    await websocket.close(code=4005, reason="Invalid session")
                    return False
            else:
                # Create new session
                from models import SessionCreate, SessionType
                session_create = SessionCreate(
                    user_id=user_id,
                    session_type=SessionType.WEB,
                    expires_in_hours=24
                )
                session = self.user_manager.create_session(session_create)
                if not session:
                    logger.error(f"Failed to create session for user {user_id}")
                    await websocket.close(code=4006, reason="Session creation failed")
                    return False
                session_id = session.session_id
            
            # Accept WebSocket connection
            await websocket.accept()
            
            # Create connection object
            connection = UserConnection(websocket, user_id, session_id)
            
            # Store connection mappings
            self.connections[connection.connection_id] = connection
            
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection.connection_id)
            
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(connection.connection_id)
            
            # Update session activity
            self.user_manager.update_session_activity(session_id)
            
            # Send connection confirmation
            welcome_message = WebSocketResponse(
                type="connection_established",
                data={
                    "user_id": user_id,
                    "session_id": session_id,
                    "username": user.username,
                    "connection_id": connection.connection_id,
                    "server_time": datetime.now().isoformat()
                },
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.now(),
                message_id=f"welcome_{connection.connection_id}"
            )
            await connection.send_message(welcome_message)
            
            # Send pending messages
            await self._send_pending_messages(user_id, connection)
            
            logger.info(f"WebSocket connected: {connection.connection_id} for user {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect user {user_id}: {e}")
            try:
                await websocket.close(code=4001, reason="Connection failed")
            except:
                pass
            return False
    
    async def disconnect(self, connection_id: str):
        """Disconnect a specific connection"""
        try:
            if connection_id not in self.connections:
                return
            
            connection = self.connections[connection_id]
            user_id = connection.user_id
            session_id = connection.session_id
            
            # Remove from mappings
            del self.connections[connection_id]
            
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(connection_id)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
                    # Update session status to idle if no connections
                    session = self.user_manager.get_session(session_id)
                    if session:
                        session.status = SessionStatus.IDLE
            
            logger.info(f"WebSocket disconnected: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error during disconnect for {connection_id}: {e}")
    
    async def disconnect_user(self, user_id: str):
        """Disconnect all connections for a user"""
        try:
            if user_id not in self.user_connections:
                return
            
            connection_ids = list(self.user_connections[user_id])
            for connection_id in connection_ids:
                if connection_id in self.connections:
                    connection = self.connections[connection_id]
                    try:
                        await connection.websocket.close(code=4002, reason="User disconnected")
                    except:
                        pass
                    await self.disconnect(connection_id)
            
            logger.info(f"All connections disconnected for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting user {user_id}: {e}")
    
    async def send_to_user(self, user_id: str, message: WebSocketResponse, 
                          exclude_session: Optional[str] = None):
        """Send message to all connections of a user (optionally excluding a session)"""
        try:
            if user_id not in self.user_connections:
                # Store message for when user comes online
                self._queue_pending_message(user_id, message)
                return False
            
            sent_count = 0
            failed_connections = []
            
            for connection_id in list(self.user_connections[user_id]):
                if connection_id not in self.connections:
                    continue
                
                connection = self.connections[connection_id]
                
                # Skip if excluding this session
                if exclude_session and connection.session_id == exclude_session:
                    continue
                
                try:
                    await connection.send_message(message)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send to connection {connection_id}: {e}")
                    failed_connections.append(connection_id)
            
            # Clean up failed connections
            for connection_id in failed_connections:
                await self.disconnect(connection_id)
            
            return sent_count > 0
            
        except Exception as e:
            logger.error(f"Error sending to user {user_id}: {e}")
            return False
    
    async def send_to_session(self, session_id: str, message: WebSocketResponse):
        """Send message to all connections in a specific session"""
        try:
            if session_id not in self.session_connections:
                return False
            
            sent_count = 0
            failed_connections = []
            
            for connection_id in list(self.session_connections[session_id]):
                if connection_id not in self.connections:
                    continue
                
                connection = self.connections[connection_id]
                
                try:
                    await connection.send_message(message)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send to session connection {connection_id}: {e}")
                    failed_connections.append(connection_id)
            
            # Clean up failed connections
            for connection_id in failed_connections:
                await self.disconnect(connection_id)
            
            return sent_count > 0
            
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")
            return False
    
    async def broadcast_to_all(self, message: WebSocketResponse, 
                              exclude_user: Optional[str] = None):
        """Broadcast message to all connected users"""
        try:
            sent_count = 0
            failed_connections = []
            
            for connection_id, connection in list(self.connections.items()):
                # Skip excluded user
                if exclude_user and connection.user_id == exclude_user:
                    continue
                
                try:
                    await connection.send_message(message)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to broadcast to {connection_id}: {e}")
                    failed_connections.append(connection_id)
            
            # Clean up failed connections
            for connection_id in failed_connections:
                await self.disconnect(connection_id)
            
            logger.info(f"Broadcast sent to {sent_count} connections")
            return sent_count > 0
            
        except Exception as e:
            logger.error(f"Error during broadcast: {e}")
            return False
    
    async def handle_message(self, connection_id: str, raw_message: str) -> bool:
        """Handle incoming WebSocket message with rate limiting"""
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            user_id = connection.user_id
            
            # Rate limiting check
            if not self._check_rate_limit(user_id):
                error_response = WebSocketResponse(
                    type="error",
                    data={"error": "Rate limit exceeded", "retry_after": 60},
                    user_id=user_id,
                    session_id=connection.session_id,
                    timestamp=datetime.now(),
                    message_id=f"rate_limit_{connection_id}"
                )
                await connection.send_message(error_response)
                return False
            
            # Parse message
            try:
                message_data = json.loads(raw_message)
                message = WebSocketMessage(**message_data)
            except Exception as e:
                error_response = WebSocketResponse(
                    type="error",
                    data={"error": "Invalid message format", "detail": str(e)},
                    user_id=user_id,
                    session_id=connection.session_id,
                    timestamp=datetime.now(),
                    message_id=f"parse_error_{connection_id}"
                )
                await connection.send_message(error_response)
                return False
            
            # Update activity
            connection.update_activity()
            self.user_manager.update_session_activity(connection.session_id)
            
            # Handle different message types
            return await self._process_message(connection, message)
            
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            return False
    
    async def _process_message(self, connection: UserConnection, message: WebSocketMessage) -> bool:
        """Process different types of WebSocket messages"""
        try:
            message_type = message.type
            
            if message_type == "ping":
                # Respond with pong
                pong_response = WebSocketResponse(
                    type="pong",
                    data={"timestamp": datetime.now().isoformat()},
                    user_id=connection.user_id,
                    session_id=connection.session_id,
                    timestamp=datetime.now(),
                    message_id=f"pong_{connection.connection_id}"
                )
                await connection.send_message(pong_response)
                return True
            
            elif message_type == "chat":
                # Handle chat message (will be processed by main app)
                return True
            
            elif message_type == "voice":
                # Handle voice message (will be processed by main app)
                return True
            
            elif message_type == "status":
                # Send status information
                status_response = WebSocketResponse(
                    type="status",
                    data={
                        "connection_id": connection.connection_id,
                        "session_id": connection.session_id,
                        "connected_at": connection.connected_at.isoformat(),
                        "last_activity": connection.last_activity.isoformat(),
                        "active_connections": len(self.connections),
                        "user_connections": len(self.user_connections.get(connection.user_id, set()))
                    },
                    user_id=connection.user_id,
                    session_id=connection.session_id,
                    timestamp=datetime.now(),
                    message_id=f"status_{connection.connection_id}"
                )
                await connection.send_message(status_response)
                return True
            
            else:
                # Unknown message type
                error_response = WebSocketResponse(
                    type="error",
                    data={"error": f"Unknown message type: {message_type}"},
                    user_id=connection.user_id,
                    session_id=connection.session_id,
                    timestamp=datetime.now(),
                    message_id=f"unknown_type_{connection.connection_id}"
                )
                await connection.send_message(error_response)
                return False
                
        except Exception as e:
            logger.error(f"Error processing message type {message.type}: {e}")
            return False
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        try:
            current_minute = int(datetime.now().timestamp() // 60)
            
            if user_id not in self.message_counts:
                self.message_counts[user_id] = {}
            
            # Clean old entries
            user_counts = self.message_counts[user_id]
            old_minutes = [minute for minute in user_counts if minute < current_minute - 5]
            for minute in old_minutes:
                del user_counts[minute]
            
            # Check current minute count
            current_count = user_counts.get(current_minute, 0)
            if current_count >= self.max_messages_per_minute:
                return False
            
            # Increment count
            user_counts[current_minute] = current_count + 1
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit for {user_id}: {e}")
            return True  # Allow on error
    
    def _queue_pending_message(self, user_id: str, message: WebSocketResponse):
        """Queue message for offline user"""
        try:
            if user_id not in self.pending_messages:
                self.pending_messages[user_id] = []
            
            # Limit pending messages per user
            if len(self.pending_messages[user_id]) >= 100:
                self.pending_messages[user_id].pop(0)  # Remove oldest
            
            self.pending_messages[user_id].append(message)
            
        except Exception as e:
            logger.error(f"Error queuing message for {user_id}: {e}")
    
    async def _send_pending_messages(self, user_id: str, connection: UserConnection):
        """Send queued messages to newly connected user"""
        try:
            if user_id not in self.pending_messages:
                return
            
            messages = self.pending_messages[user_id]
            sent_count = 0
            
            for message in messages:
                try:
                    await connection.send_message(message)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send pending message: {e}")
                    break
            
            # Clear sent messages
            del self.pending_messages[user_id]
            
            if sent_count > 0:
                logger.info(f"Sent {sent_count} pending messages to {user_id}")
                
        except Exception as e:
            logger.error(f"Error sending pending messages to {user_id}: {e}")
    
    # Information methods
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a connection"""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        return {
            "connection_id": connection_id,
            "user_id": connection.user_id,
            "session_id": connection.session_id,
            "connected_at": connection.connected_at.isoformat(),
            "last_activity": connection.last_activity.isoformat()
        }
    
    def get_user_connection_info(self, user_id: str) -> Dict[str, Any]:
        """Get connection information for a user"""
        if user_id not in self.user_connections:
            return {
                "user_id": user_id,
                "connection_count": 0,
                "connections": [],
                "is_online": False
            }
        
        connections = []
        for connection_id in self.user_connections[user_id]:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                connections.append({
                    "connection_id": connection_id,
                    "session_id": connection.session_id,
                    "connected_at": connection.connected_at.isoformat(),
                    "last_activity": connection.last_activity.isoformat()
                })
        
        return {
            "user_id": user_id,
            "connection_count": len(connections),
            "connections": connections,
            "is_online": len(connections) > 0
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide connection statistics"""
        return {
            "total_connections": len(self.connections),
            "unique_users": len(self.user_connections),
            "active_sessions": len(self.session_connections),
            "pending_message_queues": len(self.pending_messages),
            "total_pending_messages": sum(len(msgs) for msgs in self.pending_messages.values())
        }
    
    async def cleanup_inactive_connections(self, inactive_minutes: int = 30):
        """Clean up connections that have been inactive"""
        try:
            current_time = datetime.now()
            inactive_connections = []
            
            for connection_id, connection in self.connections.items():
                inactive_duration = current_time - connection.last_activity
                if inactive_duration.total_seconds() > (inactive_minutes * 60):
                    inactive_connections.append(connection_id)
            
            for connection_id in inactive_connections:
                connection = self.connections[connection_id]
                try:
                    await connection.websocket.close(code=4008, reason="Inactive connection")
                except:
                    pass
                await self.disconnect(connection_id)
            
            if inactive_connections:
                logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")
            
            return len(inactive_connections)
            
        except Exception as e:
            logger.error(f"Error during connection cleanup: {e}")
            return 0