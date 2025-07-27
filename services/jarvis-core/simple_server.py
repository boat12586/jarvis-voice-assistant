#!/usr/bin/env python3
"""
Simple FastAPI server for JARVIS Web Interface
Provides basic API endpoints without complex dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="JARVIS Voice Assistant - Simple API",
    description="Simple API server for JARVIS web interface",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models
class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    language: str = "en"

class ChatResponse(BaseModel):
    response: str
    confidence: float = 0.95
    session_id: str
    timestamp: datetime
    
class HealthStatus(BaseModel):
    status: str = "healthy"
    services: Dict[str, str] = {
        "ai_engine": "healthy",
        "voice_engine": "healthy",
        "memory_system": "healthy"
    }
    timestamp: datetime
    
class SystemStats(BaseModel):
    active_connections: int = 1
    total_messages: int = 0
    uptime_seconds: int = 0
    memory_usage: Dict[str, float] = {"used": 45.2, "total": 100.0}
    cpu_usage: float = 12.5

# In-memory storage for demo
sessions = {}
messages_count = 0
start_time = datetime.now()

# API Endpoints
@app.get("/api/v2/health", response_model=HealthStatus)
async def get_health():
    """Get system health status"""
    return HealthStatus(
        status="healthy",
        services={
            "ai_engine": "healthy",
            "voice_engine": "healthy", 
            "memory_system": "healthy"
        },
        timestamp=datetime.now()
    )

@app.get("/api/v2/admin/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    uptime = (datetime.now() - start_time).total_seconds()
    return SystemStats(
        active_connections=1,
        total_messages=messages_count,
        uptime_seconds=int(uptime),
        memory_usage={"used": 45.2, "total": 100.0},
        cpu_usage=12.5
    )

@app.post("/api/v2/chat", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest):
    """Send a chat message to JARVIS"""
    global messages_count
    messages_count += 1
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Simple AI responses for demo
    responses = {
        "hello": "Hello! I'm JARVIS, your voice assistant. How can I help you today?",
        "hi": "Hi there! I'm ready to assist you.",
        "how are you": "I'm functioning perfectly, thank you for asking!",
        "test": "All systems are operational. Voice recognition and AI processing are working correctly.",
        "status": "All JARVIS systems are online and functioning optimally.",
        "voice": "Voice processing systems are active and ready for input.",
        "help": "I can help you with various tasks including voice commands, information lookup, and system control."
    }
    
    # Simple keyword matching for demo
    message_lower = request.message.lower()
    response_text = "I understand your message. All JARVIS systems are operational and ready to assist you."
    
    for keyword, response in responses.items():
        if keyword in message_lower:
            response_text = response
            break
    
    # Store session
    if session_id not in sessions:
        sessions[session_id] = {
            "user_id": request.user_id,
            "messages": [],
            "created_at": datetime.now()
        }
    
    sessions[session_id]["messages"].append({
        "user": request.message,
        "assistant": response_text,
        "timestamp": datetime.now()
    })
    
    return ChatResponse(
        response=response_text,
        confidence=0.95,
        session_id=session_id,
        timestamp=datetime.now()
    )

@app.post("/api/v2/voice")
async def process_voice():
    """Process voice input (placeholder)"""
    return {
        "message": "Voice processing is ready but requires audio input",
        "status": "ready",
        "session_id": str(uuid.uuid4())
    }

@app.get("/api/v2/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id in sessions:
        return sessions[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "JARVIS Voice Assistant API",
        "version": "2.0.0",
        "status": "online",
        "timestamp": datetime.now(),
        "endpoints": {
            "health": "/api/v2/health",
            "chat": "/api/v2/chat", 
            "voice": "/api/v2/voice",
            "stats": "/api/v2/admin/stats"
        }
    }

if __name__ == "__main__":
    logger.info("Starting JARVIS Simple API Server...")
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )