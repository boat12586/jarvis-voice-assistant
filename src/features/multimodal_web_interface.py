"""
Multimodal Web Interface for JARVIS Voice Assistant
Web integration for image/video upload and multimodal AI processing
"""

import logging
import os
import tempfile
import uuid
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import asyncio
import base64
from io import BytesIO

from flask import Flask, request, jsonify, render_template_string, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

# Import multimodal components
from ..ai.multimodal_fusion_system import (
    MultimodalFusionSystem, 
    ModalityInput, 
    ModalityType,
    MultimodalFusionStrategy
)


class MultimodalWebInterface:
    """Web interface for multimodal AI interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.get('secret_key', 'multimodal_jarvis_2024')
        self.app.config['MAX_CONTENT_LENGTH'] = config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        
        # SocketIO for real-time communication
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # File handling
        self.upload_folder = Path(config.get('upload_folder', 'uploads'))
        self.upload_folder.mkdir(exist_ok=True)
        
        self.allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.allowed_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        self.allowed_extensions = self.allowed_image_extensions | self.allowed_video_extensions
        
        # Multimodal AI system
        self.fusion_system = None
        
        # Active sessions
        self.active_sessions = {}
        
        # Initialize routes
        self._setup_routes()
        self._setup_socketio_events()
        
        # Initialize multimodal system
        self._initialize_multimodal_system()
    
    def _initialize_multimodal_system(self):
        """Initialize multimodal AI system"""
        try:
            multimodal_config = self.config.get('multimodal', {})
            self.fusion_system = MultimodalFusionSystem(multimodal_config)
            self.logger.info("Multimodal fusion system initialized for web interface")
        except Exception as e:
            self.logger.error(f"Failed to initialize multimodal system: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(self._get_main_template())
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_file():
            return self._handle_file_upload()
        
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze_multimodal():
            return self._handle_multimodal_analysis()
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            return self._handle_chat()
        
        @self.app.route('/api/status', methods=['GET'])
        def status():
            return self._get_system_status()
        
        @self.app.route('/api/history/<session_id>', methods=['GET'])
        def get_history(session_id):
            return self._get_session_history(session_id)
        
        @self.app.route('/uploads/<filename>')
        def uploaded_file(filename):
            return send_file(self.upload_folder / filename)
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            session_id = str(uuid.uuid4())
            self.active_sessions[session_id] = {
                'created_at': datetime.now(),
                'files': [],
                'conversation': []
            }
            emit('session_created', {'session_id': session_id})
            self.logger.info(f"New session created: {session_id}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            # Clean up session if needed
            pass
        
        @self.socketio.on('analyze_image')
        def handle_analyze_image(data):
            self._socketio_analyze_image(data)
        
        @self.socketio.on('analyze_video')
        def handle_analyze_video(data):
            self._socketio_analyze_video(data)
        
        @self.socketio.on('multimodal_query')
        def handle_multimodal_query(data):
            self._socketio_multimodal_query(data)
    
    def _handle_file_upload(self):
        """Handle file upload"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            # Generate secure filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{file_id}_{filename}"
            
            # Save file
            filepath = self.upload_folder / filename
            file.save(filepath)
            
            # Get file info
            file_info = self._get_file_info(filepath)
            
            # Store in session if provided
            session_id = request.form.get('session_id')
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id]['files'].append({
                    'filename': filename,
                    'original_name': file.filename,
                    'filepath': str(filepath),
                    'file_info': file_info,
                    'uploaded_at': datetime.now().isoformat()
                })
            
            return jsonify({
                'success': True,
                'filename': filename,
                'file_info': file_info,
                'url': f'/uploads/{filename}'
            })
            
        except Exception as e:
            self.logger.error(f"File upload failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_multimodal_analysis(self):
        """Handle multimodal analysis request"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Prepare modality inputs
            modality_inputs = []
            
            # Text input
            if 'text' in data:
                text_input = ModalityInput(
                    modality=ModalityType.TEXT,
                    data=data['text'],
                    confidence=1.0
                )
                modality_inputs.append(text_input)
            
            # Image/Video input
            if 'filename' in data:
                filepath = self.upload_folder / data['filename']
                if filepath.exists():
                    vision_input = ModalityInput(
                        modality=ModalityType.VISION,
                        data=str(filepath),
                        confidence=0.9
                    )
                    modality_inputs.append(vision_input)
            
            # Voice input (if transcribed)
            if 'voice_transcription' in data:
                voice_input = ModalityInput(
                    modality=ModalityType.VOICE,
                    data=data['voice_transcription'],
                    confidence=data.get('voice_confidence', 0.8)
                )
                modality_inputs.append(voice_input)
            
            # Context input
            session_id = data.get('session_id')
            if session_id and session_id in self.active_sessions:
                context_data = {
                    'conversation_history': self.active_sessions[session_id]['conversation'][-5:],
                    'session_context': {
                        'session_id': session_id,
                        'files_uploaded': len(self.active_sessions[session_id]['files'])
                    }
                }
                context_input = ModalityInput(
                    modality=ModalityType.CONTEXT,
                    data=context_data,
                    confidence=0.6
                )
                modality_inputs.append(context_input)
            
            if not modality_inputs:
                return jsonify({'error': 'No valid inputs provided'}), 400
            
            # Choose fusion strategy
            strategy_name = data.get('fusion_strategy', 'adaptive')
            strategy = MultimodalFusionStrategy(strategy_name)
            
            # Perform fusion
            if not self.fusion_system:
                return jsonify({'error': 'Multimodal system not available'}), 503
            
            fusion_result = self.fusion_system.fuse_multimodal_input(
                modality_inputs,
                fusion_strategy=strategy,
                context=data.get('context')
            )
            
            # Store in session
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id]['conversation'].append({
                    'timestamp': fusion_result.timestamp.isoformat(),
                    'input': data.get('text', 'Multimodal input'),
                    'response': fusion_result.fused_response,
                    'confidence': fusion_result.confidence,
                    'modalities': [m.value for m in fusion_result.modalities_used]
                })
            
            return jsonify({
                'success': True,
                'response': fusion_result.fused_response,
                'confidence': fusion_result.confidence,
                'modalities_used': [m.value for m in fusion_result.modalities_used],
                'fusion_strategy': fusion_result.fusion_strategy,
                'reasoning': fusion_result.reasoning,
                'processing_time': fusion_result.processing_time,
                'detailed_analysis': fusion_result.detailed_analysis
            })
            
        except Exception as e:
            self.logger.error(f"Multimodal analysis failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_chat(self):
        """Handle chat requests"""
        try:
            data = request.get_json()
            message = data.get('message', '')
            session_id = data.get('session_id')
            
            if not message:
                return jsonify({'error': 'No message provided'}), 400
            
            # Simple text-only processing for now
            text_input = ModalityInput(
                modality=ModalityType.TEXT,
                data=message,
                confidence=1.0
            )
            
            if not self.fusion_system:
                return jsonify({'error': 'AI system not available'}), 503
            
            fusion_result = self.fusion_system.fuse_multimodal_input([text_input])
            
            # Store in session
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id]['conversation'].append({
                    'timestamp': datetime.now().isoformat(),
                    'input': message,
                    'response': fusion_result.fused_response,
                    'confidence': fusion_result.confidence,
                    'type': 'text_chat'
                })
            
            return jsonify({
                'success': True,
                'response': fusion_result.fused_response,
                'confidence': fusion_result.confidence
            })
            
        except Exception as e:
            self.logger.error(f"Chat handling failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _get_system_status(self):
        """Get system status"""
        status = {
            'multimodal_system': self.fusion_system is not None,
            'active_sessions': len(self.active_sessions),
            'upload_folder': str(self.upload_folder),
            'allowed_extensions': list(self.allowed_extensions),
            'max_file_size': self.app.config['MAX_CONTENT_LENGTH']
        }
        
        if self.fusion_system:
            status['fusion_stats'] = self.fusion_system.get_fusion_stats()
        
        return jsonify(status)
    
    def _get_session_history(self, session_id: str):
        """Get session conversation history"""
        if session_id not in self.active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = self.active_sessions[session_id]
        return jsonify({
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'files': session['files'],
            'conversation': session['conversation']
        })
    
    def _socketio_analyze_image(self, data):
        """Handle real-time image analysis via SocketIO"""
        try:
            filename = data.get('filename')
            question = data.get('question', '')
            session_id = data.get('session_id')
            
            if not filename:
                emit('analysis_error', {'error': 'No filename provided'})
                return
            
            filepath = self.upload_folder / filename
            if not filepath.exists():
                emit('analysis_error', {'error': 'File not found'})
                return
            
            emit('analysis_started', {'message': 'Starting image analysis...'})
            
            # Prepare inputs
            modality_inputs = [
                ModalityInput(modality=ModalityType.VISION, data=str(filepath), confidence=0.9)
            ]
            
            if question:
                modality_inputs.append(
                    ModalityInput(modality=ModalityType.TEXT, data=question, confidence=1.0)
                )
            
            # Perform analysis
            fusion_result = self.fusion_system.fuse_multimodal_input(modality_inputs)
            
            emit('analysis_complete', {
                'response': fusion_result.fused_response,
                'confidence': fusion_result.confidence,
                'processing_time': fusion_result.processing_time,
                'detailed_analysis': fusion_result.detailed_analysis
            })
            
            # Store in session
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id]['conversation'].append({
                    'timestamp': datetime.now().isoformat(),
                    'input': f"Image analysis: {question}" if question else "Image analysis",
                    'response': fusion_result.fused_response,
                    'confidence': fusion_result.confidence,
                    'type': 'image_analysis',
                    'filename': filename
                })
            
        except Exception as e:
            self.logger.error(f"SocketIO image analysis failed: {e}")
            emit('analysis_error', {'error': str(e)})
    
    def _socketio_analyze_video(self, data):
        """Handle real-time video analysis via SocketIO"""
        try:
            filename = data.get('filename')
            analysis_type = data.get('analysis_type', 'comprehensive')
            session_id = data.get('session_id')
            
            if not filename:
                emit('analysis_error', {'error': 'No filename provided'})
                return
            
            filepath = self.upload_folder / filename
            if not filepath.exists():
                emit('analysis_error', {'error': 'File not found'})
                return
            
            emit('analysis_started', {'message': 'Starting video analysis...'})
            
            # This would be a longer process, so we emit progress updates
            emit('analysis_progress', {'progress': 25, 'message': 'Extracting frames...'})
            
            # Prepare vision input
            vision_input = ModalityInput(
                modality=ModalityType.VISION,
                data=str(filepath),
                confidence=0.8,
                metadata={'analysis_type': analysis_type}
            )
            
            emit('analysis_progress', {'progress': 75, 'message': 'Analyzing content...'})
            
            # Perform analysis
            fusion_result = self.fusion_system.fuse_multimodal_input([vision_input])
            
            emit('analysis_complete', {
                'response': fusion_result.fused_response,
                'confidence': fusion_result.confidence,
                'processing_time': fusion_result.processing_time,
                'detailed_analysis': fusion_result.detailed_analysis
            })
            
            # Store in session
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id]['conversation'].append({
                    'timestamp': datetime.now().isoformat(),
                    'input': f"Video analysis ({analysis_type})",
                    'response': fusion_result.fused_response,
                    'confidence': fusion_result.confidence,
                    'type': 'video_analysis',
                    'filename': filename
                })
            
        except Exception as e:
            self.logger.error(f"SocketIO video analysis failed: {e}")
            emit('analysis_error', {'error': str(e)})
    
    def _socketio_multimodal_query(self, data):
        """Handle multimodal query via SocketIO"""
        try:
            query = data.get('query', '')
            filename = data.get('filename')
            session_id = data.get('session_id')
            
            if not query:
                emit('query_error', {'error': 'No query provided'})
                return
            
            emit('query_started', {'message': 'Processing multimodal query...'})
            
            # Prepare inputs
            modality_inputs = [
                ModalityInput(modality=ModalityType.TEXT, data=query, confidence=1.0)
            ]
            
            if filename:
                filepath = self.upload_folder / filename
                if filepath.exists():
                    modality_inputs.append(
                        ModalityInput(modality=ModalityType.VISION, data=str(filepath), confidence=0.9)
                    )
            
            # Add context if available
            if session_id and session_id in self.active_sessions:
                context_data = {
                    'conversation_history': self.active_sessions[session_id]['conversation'][-3:]
                }
                modality_inputs.append(
                    ModalityInput(modality=ModalityType.CONTEXT, data=context_data, confidence=0.6)
                )
            
            # Perform fusion
            fusion_result = self.fusion_system.fuse_multimodal_input(
                modality_inputs,
                fusion_strategy=MultimodalFusionStrategy.ADAPTIVE
            )
            
            emit('query_complete', {
                'response': fusion_result.fused_response,
                'confidence': fusion_result.confidence,
                'modalities_used': [m.value for m in fusion_result.modalities_used],
                'reasoning': fusion_result.reasoning
            })
            
        except Exception as e:
            self.logger.error(f"SocketIO multimodal query failed: {e}")
            emit('query_error', {'error': str(e)})
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return Path(filename).suffix.lower() in self.allowed_extensions
    
    def _get_file_info(self, filepath: Path) -> Dict[str, Any]:
        """Get file information"""
        try:
            stat = filepath.stat()
            info = {
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': filepath.suffix.lower()
            }
            
            # Add media-specific info
            if info['extension'] in self.allowed_image_extensions:
                try:
                    with Image.open(filepath) as img:
                        info.update({
                            'type': 'image',
                            'dimensions': img.size,
                            'mode': img.mode,
                            'format': img.format
                        })
                except Exception as e:
                    self.logger.warning(f"Could not get image info for {filepath}: {e}")
                    info['type'] = 'image'
            
            elif info['extension'] in self.allowed_video_extensions:
                try:
                    cap = cv2.VideoCapture(str(filepath))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    info.update({
                        'type': 'video',
                        'dimensions': [width, height],
                        'fps': fps,
                        'frame_count': frame_count,
                        'duration': duration
                    })
                except Exception as e:
                    self.logger.warning(f"Could not get video info for {filepath}: {e}")
                    info['type'] = 'video'
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {filepath}: {e}")
            return {'error': str(e)}
    
    def _get_main_template(self) -> str:
        """Get main HTML template"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS Multimodal AI Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #4a5568;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .interface-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .upload-section, .chat-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border: 2px dashed #ddd;
        }
        
        .upload-section h3, .chat-section h3 {
            margin-top: 0;
            color: #4a5568;
        }
        
        .file-input {
            margin-bottom: 15px;
        }
        
        .file-input input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
        }
        
        .text-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 15px;
            min-height: 100px;
            resize: vertical;
        }
        
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .results-section h3 {
            margin-top: 0;
            color: #4a5568;
        }
        
        .result-item {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .confidence {
            display: inline-block;
            padding: 3px 8px;
            background: #e2e8f0;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        
        .confidence.high { background: #c6f6d5; color: #22543d; }
        .confidence.medium { background: #fef5e7; color: #744210; }
        .confidence.low { background: #fed7d7; color: #742a2a; }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        @media (max-width: 768px) {
            .interface-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 JARVIS Multimodal AI</h1>
            <p>Upload images/videos and interact with advanced AI vision capabilities</p>
        </div>
        
        <div class="interface-grid">
            <div class="upload-section">
                <h3>📁 File Upload & Analysis</h3>
                <div class="file-input">
                    <input type="file" id="fileInput" accept="image/*,video/*" />
                </div>
                <textarea id="questionInput" class="text-input" placeholder="Ask a question about your image/video (optional)..."></textarea>
                <button class="btn" onclick="analyzeFile()">🔍 Analyze</button>
            </div>
            
            <div class="chat-section">
                <h3>💬 Text Chat</h3>
                <textarea id="chatInput" class="text-input" placeholder="Type your message here..."></textarea>
                <button class="btn" onclick="sendChat()">💭 Send</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div id="loadingMessage">Processing...</div>
            <div class="progress-bar">
                <div class="progress-bar-fill" id="progressBar"></div>
            </div>
        </div>
        
        <div class="results-section">
            <h3>📊 Results</h3>
            <div id="results">
                <p>Upload a file or send a message to get started!</p>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        let sessionId = null;
        
        // Socket event handlers
        socket.on('session_created', (data) => {
            sessionId = data.session_id;
            console.log('Session created:', sessionId);
        });
        
        socket.on('analysis_started', (data) => {
            showLoading(data.message);
        });
        
        socket.on('analysis_progress', (data) => {
            updateProgress(data.progress, data.message);
        });
        
        socket.on('analysis_complete', (data) => {
            hideLoading();
            displayResult(data, 'File Analysis');
        });
        
        socket.on('analysis_error', (data) => {
            hideLoading();
            displayError(data.error);
        });
        
        socket.on('query_complete', (data) => {
            hideLoading();
            displayResult(data, 'Multimodal Query');
        });
        
        socket.on('query_error', (data) => {
            hideLoading();
            displayError(data.error);
        });
        
        // UI functions
        function showLoading(message = 'Processing...') {
            document.getElementById('loading').classList.add('show');
            document.getElementById('loadingMessage').textContent = message;
            document.getElementById('progressBar').style.width = '0%';
        }
        
        function hideLoading() {
            document.getElementById('loading').classList.remove('show');
        }
        
        function updateProgress(progress, message) {
            document.getElementById('progressBar').style.width = progress + '%';
            document.getElementById('loadingMessage').textContent = message;
        }
        
        function displayResult(data, title) {
            const resultsDiv = document.getElementById('results');
            
            const confidenceClass = data.confidence > 0.7 ? 'high' : data.confidence > 0.4 ? 'medium' : 'low';
            
            const resultHtml = `
                <div class="result-item">
                    <h4>${title} <span class="confidence ${confidenceClass}">${Math.round(data.confidence * 100)}% confident</span></h4>
                    <p><strong>Response:</strong> ${data.response}</p>
                    ${data.modalities_used ? `<p><strong>Modalities:</strong> ${data.modalities_used.join(', ')}</p>` : ''}
                    ${data.reasoning ? `<p><strong>Reasoning:</strong> ${data.reasoning}</p>` : ''}
                    ${data.processing_time ? `<p><strong>Processing Time:</strong> ${data.processing_time.toFixed(2)}s</p>` : ''}
                </div>
            `;
            
            resultsDiv.innerHTML = resultHtml + resultsDiv.innerHTML;
        }
        
        function displayError(error) {
            const resultsDiv = document.getElementById('results');
            
            const errorHtml = `
                <div class="result-item" style="border-left-color: #e53e3e;">
                    <h4>❌ Error</h4>
                    <p>${error}</p>
                </div>
            `;
            
            resultsDiv.innerHTML = errorHtml + resultsDiv.innerHTML;
        }
        
        // Main functions
        async function analyzeFile() {
            const fileInput = document.getElementById('fileInput');
            const questionInput = document.getElementById('questionInput');
            
            if (!fileInput.files[0]) {
                alert('Please select a file first');
                return;
            }
            
            const file = fileInput.files[0];
            const question = questionInput.value.trim();
            
            // Upload file first
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);
            
            showLoading('Uploading file...');
            
            try {
                const uploadResponse = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const uploadResult = await uploadResponse.json();
                
                if (!uploadResult.success) {
                    throw new Error(uploadResult.error);
                }
                
                // Analyze file via Socket.IO
                const isVideo = file.type.startsWith('video/');
                
                if (isVideo) {
                    socket.emit('analyze_video', {
                        filename: uploadResult.filename,
                        analysis_type: 'comprehensive',
                        session_id: sessionId
                    });
                } else {
                    socket.emit('analyze_image', {
                        filename: uploadResult.filename,
                        question: question,
                        session_id: sessionId
                    });
                }
                
            } catch (error) {
                hideLoading();
                displayError(error.message);
            }
        }
        
        async function sendChat() {
            const chatInput = document.getElementById('chatInput');
            const message = chatInput.value.trim();
            
            if (!message) {
                alert('Please enter a message');
                return;
            }
            
            showLoading('Processing message...');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });
                
                const result = await response.json();
                
                hideLoading();
                
                if (result.success) {
                    displayResult(result, 'Chat Response');
                    chatInput.value = '';
                } else {
                    displayError(result.error);
                }
                
            } catch (error) {
                hideLoading();
                displayError(error.message);
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                const activeElement = document.activeElement;
                if (activeElement.id === 'chatInput') {
                    sendChat();
                } else if (activeElement.id === 'questionInput') {
                    analyzeFile();
                }
            }
        });
    </script>
</body>
</html>
        '''
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the web interface"""
        self.logger.info(f"Starting multimodal web interface on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def shutdown(self):
        """Shutdown web interface"""
        self.logger.info("Shutting down multimodal web interface")
        
        # Clean up sessions
        self.active_sessions.clear()
        
        # Shutdown multimodal system
        if self.fusion_system:
            self.fusion_system.shutdown()
        
        self.logger.info("Multimodal web interface shutdown complete")