#!/usr/bin/env python3
"""
Simple Web App Test for JARVIS Voice Assistant
Tests only the AI engine with web interface
"""

import sys
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = Flask(__name__)
app.secret_key = 'test-key'

# Global AI engine
ai_engine = None

def initialize_ai():
    """Initialize only the AI engine"""
    global ai_engine
    try:
        from ai.ai_engine import AIEngine
        from system.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()  # Get all configuration
        ai_engine = AIEngine(config)
        
        print("✅ AI engine initialized for web app")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize AI engine: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>JARVIS Voice Assistant - Simple Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #0f0f0f; color: #00ff00; }
            .container { max-width: 800px; margin: 0 auto; }
            .chat-container { border: 1px solid #00ff00; height: 400px; overflow-y: auto; padding: 10px; margin: 20px 0; background: #1a1a1a; }
            .message { margin: 10px 0; padding: 5px; }
            .user { color: #00ffff; }
            .jarvis { color: #00ff00; }
            input[type="text"] { width: 70%; padding: 10px; background: #2a2a2a; color: #00ff00; border: 1px solid #00ff00; }
            button { padding: 10px 20px; background: #004400; color: #00ff00; border: 1px solid #00ff00; cursor: pointer; }
            button:hover { background: #006600; }
            .status { margin: 10px 0; padding: 10px; background: #2a2a2a; border-left: 3px solid #00ff00; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 JARVIS Voice Assistant - Simple Test</h1>
            
            <div class="status" id="status">
                AI Engine Status: <span id="engine-status">Checking...</span>
            </div>
            
            <div class="chat-container" id="chat">
                <div class="message jarvis">JARVIS: Hello! I'm ready to assist you.</div>
            </div>
            
            <div>
                <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
                <button onclick="clearChat()">Clear</button>
            </div>
            
            <div class="status">
                <h3>Test Commands:</h3>
                <ul>
                    <li>Hello</li>
                    <li>What time is it?</li>
                    <li>Who are you?</li>
                    <li>สวัสดี (Thai greeting)</li>
                    <li>Help me</li>
                </ul>
            </div>
        </div>

        <script>
            // Check engine status on load
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('engine-status').textContent = 
                        data.ready ? '✅ Ready' : '❌ Not Ready';
                });

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessage('user', 'You: ' + message);
                input.value = '';
                
                // Show thinking indicator
                const thinkingId = addMessage('jarvis', 'JARVIS: Thinking...');
                
                // Send to server
                fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                })
                .then(response => response.json())
                .then(data => {
                    // Remove thinking indicator
                    document.getElementById(thinkingId).remove();
                    
                    if (data.success) {
                        addMessage('jarvis', 'JARVIS: ' + data.response);
                    } else {
                        addMessage('jarvis', 'JARVIS: Error - ' + data.error);
                    }
                })
                .catch(error => {
                    document.getElementById(thinkingId).remove();
                    addMessage('jarvis', 'JARVIS: Connection error');
                });
            }

            function addMessage(type, text) {
                const chat = document.getElementById('chat');
                const messageDiv = document.createElement('div');
                const messageId = 'msg-' + Date.now();
                messageDiv.id = messageId;
                messageDiv.className = 'message ' + type;
                messageDiv.textContent = text;
                chat.appendChild(messageDiv);
                chat.scrollTop = chat.scrollHeight;
                return messageId;
            }

            function clearChat() {
                const chat = document.getElementById('chat');
                chat.innerHTML = '<div class="message jarvis">JARVIS: Chat cleared. How can I help you?</div>';
            }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/status')
def status():
    """Get AI engine status"""
    if ai_engine:
        return jsonify({
            'ready': ai_engine.is_ready,
            'processing': ai_engine.is_processing,
            'info': ai_engine.get_engine_info()
        })
    else:
        return jsonify({'ready': False, 'error': 'AI engine not initialized'})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'error': 'Empty message'})
        
        if not ai_engine or not ai_engine.is_ready:
            return jsonify({'success': False, 'error': 'AI engine not ready'})
        
        # Set up response handler
        response_data = {'received': False, 'text': '', 'error': ''}
        
        def on_response(text, metadata):
            response_data['received'] = True
            response_data['text'] = text
        
        def on_error(error):
            response_data['received'] = True
            response_data['error'] = str(error)
        
        # Connect signals
        ai_engine.response_ready.connect(on_response)
        ai_engine.error_occurred.connect(on_error)
        
        # Detect language
        thai_chars = "กขคงจฉชซณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ"
        is_thai = any(char in thai_chars for char in message)
        language = "th" if is_thai else "en"
        
        # Process query
        ai_engine.process_query(message, language)
        
        # Wait for response (simple polling)
        wait_time = 0
        while not response_data['received'] and wait_time < 10:
            time.sleep(0.1)
            wait_time += 0.1
        
        # Disconnect signals
        ai_engine.response_ready.disconnect(on_response)
        ai_engine.error_occurred.disconnect(on_error)
        
        if response_data['received']:
            if response_data['error']:
                return jsonify({'success': False, 'error': response_data['error']})
            else:
                return jsonify({'success': True, 'response': response_data['text']})
        else:
            return jsonify({'success': False, 'error': 'Response timeout'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting JARVIS Simple Web Test...")
    
    if initialize_ai():
        print("🌐 Starting web server...")
        print("📡 Open http://localhost:5001 to test")
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("❌ Cannot start web app - AI engine initialization failed")
        sys.exit(1)