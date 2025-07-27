#!/usr/bin/env python3
"""
Jarvis v2.0 Integration Test Suite
Comprehensive testing of all services and features
"""

import asyncio
import json
import logging
import time
import websockets
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os

# Configuration
CORE_SERVICE_URL = "http://localhost:8000"
AUDIO_SERVICE_URL = "http://localhost:8001"
WEB_SERVICE_URL = "http://localhost:3000"
CORE_WS_URL = "ws://localhost:8000/ws"
AUDIO_WS_URL = "ws://localhost:8001/ws/audio"

# Test user
TEST_USER_ID = "test_user_integration"
TEST_USERNAME = "integration_test"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JarvisIntegrationTest:
    """Comprehensive integration test suite for Jarvis v2.0"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        self.test_user_token = None
        
    def log_test_result(self, test_name: str, success: bool, message: str = "", duration: float = 0.0):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        logger.info(f"{status} {test_name} ({duration:.2f}s) - {message}")
    
    def test_service_health(self):
        """Test all service health endpoints"""
        logger.info("🔍 Testing service health...")
        
        services = [
            ("Core Service", f"{CORE_SERVICE_URL}/api/v2/health"),
            ("Audio Service", f"{AUDIO_SERVICE_URL}/api/v2/audio/health"),
            ("Web Service", f"{WEB_SERVICE_URL}"),
            ("Nginx", "http://localhost/health")
        ]
        
        for service_name, url in services:
            start_time = time.time()
            try:
                response = requests.get(url, timeout=10)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    self.log_test_result(
                        f"{service_name} Health Check",
                        True,
                        f"HTTP {response.status_code}",
                        duration
                    )
                else:
                    self.log_test_result(
                        f"{service_name} Health Check",
                        False,
                        f"HTTP {response.status_code}",
                        duration
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(
                    f"{service_name} Health Check",
                    False,
                    str(e),
                    duration
                )
    
    def test_user_management(self):
        """Test user creation and management"""
        logger.info("👤 Testing user management...")
        
        # Create guest user
        start_time = time.time()
        try:
            response = requests.post(f"{CORE_SERVICE_URL}/api/v2/guest")
            duration = time.time() - start_time
            
            if response.status_code == 200:
                user_data = response.json()
                self.test_user_token = user_data.get("user_id")
                
                self.log_test_result(
                    "Guest User Creation",
                    True,
                    f"Created user: {user_data.get('username')}",
                    duration
                )
                
                # Test user info retrieval
                self.test_user_info()
            else:
                self.log_test_result(
                    "Guest User Creation",
                    False,
                    f"HTTP {response.status_code}",
                    duration
                )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Guest User Creation",
                False,
                str(e),
                duration
            )
    
    def test_user_info(self):
        """Test user information retrieval"""
        if not self.test_user_token:
            return
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            response = requests.get(f"{CORE_SERVICE_URL}/api/v2/users/me", headers=headers)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                user_data = response.json()
                self.log_test_result(
                    "User Info Retrieval",
                    True,
                    f"Retrieved user: {user_data.get('username')}",
                    duration
                )
            else:
                self.log_test_result(
                    "User Info Retrieval",
                    False,
                    f"HTTP {response.status_code}",
                    duration
                )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "User Info Retrieval",
                False,
                str(e),
                duration
            )
    
    def test_plugin_system(self):
        """Test plugin system functionality"""
        logger.info("🔌 Testing plugin system...")
        
        # Test plugin listing
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            response = requests.get(f"{CORE_SERVICE_URL}/api/v2/plugins", headers=headers)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                plugins_data = response.json()
                plugin_count = len(plugins_data.get("plugins", []))
                command_count = len(plugins_data.get("commands", {}))
                
                self.log_test_result(
                    "Plugin System Listing",
                    True,
                    f"Found {plugin_count} plugins, {command_count} commands",
                    duration
                )
                
                # Test specific plugin commands
                self.test_plugin_commands()
            else:
                self.log_test_result(
                    "Plugin System Listing",
                    False,
                    f"HTTP {response.status_code}",
                    duration
                )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Plugin System Listing",
                False,
                str(e),
                duration
            )
    
    def test_plugin_commands(self):
        """Test plugin command execution"""
        if not self.test_user_token:
            return
        
        commands = [
            ("/help", "Help command"),
            ("/time", "Time command"),
            ("/calc 2+2", "Calculator command"),
            ("hello", "Greeting detection"),
            ("thank you", "Courtesy response")
        ]
        
        for command, description in commands:
            start_time = time.time()
            try:
                headers = {"Authorization": f"Bearer {self.test_user_token}"}
                payload = {
                    "message": command,
                    "user_id": self.test_user_token
                }
                
                response = requests.post(
                    f"{CORE_SERVICE_URL}/api/v2/chat",
                    json=payload,
                    headers=headers
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    chat_data = response.json()
                    response_text = chat_data.get("response", "")
                    
                    self.log_test_result(
                        f"Plugin Command: {command}",
                        True,
                        f"Response: {response_text[:50]}...",
                        duration
                    )
                else:
                    self.log_test_result(
                        f"Plugin Command: {command}",
                        False,
                        f"HTTP {response.status_code}",
                        duration
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(
                    f"Plugin Command: {command}",
                    False,
                    str(e),
                    duration
                )
    
    def test_audio_service(self):
        """Test audio service functionality"""
        logger.info("🎵 Testing audio service...")
        
        if not self.test_user_token:
            return
        
        # Test audio session start
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            response = requests.post(
                f"{AUDIO_SERVICE_URL}/api/v2/audio/session/start",
                headers=headers
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                session_data = response.json()
                session_id = session_data.get("session_id")
                
                self.log_test_result(
                    "Audio Session Start",
                    True,
                    f"Session ID: {session_id}",
                    duration
                )
                
                # Test TTS
                self.test_tts()
                
                # Test audio session stop
                self.test_audio_session_stop()
            else:
                self.log_test_result(
                    "Audio Session Start",
                    False,
                    f"HTTP {response.status_code}",
                    duration
                )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Audio Session Start",
                False,
                str(e),
                duration
            )
    
    def test_tts(self):
        """Test text-to-speech functionality"""
        if not self.test_user_token:
            return
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            payload = {
                "text": "Hello, this is a test of the text-to-speech system.",
                "voice": "en-US-AriaNeural"
            }
            
            response = requests.post(
                f"{AUDIO_SERVICE_URL}/api/v2/audio/tts",
                json=payload,
                headers=headers
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                tts_data = response.json()
                
                self.log_test_result(
                    "Text-to-Speech",
                    True,
                    f"Duration: {tts_data.get('duration', 0):.2f}s",
                    duration
                )
            else:
                self.log_test_result(
                    "Text-to-Speech",
                    False,
                    f"HTTP {response.status_code}",
                    duration
                )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Text-to-Speech",
                False,
                str(e),
                duration
            )
    
    def test_audio_session_stop(self):
        """Test audio session stop"""
        if not self.test_user_token:
            return
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            response = requests.post(
                f"{AUDIO_SERVICE_URL}/api/v2/audio/session/stop",
                headers=headers
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test_result(
                    "Audio Session Stop",
                    True,
                    "Session stopped successfully",
                    duration
                )
            else:
                self.log_test_result(
                    "Audio Session Stop",
                    False,
                    f"HTTP {response.status_code}",
                    duration
                )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Audio Session Stop",
                False,
                str(e),
                duration
            )
    
    async def test_websocket_connections(self):
        """Test WebSocket connectivity"""
        logger.info("🔗 Testing WebSocket connections...")
        
        if not self.test_user_token:
            return
        
        # Test core WebSocket
        await self.test_core_websocket()
        
        # Test audio WebSocket
        await self.test_audio_websocket()
    
    async def test_core_websocket(self):
        """Test core service WebSocket"""
        start_time = time.time()
        try:
            uri = f"{CORE_WS_URL}/{self.test_user_token}"
            
            async with websockets.connect(uri) as websocket:
                # Send ping
                await websocket.send(json.dumps({"type": "ping"}))
                
                # Wait for pong
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                duration = time.time() - start_time
                
                if data.get("type") == "pong":
                    self.log_test_result(
                        "Core WebSocket Connection",
                        True,
                        "Ping/Pong successful",
                        duration
                    )
                else:
                    self.log_test_result(
                        "Core WebSocket Connection",
                        False,
                        f"Unexpected response: {data}",
                        duration
                    )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Core WebSocket Connection",
                False,
                str(e),
                duration
            )
    
    async def test_audio_websocket(self):
        """Test audio service WebSocket"""
        start_time = time.time()
        try:
            uri = f"{AUDIO_WS_URL}/{self.test_user_token}"
            
            async with websockets.connect(uri) as websocket:
                # Send ping
                await websocket.send(json.dumps({"type": "ping"}))
                
                # Wait for pong
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                duration = time.time() - start_time
                
                if data.get("type") == "pong":
                    self.log_test_result(
                        "Audio WebSocket Connection",
                        True,
                        "Ping/Pong successful",
                        duration
                    )
                else:
                    self.log_test_result(
                        "Audio WebSocket Connection",
                        False,
                        f"Unexpected response: {data}",
                        duration
                    )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Audio WebSocket Connection",
                False,
                str(e),
                duration
            )
    
    def test_system_metrics(self):
        """Test system metrics and monitoring"""
        logger.info("📊 Testing system metrics...")
        
        if not self.test_user_token:
            return
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.test_user_token}"}
            response = requests.get(
                f"{CORE_SERVICE_URL}/api/v2/admin/stats",
                headers=headers
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                stats_data = response.json()
                
                self.log_test_result(
                    "System Metrics",
                    True,
                    f"Uptime: {stats_data.get('uptime_seconds', 0)}s",
                    duration
                )
            else:
                self.log_test_result(
                    "System Metrics",
                    False,
                    f"HTTP {response.status_code}",
                    duration
                )
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "System Metrics",
                False,
                str(e),
                duration
            )
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("🎯 JARVIS v2.0 INTEGRATION TEST REPORT")
        print("="*80)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🧪 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print()
        
        if failed_tests > 0:
            print("❌ FAILED TESTS:")
            print("-" * 40)
            for result in self.test_results:
                if not result["success"]:
                    print(f"• {result['test_name']}: {result['message']}")
            print()
        
        print("📋 DETAILED RESULTS:")
        print("-" * 40)
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test_name']} ({result['duration']:.2f}s)")
            if result["message"]:
                print(f"   {result['message']}")
        
        print("\n" + "="*80)
        
        # Save detailed report
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": self.test_results
        }
        
        with open("integration_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: integration_test_report.json")
        print("="*80)
        
        return success_rate > 80  # Return True if success rate > 80%
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting Jarvis v2.0 Integration Test Suite...")
        
        # Wait for services to be ready
        logger.info("⏳ Waiting for services to be ready...")
        await asyncio.sleep(10)
        
        # Run tests
        self.test_service_health()
        self.test_user_management()
        self.test_plugin_system()
        self.test_audio_service()
        await self.test_websocket_connections()
        self.test_system_metrics()
        
        # Generate report
        success = self.generate_report()
        
        if success:
            logger.info("🎉 Integration tests completed successfully!")
            return 0
        else:
            logger.error("❌ Integration tests failed!")
            return 1

async def main():
    """Main test runner"""
    test_suite = JarvisIntegrationTest()
    return await test_suite.run_all_tests()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)