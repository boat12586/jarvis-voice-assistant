#!/usr/bin/env python3
"""
Test script for Voice Command Router
Tests command parsing, routing, and execution
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice.command_parser import VoiceCommandParser, CommandType, CommandPriority
from voice.command_router import VoiceCommandRouter
from PyQt6.QtCore import QCoreApplication
import yaml

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_test_config():
    """Load test configuration"""
    return {
        "command_parser": {
            "thai_support": True,
            "confidence_threshold": 0.5
        },
        "command_router": {
            "max_queue_size": 10,
            "max_history": 50
        }
    }

def test_command_parsing():
    """Test command parsing functionality"""
    print("🎯 Testing Command Parser...")
    
    config = load_test_config()
    parser = VoiceCommandParser(config)
    
    # Test commands
    test_commands = [
        ("Hello JARVIS", "greeting"),
        ("What is artificial intelligence?", "information_request"),
        ("How do I use this feature?", "how_to_request"),
        ("Open the settings", "action_request"),
        ("Turn up the volume", "system_control"),
        ("สวัสดีครับ จาร์วิส", "greeting"),
        ("ปัญญาประดิษฐ์คืออะไร", "information_request"),
        ("ช่วยเปิดการตั้งค่าหน่อยครับ", "action_request"),
    ]
    
    results = []
    for command_text, expected_intent in test_commands:
        parsed = parser.parse_command(command_text)
        success = parsed.intent == expected_intent
        results.append(success)
        
        print(f"  Command: '{command_text}'")
        print(f"    Intent: {parsed.intent} (expected: {expected_intent}) {'✅' if success else '❌'}")
        print(f"    Confidence: {parsed.confidence:.3f}")
        print(f"    Type: {parsed.command_type.value}")
        print(f"    Language: {parsed.language}")
        if parsed.entities:
            print(f"    Entities: {parsed.entities}")
        if parsed.parameters:
            print(f"    Parameters: {parsed.parameters}")
        print()
    
    accuracy = sum(results) / len(results) * 100
    print(f"📊 Parser Accuracy: {accuracy:.1f}% ({sum(results)}/{len(results)})")
    return accuracy > 80

async def test_command_routing():
    """Test command routing and execution"""
    print("🚀 Testing Command Router...")
    
    config = load_test_config()
    parser = VoiceCommandParser(config)
    router = VoiceCommandRouter(config)
    
    # Test command execution
    test_commands = [
        "Hello JARVIS",
        "What is machine learning?",
        "Open the application",
        "Turn up the volume",
        "สวัสดีครับ",
    ]
    
    execution_results = []
    
    for command_text in test_commands:
        print(f"\n  Testing: '{command_text}'")
        
        # Parse command
        parsed_command = parser.parse_command(command_text)
        print(f"    Parsed as: {parsed_command.intent}")
        
        # Route command
        execution_id = router.route_command(parsed_command)
        
        if execution_id:
            print(f"    Execution ID: {execution_id}")
            
            # Wait for completion (with timeout)
            for i in range(50):  # 5 second timeout
                await asyncio.sleep(0.1)
                status = router.get_execution_status(execution_id)
                if status and status['status'] in ['completed', 'failed']:
                    break
            
            # Check final status
            final_status = router.get_execution_status(execution_id)
            if final_status:
                print(f"    Status: {final_status['status']}")
                if final_status['status'] == 'completed':
                    execution_results.append(True)
                    print(f"    Result: {final_status.get('result', {})}")
                else:
                    execution_results.append(False)
                    print(f"    Error: {final_status.get('error', 'Unknown error')}")
            else:
                execution_results.append(False)
                print("    ❌ No status found")
        else:
            execution_results.append(False)
            print("    ❌ Failed to route command")
    
    # Router statistics
    stats = router.get_router_stats()
    print(f"\n📊 Router Statistics:")
    print(f"  Registered handlers: {stats['registered_handlers']}")
    print(f"  Supported intents: {len(stats['supported_intents'])}")
    print(f"  Commands processed: {stats['statistics']['commands_processed']}")
    print(f"  Success rate: {stats['statistics']['commands_successful']}/{stats['statistics']['commands_processed']}")
    
    success_rate = sum(execution_results) / len(execution_results) * 100
    print(f"🎯 Execution Success Rate: {success_rate:.1f}%")
    
    return success_rate > 80

def test_handler_registration():
    """Test custom handler registration"""
    print("🔧 Testing Custom Handler Registration...")
    
    from voice.command_router import CommandHandler
    
    class TestHandler(CommandHandler):
        def __init__(self):
            super().__init__("test_handler", ["test_intent"])
        
        async def handle(self, command):
            return {
                "type": "test_response",
                "message": f"Test handler processed: {command.cleaned_text}",
                "success": True
            }
    
    config = load_test_config()
    router = VoiceCommandRouter(config)
    
    # Register custom handler
    custom_handler = TestHandler()
    success = router.register_handler(custom_handler)
    
    print(f"  Custom handler registration: {'✅' if success else '❌'}")
    
    # Check handler count
    stats = router.get_router_stats()
    handler_count = stats['registered_handlers']
    print(f"  Total handlers: {handler_count}")
    
    # Unregister handler
    unregister_success = router.unregister_handler("test_handler")
    print(f"  Handler unregistration: {'✅' if unregister_success else '❌'}")
    
    return success and unregister_success

def test_performance():
    """Test command processing performance"""
    print("⚡ Testing Performance...")
    
    import time
    
    config = load_test_config()
    parser = VoiceCommandParser(config)
    router = VoiceCommandRouter(config)
    
    # Test parsing performance
    commands = ["Hello JARVIS"] * 100
    
    start_time = time.time()
    for command in commands:
        parsed = parser.parse_command(command)
    parse_time = time.time() - start_time
    
    print(f"  Parsing 100 commands: {parse_time:.3f}s ({parse_time*10:.1f}ms per command)")
    
    # Test routing performance
    parsed_command = parser.parse_command("Hello JARVIS")
    
    start_time = time.time()
    for _ in range(100):
        execution_id = router.route_command(parsed_command)
    route_time = time.time() - start_time
    
    print(f"  Routing 100 commands: {route_time:.3f}s ({route_time*10:.1f}ms per command)")
    
    return parse_time < 1.0 and route_time < 0.5

async def main():
    """Run all command router tests"""
    print("🧪 Voice Command Router Testing")
    print("=" * 50)
    
    setup_logging()
    
    # Initialize Qt application
    app = QCoreApplication(sys.argv)
    
    try:
        results = []
        
        # Test command parsing
        results.append(test_command_parsing())
        print()
        
        # Test command routing
        results.append(await test_command_routing())
        print()
        
        # Test handler registration
        results.append(test_handler_registration())
        print()
        
        # Test performance
        results.append(test_performance())
        print()
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("=" * 50)
        print(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! Command router is working correctly.")
        else:
            print("❌ Some tests failed. Please check the implementation.")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    exit(0 if success else 1)