#!/usr/bin/env python3
"""
🤖 JARVIS Voice Assistant Launcher
"""

import sys
import os
from pathlib import Path

# Add src to path
jarvis_root = Path(__file__).parent
sys.path.insert(0, str(jarvis_root / "src"))

def main():
    try:
        from jarvis_complete_voice_system import JarvisVoiceSystem
        
        print("🚀 Starting JARVIS Voice Assistant...")
        jarvis = JarvisVoiceSystem()
        
        if jarvis.start_system():
            print("✅ JARVIS is now active!")
            print("🎙️ Say 'Hey JARVIS' to start a conversation")
            
            try:
                # Keep running
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down JARVIS...")
                jarvis.stop_system()
                print("👋 Goodbye!")
        else:
            print("❌ Failed to start JARVIS")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python install.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
