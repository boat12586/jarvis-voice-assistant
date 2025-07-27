#!/usr/bin/env python3
"""
🤖 JARVIS Simple Launcher
เวอร์ชันง่ายๆ ที่ใช้ Fallback AI
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🚀 Starting JARVIS Simple Voice Assistant...")
    
    try:
        # Import fallback AI directly
        from ai.fallback_ai import FallbackAI
        
        print("🤖 Initializing JARVIS Fallback AI...")
        jarvis = FallbackAI()
        
        print(f"✅ {jarvis.name} is ready!")
        print("💬 Type your messages below (type 'quit' to exit):")
        print("🇹🇭 รองรับภาษาไทย | 🇺🇸 English supported")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'ออก']:
                    print("👋 Goodbye! | ลาก่อนครับ!")
                    break
                
                if user_input:
                    response = jarvis.generate_response(user_input)
                    print(f"🤖 JARVIS: {response}")
                    print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! | ลาก่อนครับ!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python install.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()