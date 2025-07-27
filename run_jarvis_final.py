#!/usr/bin/env python3
"""
🚀 JARVIS Final Launcher
เปิดใช้งาน JARVIS ในโหมดที่สมบูรณ์ที่สุด
"""

import sys
import logging
from pathlib import Path

def main():
    """เรียกใช้ JARVIS ในรูปแบบที่เหมาะสม"""
    
    print("🤖 JARVIS AI Assistant - Final Complete System")
    print("=" * 50)
    print("🎙️ Voice Recognition: Faster-Whisper")
    print("🗣️ Text-to-Speech: Tacotron2")
    print("🧠 Memory: Conversation learning")
    print("👂 Wake Word: 'Hey JARVIS'")
    print("🖥️ Interface: Holographic GUI")
    print("=" * 50)
    
    # ตรวจสอบโหมดที่จะใช้
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nAvailable modes:")
        print("  python3 run_jarvis_final.py gui      - Full GUI mode")
        print("  python3 run_jarvis_final.py voice    - Voice-only mode")
        print("  python3 run_jarvis_final.py text     - Text-only mode")
        print("  python3 run_jarvis_final.py test     - Quick test mode")
        print()
        mode = input("Choose mode (gui/voice/text/test) [gui]: ").lower() or "gui"
    
    # ตั้งค่า logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if mode == "gui":
            print("\n🖥️ Starting JARVIS with GUI interface...")
            from jarvis_complete_gui import main as gui_main
            gui_main()
            
        elif mode == "voice":
            print("\n🎙️ Starting JARVIS voice-only mode...")
            from jarvis_final_complete import main as voice_main
            voice_main()
            
        elif mode == "text":
            print("\n💬 Starting JARVIS text-only mode...")
            from jarvis_simple_working import main as text_main
            text_main()
            
        elif mode == "test":
            print("\n🧪 Running JARVIS test suite...")
            from test_final_complete import test_complete_system
            test_complete_system()
            
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: gui, voice, text, test")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install PyQt6 torch transformers faster-whisper TTS sounddevice")
        
    except KeyboardInterrupt:
        print("\n\n🛑 JARVIS shutdown by user")
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Error starting JARVIS: {e}")
        logging.error(f"JARVIS startup error: {e}")

if __name__ == "__main__":
    main()