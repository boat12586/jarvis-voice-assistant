#!/usr/bin/env python3
"""
🖥️ JARVIS GUI Launcher
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    try:
        from ui.holographic_interface import test_holographic_interface
        app, interface = test_holographic_interface()
        app.exec()
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
