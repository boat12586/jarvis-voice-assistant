#!/usr/bin/env python3
"""
Run script for Jarvis Voice Assistant
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Run the application
if __name__ == "__main__":
    from main import main
    main()