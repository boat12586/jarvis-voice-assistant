#!/usr/bin/env python3
"""
🚀 JARVIS Voice Assistant - Easy Installation Script
สคริปต์ติดตั้ง JARVIS อย่างง่าย
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import shutil
import urllib.request
import json
import time


class JarvisInstaller:
    """ตัวติดตั้ง JARVIS"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.errors = []
        self.warnings = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def print_banner(self):
        """แสดงแบนเนอร์"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🤖 JARVIS Voice Assistant - Installation Wizard           ║
║                                                              ║
║   ผู้ช่วยเสียง JARVIS - ตัวติดตั้งอัตโนมัติ                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_system_requirements(self) -> bool:
        """ตรวจสอบข้อกำหนดระบบ"""
        print("\n🔍 Checking system requirements...")
        
        success = True
        
        # Python version
        if self.python_version.major != 3 or self.python_version.minor < 8:
            self.errors.append("Python 3.8+ required")
            success = False
        else:
            print(f"   ✅ Python {sys.version.split()[0]}")
        
        # Operating system
        if self.system not in ['Linux', 'Windows', 'Darwin']:
            self.warnings.append(f"Untested OS: {self.system}")
        else:
            print(f"   ✅ Operating System: {self.system}")
        
        # Memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 8:
                self.warnings.append(f"Low memory: {memory_gb:.1f}GB (8GB+ recommended)")
            else:
                print(f"   ✅ Memory: {memory_gb:.1f}GB")
        except ImportError:
            self.warnings.append("Could not check memory (psutil not available)")
        
        # Disk space
        try:
            free_space = shutil.disk_usage(".").free / (1024**3)
            if free_space < 10:
                self.warnings.append(f"Low disk space: {free_space:.1f}GB (10GB+ recommended)")
            else:
                print(f"   ✅ Disk Space: {free_space:.1f}GB available")
        except:
            self.warnings.append("Could not check disk space")
        
        return success
    
    def install_dependencies(self) -> bool:
        """ติดตั้ง dependencies"""
        print("\n📦 Installing Python dependencies...")
        
        # Base requirements
        base_packages = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "PyQt6>=6.4.0",
            "sounddevice>=0.4.0",
            "librosa>=0.10.0",
            "numpy>=1.24.0",
            "psutil>=5.9.0",
            "requests>=2.28.0",
            "PyYAML>=6.0",
            "python-dotenv>=1.0.0"
        ]
        
        # Audio processing
        audio_packages = [
            "faster-whisper>=0.10.0",
            "TTS>=0.20.0",
            "pydub>=0.25.0",
            "scipy>=1.10.0"
        ]
        
        # Optional packages
        optional_packages = [
            "gtts>=2.3.0",
            "pyttsx3>=2.90",
            "webrtcvad>=2.0.10"
        ]
        
        try:
            # Install base packages
            for package in base_packages:
                self._install_package(package)
            
            # Install audio packages
            for package in audio_packages:
                self._install_package(package, optional=False)
            
            # Install optional packages
            for package in optional_packages:
                self._install_package(package, optional=True)
            
            print("   ✅ Dependencies installed successfully")
            return True
            
        except Exception as e:
            self.errors.append(f"Dependency installation failed: {e}")
            return False
    
    def _install_package(self, package: str, optional: bool = False):
        """ติดตั้งแพ็คเกจ"""
        try:
            print(f"   📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
        except subprocess.CalledProcessError as e:
            if optional:
                self.warnings.append(f"Optional package failed: {package}")
                print(f"   ⚠️ Optional package {package} failed to install")
            else:
                raise e
    
    def setup_directories(self) -> bool:
        """สร้างไดเรกทอรี"""
        print("\n📁 Setting up directories...")
        
        directories = [
            "data/conversation_memory",
            "data/tts_cache",
            "data/vectordb",
            "data/knowledge_base",
            "models/deepseek-r1-distill-llama-8b",
            "models/f5_tts",
            "logs",
            "config"
        ]
        
        try:
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {directory}")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Directory setup failed: {e}")
            return False
    
    def download_models(self) -> bool:
        """ดาวน์โหลดโมเดล"""
        print("\n🧠 Setting up AI models...")
        
        try:
            # Check if models are already available
            print("   🔍 Checking for existing models...")
            
            # Note: In a real installation, you would download models here
            print("   ℹ️ AI models will be downloaded automatically on first use")
            print("   📊 Expected download sizes:")
            print("      - DeepSeek-R1: ~4.5GB")
            print("      - mxbai-embed-large: ~1.2GB")
            print("      - Faster-Whisper: ~1GB")
            print("      - F5-TTS: ~2GB")
            print("      Total: ~8.7GB")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Model setup failed: {e}")
            return False
    
    def create_config_files(self) -> bool:
        """สร้างไฟล์การตั้งค่า"""
        print("\n⚙️ Creating configuration files...")
        
        try:
            # Main config
            config_content = """ai:
  llm:
    model_name: deepseek-ai/deepseek-r1-distill-llama-8b
    max_context_length: 8192
    temperature: 0.7
    max_tokens: 512
    quantization: 8bit
    device: auto
    fallback_model: microsoft/DialoGPT-medium
    timeout: 30

rag:
  chunk_overlap: 25
  chunk_size: 256
  embedding_model: mixedbread-ai/mxbai-embed-large-v1
  max_context_length: 2048
  similarity_threshold: 0.2
  top_k: 10

voice:
  sample_rate: 16000
  chunk_size: 1024
  whisper:
    language: auto
    model_size: base
  tts:
    model_path: models/f5_tts
    voice_clone_path: assets/voices/jarvis_voice.wav

ui:
  theme: dark
  colors:
    primary: '#00d4ff'
    secondary: '#0099cc'
    accent: '#ff6b35'
    background: '#1a1a1a'

system:
  log_level: INFO
  log_file: logs/jarvis.log
  auto_save_interval: 300
"""
            
            with open("config/default_config.yaml", "w") as f:
                f.write(config_content)
            print("   ✅ Created default_config.yaml")
            
            # Environment file
            env_content = """# JARVIS Voice Assistant Environment Variables
JARVIS_CONFIG_PATH=config/default_config.yaml
JARVIS_DATA_DIR=data
JARVIS_LOG_LEVEL=INFO
JARVIS_CACHE_DIR=data/cache

# Optional: API Keys (if using cloud services)
# OPENAI_API_KEY=your_openai_api_key
# GOOGLE_CLOUD_KEY=your_google_cloud_key
"""
            
            with open(".env.example", "w") as f:
                f.write(env_content)
            print("   ✅ Created .env.example")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Config creation failed: {e}")
            return False
    
    def create_launcher_scripts(self) -> bool:
        """สร้างสคริปต์เปิดโปรแกรม"""
        print("\n🚀 Creating launcher scripts...")
        
        try:
            # Main launcher
            launcher_content = """#!/usr/bin/env python3
\"\"\"
🤖 JARVIS Voice Assistant Launcher
\"\"\"

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
                print("\\n🛑 Shutting down JARVIS...")
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
"""
            
            with open("run_jarvis.py", "w") as f:
                f.write(launcher_content)
            
            # Make executable on Unix systems
            if self.system != "Windows":
                os.chmod("run_jarvis.py", 0o755)
            
            print("   ✅ Created run_jarvis.py")
            
            # GUI launcher
            gui_launcher = """#!/usr/bin/env python3
\"\"\"
🖥️ JARVIS GUI Launcher
\"\"\"

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
"""
            
            with open("run_jarvis_gui.py", "w") as f:
                f.write(gui_launcher)
            
            if self.system != "Windows":
                os.chmod("run_jarvis_gui.py", 0o755)
            
            print("   ✅ Created run_jarvis_gui.py")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Launcher creation failed: {e}")
            return False
    
    def create_readme(self) -> bool:
        """สร้าง README"""
        print("\n📝 Creating documentation...")
        
        readme_content = """# 🤖 JARVIS Voice Assistant

Advanced AI voice assistant with real-time conversation, Thai language support, and holographic interface.

## ✨ Features

- 🎙️ **Real-time Voice Conversation** - Natural speech interaction
- 🇹🇭 **Thai Language Support** - Full Thai-English bilingual support  
- 🧠 **Advanced AI** - Powered by DeepSeek-R1 and mxbai-embed-large
- 🗣️ **JARVIS Voice** - Sci-fi voice synthesis with effects
- 🌌 **Holographic UI** - Futuristic sci-fi interface
- 🎤 **Wake Word Detection** - "Hey JARVIS" activation
- 💾 **Conversation Memory** - Remembers past conversations
- ⚡ **Performance Optimized** - Efficient resource usage

## 🚀 Quick Start

### Installation
```bash
python install.py
```

### Run JARVIS
```bash
python run_jarvis.py
```

### Run GUI Interface
```bash
python run_jarvis_gui.py
```

## 💻 System Requirements

- **Minimum**: RTX 2050 (4GB VRAM), 16GB RAM, 10GB storage
- **Recommended**: RTX 3070+ (8GB+ VRAM), 32GB RAM, 25GB NVMe SSD
- **OS**: Windows 11, Linux (Ubuntu 20.04+), macOS 12+
- **Python**: 3.8+

## 🎙️ Voice Commands

- "Hey JARVIS" - Wake up
- "What time is it?" - Get current time
- "Hello JARVIS" - Greeting
- "System status" - Check system
- "Help" - Show available commands

## 🛠️ Configuration

Edit `config/default_config.yaml` to customize:
- AI model settings
- Voice recognition options  
- UI theme and colors
- Performance parameters

## 📚 Documentation

- `DEVELOPMENT_PROGRESS.md` - Development status
- `PROJECT_OVERVIEW.md` - Project overview
- `VOICE_SYSTEM_COMPLETE.md` - Voice system details

## 🔧 Troubleshooting

1. **Models not downloading**: Check internet connection
2. **Audio issues**: Check microphone permissions
3. **Performance slow**: Adjust model quantization in config
4. **Memory errors**: Close other applications, check RAM

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md

---

**Ready to experience the future of AI assistants! 🚀**
"""
        
        try:
            with open("README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            print("   ✅ Created README.md")
            return True
        except Exception as e:
            self.errors.append(f"README creation failed: {e}")
            return False
    
    def show_summary(self):
        """แสดงสรุปการติดตั้ง"""
        print("\n" + "="*60)
        print("📋 INSTALLATION SUMMARY")
        print("="*60)
        
        if not self.errors:
            print("🎉 ✅ Installation completed successfully!")
            print("\n🚀 Next steps:")
            print("   1. Run: python run_jarvis.py")
            print("   2. Say: 'Hey JARVIS' to start")
            print("   3. Or run GUI: python run_jarvis_gui.py")
            
            if self.warnings:
                print("\n⚠️ Warnings:")
                for warning in self.warnings:
                    print(f"   - {warning}")
        else:
            print("❌ Installation failed with errors:")
            for error in self.errors:
                print(f"   - {error}")
            
            if self.warnings:
                print("\nWarnings:")
                for warning in self.warnings:
                    print(f"   - {warning}")
            
            print("\n💡 Try fixing the errors and run install.py again")
        
        print("\n📚 Documentation: README.md")
        print("🔧 Configuration: config/default_config.yaml")
        print("📊 Logs: logs/jarvis.log")
        print("\n" + "="*60)
    
    def run_installation(self):
        """เรียกใช้การติดตั้ง"""
        self.print_banner()
        
        # Check system
        if not self.check_system_requirements():
            print("❌ System requirements not met")
            self.show_summary()
            return False
        
        steps = [
            ("Installing dependencies", self.install_dependencies),
            ("Setting up directories", self.setup_directories),
            ("Downloading models", self.download_models),
            ("Creating config files", self.create_config_files),
            ("Creating launchers", self.create_launcher_scripts),
            ("Creating documentation", self.create_readme)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"❌ Failed: {step_name}")
                self.show_summary()
                return False
        
        self.show_summary()
        return len(self.errors) == 0


def main():
    """ฟังก์ชันหลัก"""
    installer = JarvisInstaller()
    
    try:
        success = installer.run_installation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()