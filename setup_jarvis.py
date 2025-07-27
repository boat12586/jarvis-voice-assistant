#!/usr/bin/env python3
"""
🤖 JARVIS Voice Assistant - Setup Script
แสกรปรต์ติดตั้งและตั้งค่า JARVIS Voice Assistant

Version: 2.0.0 (2025 Edition)
Author: JARVIS Development Team
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JarvisSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / "venv"
        self.models_path = self.project_root / "models"
        self.config_path = self.project_root / "config"
        
        # System requirements
        self.min_python_version = (3, 9)
        self.recommended_python_version = (3, 10)
        
        logger.info("🤖 JARVIS Voice Assistant Setup")
        logger.info(f"📁 Project path: {self.project_root}")
    
    def check_system_requirements(self):
        """ตรวจสอบความต้องการของระบบ"""
        logger.info("🔍 ตรวจสอบความต้องการของระบบ...")
        
        # Check Python version
        current_version = sys.version_info[:2]
        if current_version < self.min_python_version:
            raise RuntimeError(
                f"Python {'.'.join(map(str, self.min_python_version))}+ required. "
                f"Current: {'.'.join(map(str, current_version))}"
            )
        
        if current_version < self.recommended_python_version:
            logger.warning(
                f"⚠️  Python {'.'.join(map(str, self.recommended_python_version))} recommended. "
                f"Current: {'.'.join(map(str, current_version))}"
            )
        
        # Check OS
        os_name = platform.system()
        logger.info(f"💻 Operating System: {os_name}")
        
        if os_name == "Windows":
            logger.info("🪟 Windows detected - WSL2 recommended for better performance")
        elif os_name == "Darwin":
            logger.info("🍎 macOS detected")
        elif os_name == "Linux":
            logger.info("🐧 Linux detected")
        
        # Check available disk space
        free_space = self._get_free_space()
        required_space = 15 * 1024 * 1024 * 1024  # 15 GB
        
        if free_space < required_space:
            logger.warning(
                f"⚠️  Low disk space: {free_space / (1024**3):.1f}GB available, "
                f"{required_space / (1024**3):.0f}GB recommended"
            )
        
        logger.info("✅ System requirements check passed")
    
    def _get_free_space(self):
        """คำนวณพื้นที่ว่างในดิสก์"""
        statvfs = os.statvfs(self.project_root)
        return statvfs.f_frsize * statvfs.f_bavail
    
    def setup_virtual_environment(self):
        """สร้างและตั้งค่า virtual environment"""
        logger.info("🐍 กำลังสร้าง virtual environment...")
        
        if self.venv_path.exists():
            logger.info("♻️  Virtual environment มีอยู่แล้ว - กำลังลบและสร้างใหม่")
            import shutil
            shutil.rmtree(self.venv_path)
        
        # Create new virtual environment
        subprocess.run([
            sys.executable, "-m", "venv", str(self.venv_path)
        ], check=True)
        
        # Get python and pip paths
        if platform.system() == "Windows":
            python_path = self.venv_path / "Scripts" / "python.exe"
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
            pip_path = self.venv_path / "bin" / "pip"
        
        # Upgrade pip
        logger.info("📦 กำลังอัปเกรด pip...")
        subprocess.run([
            str(python_path), "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        logger.info("✅ Virtual environment สร้างเสร็จแล้ว")
        return python_path, pip_path
    
    def install_dependencies(self, python_path, pip_path):
        """ติดตั้ง dependencies"""
        logger.info("📦 กำลังติดตั้ง dependencies...")
        
        requirements_files = [
            "requirements.txt",
            "requirements_minimal.txt"
        ]
        
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                logger.info(f"📋 ติดตั้งจาก {req_file}...")
                try:
                    subprocess.run([
                        str(pip_path), "install", "-r", str(req_path)
                    ], check=True, timeout=1800)  # 30 minutes timeout
                    logger.info(f"✅ ติดตั้งจาก {req_file} เสร็จแล้ว")
                except subprocess.TimeoutExpired:
                    logger.error(f"⏰ Timeout ขณะติดตั้ง {req_file}")
                    raise
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ ไม่สามารถติดตั้งจาก {req_file}: {e}")
                    raise
        
        logger.info("✅ Dependencies ติดตั้งเสร็จทั้งหมด")
    
    def setup_directories(self):
        """สร้าง directories ที่จำเป็น"""
        logger.info("📁 กำลังสร้าง directories...")
        
        directories = [
            "models",
            "logs",
            "data/conversation_memory",
            "data/knowledge_base",
            "data/vectordb",
            "config",
            ".claudedocs/reports",
            ".claudedocs/context"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📂 สร้าง {dir_name}")
        
        logger.info("✅ Directories สร้างเสร็จแล้ว")
    
    def create_config_files(self):
        """สร้างไฟล์ config เริ่มต้น"""
        logger.info("⚙️  กำลังสร้าง config files...")
        
        # Main config
        config = {
            "jarvis": {
                "name": "JARVIS",
                "version": "2.0.0",
                "language": "th",
                "personality": "helpful_assistant"
            },
            "models": {
                "llm_model": "deepseek-ai/deepseek-r1-distill-llama-8b",
                "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
                "whisper_model": "large-v3",
                "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2"
            },
            "voice": {
                "wake_word": "hey jarvis",
                "language": "th",
                "voice_speed": 1.0,
                "voice_volume": 0.8
            },
            "performance": {
                "use_gpu": True,
                "max_memory": "auto",
                "batch_size": 1,
                "max_tokens": 2048
            }
        }
        
        config_file = self.config_path / "default_config.yaml"
        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Startup script
        startup_script = self.project_root / "start_jarvis.sh"
        with open(startup_script, 'w') as f:
            f.write(f"""#!/bin/bash
# JARVIS Voice Assistant Startup Script

echo "🤖 Starting JARVIS Voice Assistant..."
cd "{self.project_root}"
source venv/bin/activate
python run.py

echo "👋 JARVIS shutdown complete"
""")
        
        # Make startup script executable
        if platform.system() != "Windows":
            os.chmod(startup_script, 0o755)
        
        logger.info("✅ Config files สร้างเสร็จแล้ว")
    
    def verify_installation(self, python_path):
        """ตรวจสอบการติดตั้ง"""
        logger.info("🔍 ตรวจสอบการติดตั้ง...")
        
        test_imports = [
            "torch",
            "transformers",
            "faster_whisper",
            "PyQt6",
            "numpy",
            "scipy"
        ]
        
        for module in test_imports:
            try:
                result = subprocess.run([
                    str(python_path), "-c", f"import {module}; print(f'{module}: OK')"
                ], capture_output=True, text=True, check=True)
                logger.info(f"✅ {result.stdout.strip()}")
            except subprocess.CalledProcessError:
                logger.error(f"❌ {module}: Failed to import")
                return False
        
        logger.info("✅ การติดตั้งสำเร็จทั้งหมด!")
        return True
    
    def run_setup(self):
        """เรียกใช้ setup ทั้งหมด"""
        try:
            logger.info("🚀 เริ่มการติดตั้ง JARVIS Voice Assistant...")
            
            self.check_system_requirements()
            python_path, pip_path = self.setup_virtual_environment()
            self.setup_directories()
            self.install_dependencies(python_path, pip_path)
            self.create_config_files()
            
            if self.verify_installation(python_path):
                logger.info("🎉 การติดตั้ง JARVIS เสร็จสมบูรณ์!")
                logger.info("🚀 เริ่มใช้งาน: ./start_jarvis.sh หรือ python run.py")
                return True
            else:
                logger.error("❌ การติดตั้งไม่สำเร็จ")
                return False
                
        except Exception as e:
            logger.error(f"💥 เกิดข้อผิดพลาดขณะติดตั้ง: {e}")
            return False

if __name__ == "__main__":
    setup = JarvisSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)