#!/usr/bin/env python3
"""
System Check for Jarvis Voice Assistant
Checks system requirements and dependencies
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check:")
    version = sys.version_info
    print(f"   Current: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version OK")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_os():
    """Check operating system"""
    print("\n💻 Operating System Check:")
    system = platform.system()
    machine = platform.machine()
    print(f"   System: {system}")
    print(f"   Architecture: {machine}")
    print(f"   Platform: {platform.platform()}")
    
    if system in ["Linux", "Windows", "Darwin"]:
        print("   ✅ Operating system supported")
        return True
    else:
        print("   ⚠️ Operating system may not be fully supported")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🎮 GPU Check:")
    
    try:
        # Try to detect NVIDIA GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ NVIDIA GPU detected")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                    print(f"   GPU: {line.strip()}")
                    break
            return True
        else:
            print("   ⚠️ NVIDIA GPU not detected")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ⚠️ nvidia-smi not found")
    
    try:
        # Check for other GPU types
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
        if 'VGA' in result.stdout:
            print("   ℹ️ VGA device detected")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("   ⚠️ GPU acceleration may not be available")
    return False

def check_memory():
    """Check system memory"""
    print("\n💾 Memory Check:")
    
    try:
        # Check available memory
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / 1024 / 1024
                        print(f"   Total RAM: {mem_gb:.1f} GB")
                        
                        if mem_gb >= 8:
                            print("   ✅ Sufficient RAM")
                            return True
                        else:
                            print("   ⚠️ 8GB+ RAM recommended")
                            return False
        else:
            print("   ℹ️ Memory check not available on this platform")
            return True
            
    except Exception as e:
        print(f"   ⚠️ Could not check memory: {e}")
        return False

def check_audio():
    """Check audio system"""
    print("\n🔊 Audio System Check:")
    
    try:
        # Check if pulseaudio is running (Linux)
        if platform.system() == "Linux":
            result = subprocess.run(['pulseaudio', '--check'], capture_output=True)
            if result.returncode == 0:
                print("   ✅ PulseAudio running")
            else:
                print("   ⚠️ PulseAudio not running")
        
        # Check for audio devices
        try:
            import subprocess
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'card' in result.stdout:
                print("   ✅ Audio output devices found")
                return True
            else:
                print("   ⚠️ No audio output devices found")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ℹ️ Could not list audio devices")
            
    except Exception as e:
        print(f"   ℹ️ Audio check: {e}")
    
    return True

def check_dependencies():
    """Check Python dependencies"""
    print("\n📦 Dependencies Check:")
    
    required_packages = [
        'PyQt6',
        'yaml',
        'numpy',
        'requests'
    ]
    
    all_ok = True
    
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
                print(f"   ✅ {package}")
            elif package == 'PyQt6':
                import PyQt6
                print(f"   ✅ {package}")
            elif package == 'numpy':
                import numpy
                print(f"   ✅ {package}")
            elif package == 'requests':
                import requests
                print(f"   ✅ {package}")
            else:
                __import__(package)
                print(f"   ✅ {package}")
                
        except ImportError:
            print(f"   ❌ {package} not installed")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check project directories"""
    print("\n📁 Project Structure Check:")
    
    required_dirs = [
        'src',
        'config',
        'data',
        'logs'
    ]
    
    all_ok = True
    base_path = Path(__file__).parent
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/ directory exists")
        else:
            print(f"   ❌ {dir_name}/ directory missing")
            all_ok = False
            
            # Create missing directories
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"   ✅ Created {dir_name}/ directory")
            except Exception as e:
                print(f"   ❌ Could not create {dir_name}/: {e}")
    
    return all_ok

def check_config_files():
    """Check configuration files"""
    print("\n⚙️ Configuration Check:")
    
    config_files = [
        'config/default_config.yaml',
        'data/knowledge_base.json'
    ]
    
    all_ok = True
    base_path = Path(__file__).parent
    
    for config_file in config_files:
        file_path = base_path / config_file
        if file_path.exists():
            print(f"   ✅ {config_file}")
        else:
            print(f"   ❌ {config_file} missing")
            all_ok = False
    
    return all_ok

def main():
    """Main system check"""
    print("🤖 Jarvis Voice Assistant - System Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Operating System", check_os),
        ("GPU", check_gpu),
        ("Memory", check_memory),
        ("Audio", check_audio),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Config Files", check_config_files)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ Error during {check_name} check: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SYSTEM CHECK SUMMARY:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 System ready for Jarvis Voice Assistant!")
        print("\nNext steps:")
        print("1. Run: python test_ui.py (test UI)")
        print("2. Install AI models (if needed)")
        print("3. Run: python run.py (full application)")
    elif passed >= total * 0.7:
        print("\n⚠️ System mostly ready with some warnings")
        print("You can proceed but may encounter limitations")
    else:
        print("\n❌ System not ready - please address the failed checks")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)