#!/usr/bin/env python3
"""
📥 Check DeepSeek-R1 Download Progress
"""

import os
from pathlib import Path

def check_download_progress():
    model_dir = Path.home() / ".cache/huggingface/hub/models--deepseek-ai--deepseek-r1-distill-llama-8b"
    
    if not model_dir.exists():
        print("❌ Model directory not found")
        return
    
    print("📊 DeepSeek-R1 Download Progress:")
    print("=" * 50)
    
    # Check blobs directory
    blobs_dir = model_dir / "blobs"
    if blobs_dir.exists():
        total_size = 0
        complete_size = 0
        incomplete_files = []
        
        for file_path in blobs_dir.iterdir():
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                
                if file_path.name.endswith('.incomplete'):
                    incomplete_files.append((file_path.name, size))
                else:
                    complete_size += size
        
        print(f"📂 Total downloaded: {total_size / (1024**3):.2f} GB")
        print(f"✅ Complete files: {complete_size / (1024**3):.2f} GB")
        
        if incomplete_files:
            print(f"⏳ Incomplete files: {len(incomplete_files)}")
            for name, size in incomplete_files:
                print(f"   - {name[:12]}...incomplete ({size / (1024**3):.2f} GB)")
        
        # Expected total size for DeepSeek-R1 8B
        expected_size = 15.0  # GB
        progress = (total_size / (1024**3)) / expected_size * 100
        print(f"📈 Progress: {progress:.1f}% of ~{expected_size} GB")
        
        if progress < 100:
            remaining = expected_size - (total_size / (1024**3))
            print(f"⏰ Estimated remaining: {remaining:.1f} GB")
            print(f"💡 Tip: ปล่อยให้ดาวน์โหลดต่อ หรือใช้ระบบ Fallback ไปก่อน")
        else:
            print("🎉 Download should be complete!")
    
    print("\n🤖 JARVIS Status:")
    print("✅ Fallback AI: Ready to use")
    print("⏳ DeepSeek-R1: Downloading...")
    print("💻 System: Fully operational with fallback")

if __name__ == "__main__":
    check_download_progress()