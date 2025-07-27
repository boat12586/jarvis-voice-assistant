#!/usr/bin/env python3
"""
JARVIS Text Interface - รองรับภาษาไทยเต็มรูปแบบ
Command-line interface that works without GUI dependencies
"""

import sys
import time
import os
from datetime import datetime
from pathlib import Path

class SimpleJarvis:
    """Simple JARVIS for command-line interaction"""
    
    def __init__(self):
        self.session_turns = []
        self.is_running = True
        
    def display_welcome(self):
        """Display welcome message"""
        print("=" * 60)
        print("🤖 JARVIS - ผู้ช่วยอัจฉริยะ")
        print("🤖 JARVIS - Intelligent Assistant")
        print("=" * 60)
        print()
        print("🎉 ยินดีต้อนรับสู่ JARVIS Text Interface!")
        print("🎉 Welcome to JARVIS Text Interface!")
        print()
        print("✨ ความสามารถ / Capabilities:")
        print("• ตอบคำถามเกี่ยวกับเทคโนโลยีและ AI")
        print("• Answer questions about technology and AI")
        print("• อธิบายแนวคิดซับซ้อนให้เข้าใจง่าย")
        print("• Explain complex concepts simply")
        print("• สนทนาภาษาไทยและอังกฤษ")
        print("• Chat in Thai and English")
        print()
        print("📝 วิธีใช้งาน / How to use:")
        print("• พิมพ์คำถามหรือคำสั่ง / Type your question or command")
        print("• พิมพ์ 'quit' หรือ 'exit' เพื่อออก / Type 'quit' or 'exit' to leave")
        print("• พิมพ์ 'help' สำหรับความช่วยเหลือ / Type 'help' for assistance")
        print()
        print("🗣️ ตัวอย่าง / Examples:")
        print("• สวัสดี JARVIS")
        print("• AI คืออะไร")
        print("• Explain machine learning")
        print("• ช่วยอธิบาย Python")
        print()
        print("-" * 60)
    
    def process_command(self, text: str) -> str:
        """Process user command"""
        if not text.strip():
            return "กรุณาพิมพ์ข้อความครับ / Please type a message"
        
        text_lower = text.lower().strip()
        
        # Exit commands
        if text_lower in ['quit', 'exit', 'ออก', 'จบ']:
            self.is_running = False
            return "ขอบคุณที่ใช้งาน JARVIS ครับ! / Thank you for using JARVIS!"
        
        # Help command
        if text_lower in ['help', 'ช่วยเหลือ', 'คำสั่ง']:
            return self.get_help()
        
        # Clear command
        if text_lower in ['clear', 'cls', 'ล้าง']:
            os.system('clear' if os.name == 'posix' else 'cls')
            self.display_welcome()
            return ""
        
        # Detect language
        is_thai = any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text)
        
        # Process based on content
        response = self.generate_response(text, is_thai)
        
        # Add to session
        self.session_turns.append({
            'user': text,
            'assistant': response,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'language': 'thai' if is_thai else 'english'
        })
        
        return response
    
    def generate_response(self, text: str, is_thai: bool) -> str:
        """Generate response based on input"""
        text_lower = text.lower()
        
        # Greetings
        if any(word in text_lower for word in ["สวัสดี", "hello", "hi", "ทักทาย", "หวัดดี"]):
            if is_thai:
                return """🤖 สวัสดีครับ! ผม JARVIS ผู้ช่วยอัจฉริยะของคุณ

✨ ผมสามารถช่วยคุณได้ในเรื่อง:
🔹 ตอบคำถามเกี่ยวกับเทคโนโลยี AI และวิทยาศาสตร์
🔹 อธิบายแนวคิดซับซ้อนให้เข้าใจง่าย
🔹 แนะนำการเรียนรู้ภาษาโปรแกรม
🔹 สนทนาและให้คำปรึกษา

มีอะไรให้ช่วยไหมครับ? 😊"""
            else:
                return """🤖 Hello! I'm JARVIS, your intelligent assistant

✨ I can help you with:
🔹 Answering questions about AI technology and science
🔹 Explaining complex concepts simply
🔹 Programming language guidance
🔹 General conversation and advice

How can I help you today? 😊"""
        
        # AI/Technology questions
        elif any(word in text_lower for word in ["ai", "ปัญญาประดิษฐ์", "artificial intelligence"]):
            if is_thai:
                return """🧠 ปัญญาประดิษฐ์ (Artificial Intelligence - AI)

ปัญญาประดิษฐ์คือเทคโนโลยีที่ทำให้คอมพิวเตอร์สามารถ:

📚 ความสามารถหลัก:
🔸 เรียนรู้จากข้อมูล (Machine Learning)
🔸 เข้าใจภาษามนุษย์ (Natural Language Processing)
🔸 รับรู้รูปแบบและภาพ (Computer Vision)
🔸 ตัดสินใจและแก้ปัญหา (Decision Making)
🔸 เลียนแบบการคิดของมนุษย์ (Cognitive Computing)

🚀 การใช้งานในปัจจุบัน:
🔸 ผู้ช่วยเสียงอัจฉริยะ (เช่น ผมเอง!)
🔸 รถยนต์ขับขี่อัตโนมัติ
🔸 ระบบแนะนำสินค้าและเนื้อหา
🔸 การแปลภาษาอัตโนมัติ
🔸 การวินิจฉัยทางการแพทย์
🔸 ระบบรักษาความปลอดภัย

คุณสนใจด้านไหนเป็นพิเศษครับ?"""
            else:
                return """🧠 Artificial Intelligence (AI)

AI is technology that enables computers to:

📚 Core Capabilities:
🔸 Learn from data (Machine Learning)
🔸 Understand human language (NLP)
🔸 Recognize patterns and images (Computer Vision)
🔸 Make decisions and solve problems
🔸 Mimic human thinking (Cognitive Computing)

🚀 Current Applications:
🔸 Intelligent voice assistants (like me!)
🔸 Autonomous vehicles
🔸 Recommendation systems
🔸 Automatic language translation
🔸 Medical diagnosis
🔸 Security systems

What specific aspect interests you most?"""
        
        # Machine Learning
        elif any(word in text_lower for word in ["machine learning", "แมชชีนเลิร์นนิง", "การเรียนรู้ของเครื่อง"]):
            if is_thai:
                return """🎯 Machine Learning (การเรียนรู้ของเครื่อง)

เป็นสาขาหนึ่งของ AI ที่เครื่องจักรเรียนรู้จากข้อมูลโดยอัตโนมัติ

📊 ประเภทหลัก:
🔹 Supervised Learning (การเรียนรู้แบบมีผู้สอน)
   • เรียนรู้จากตัวอย่างที่มีคำตอบแล้ว
   • เช่น: การจำแนกภาพ, การทำนายราคา

🔹 Unsupervised Learning (การเรียนรู้แบบไม่มีผู้สอน)
   • หาแพทเทิร์นและกลุ่มจากข้อมูลเอง
   • เช่น: การจัดกลุ่มลูกค้า, การลดมิติข้อมูล

🔹 Reinforcement Learning (การเรียนรู้แบบเสริมแรง)
   • เรียนรู้จากการลองผิดลองถูก
   • เช่น: AI เล่นเกม, หุ่นยนต์เดิน

💡 ตัวอย่างการใช้งาน:
🔸 Netflix แนะนำหนังที่คุณอาจชอบ
🔸 Google แปลภาษาได้อัตโนมัติ
🔸 ธนาคารตรวจจับการทำธุรกรรมผิดปกติ
🔸 โรงพยาบาลวินิจฉัยโรคจากภาพ X-ray

อยากรู้เรื่องไหนเพิ่มเติมครับ?"""
            else:
                return """🎯 Machine Learning

A subset of AI where machines automatically learn from data

📊 Main Types:
🔹 Supervised Learning
   • Learn from labeled examples
   • E.g., image classification, price prediction

🔹 Unsupervised Learning  
   • Find patterns in data without labels
   • E.g., customer segmentation, data compression

🔹 Reinforcement Learning
   • Learn through trial and error
   • E.g., AI playing games, robot walking

💡 Real-world Examples:
🔸 Netflix recommends movies you might like
🔸 Google Translate works automatically
🔸 Banks detect fraudulent transactions
🔸 Hospitals diagnose diseases from X-rays

What would you like to know more about?"""
        
        # Programming questions
        elif any(word in text_lower for word in ["python", "programming", "โปรแกรม", "เขียนโปรแกรม"]):
            if is_thai:
                return """🐍 Python Programming

Python เป็นภาษาโปรแกรมที่ได้รับความนิยมสูงสุดในโลก!

⭐ จุดเด่นของ Python:
🔸 ไวยากรณ์ง่าย อ่านเข้าใจได้เหมือนภาษาอังกฤษ
🔸 มี Library เยอะมาก ทำอะไรก็ได้
🔸 เหมาะกับผู้เริ่มต้นและผู้เชี่ยวชาญ
🔸 ใช้ได้ทั้ง Web, AI, Data Science, Automation

🚀 การใช้งานยอดนิยม:
🔸 Data Science และ Machine Learning
🔸 Web Development (Django, Flask)
🔸 Automation และ Scripting
🔸 การพัฒนา AI และ Deep Learning

📚 เริ่มต้นเรียน Python:
1. เรียนพื้นฐาน: ตัวแปร, ลูป, ฟังก์ชัน
2. ฝึกทำโปรเจค: เกมง่ายๆ, คำนวณ
3. เรียน Library: pandas, numpy, matplotlib
4. ท้าทายตัวเอง: สร้าง AI หรือ Web App

อยากเริ่มต้นตรงไหนครับ?"""
            else:
                return """🐍 Python Programming

Python is the world's most popular programming language!

⭐ Python's Strengths:
🔸 Simple syntax, reads like English
🔸 Massive library ecosystem
🔸 Great for beginners and experts
🔸 Versatile: Web, AI, Data Science, Automation

🚀 Popular Uses:
🔸 Data Science and Machine Learning
🔸 Web Development (Django, Flask)
🔸 Automation and Scripting
🔸 AI and Deep Learning Development

📚 Learning Path:
1. Master basics: variables, loops, functions
2. Build projects: simple games, calculators
3. Learn libraries: pandas, numpy, matplotlib
4. Create challenges: AI or Web Apps

Where would you like to start?"""
        
        # Help requests
        elif any(word in text_lower for word in ["ช่วย", "help", "สอน", "teach", "แนะนำ"]):
            return self.get_help()
        
        # General information requests
        elif any(word in text_lower for word in ["คือ", "อะไร", "what", "explain", "อธิบาย", "หมายถึง"]):
            if is_thai:
                return f"""🤔 เรื่อง "{text}" เป็นหัวข้อที่น่าสนใจมากครับ!

💡 ผมสามารถช่วยอธิบายได้ในหลายมุมมอง:
🔸 ความหมายและนิยาม
🔸 หลักการทำงาน
🔸 ตัวอย่างการใช้งาน
🔸 ข้อดีและข้อจำกัด
🔸 แนวโน้มในอนาคต

คุณต้องการให้ผมอธิบายในมุมมองไหนเป็นพิเศษครับ?
หรือถามเฉพาะเจาะจงมากขึ้น เช่น "AI ทำงานยังไง" หรือ "Python ใช้ทำอะไรได้บ้าง" 🎯"""
            else:
                return f"""🤔 "{text}" is a very interesting topic!

💡 I can help explain it from multiple perspectives:
🔸 Definition and meaning
🔸 How it works
🔸 Real-world applications
🔸 Advantages and limitations
🔸 Future trends

What specific aspect would you like me to focus on?
Or ask more specifically like "How does AI work?" or "What can Python do?" 🎯"""
        
        # General conversation
        else:
            if is_thai:
                return f"""ผมได้รับข้อความ: "{text}" แล้วครับ

🎯 การวิเคราะห์:
🔸 ภาษา: ไทย
🔸 ความยาว: {len(text)} ตัวอักษร
🔸 เวลา: {datetime.now().strftime("%H:%M:%S")}

💭 ผมพร้อมช่วยเหลือในเรื่อง:
🔸 คำถามเกี่ยวกับเทคโนโลยี AI
🔸 การเขียนโปรแกรม
🔸 วิทยาศาสตร์คอมพิวเตอร์
🔸 การเรียนรู้เทคโนโลยีใหม่ๆ

ลองถามคำถามที่เฉพาะเจาะจงมากกว่านี้ได้ครับ! 🤖"""
            else:
                return f"""I received your message: "{text}"

🎯 Analysis:
🔸 Language: English
🔸 Length: {len(text)} characters
🔸 Time: {datetime.now().strftime("%H:%M:%S")}

💭 I'm ready to help with:
🔸 AI and technology questions
🔸 Programming guidance
🔸 Computer science topics
🔸 Learning new technologies

Feel free to ask more specific questions! 🤖"""
    
    def get_help(self) -> str:
        """Get help information"""
        return """📚 JARVIS Help / คำสั่งที่ใช้ได้

🎯 คำสั่งพื้นฐาน / Basic Commands:
🔸 help - แสดงความช่วยเหลือ / Show this help
🔸 clear - ล้างหน้าจอ / Clear screen
🔸 quit/exit - ออกจากโปรแกรม / Exit program

🗣️ ตัวอย่างคำถาม / Example Questions:
🔸 "AI คืออะไร" / "What is AI"
🔸 "อธิบาย Machine Learning" / "Explain Machine Learning"
🔸 "Python ใช้ทำอะไรได้บ้าง" / "What can Python do"
🔸 "สอนการเขียนโปรแกรม" / "Teach me programming"

💡 เคล็ดลับ / Tips:
🔸 ถามคำถามที่เฉพาะเจาะจง จะได้คำตอบที่ดีกว่า
🔸 Ask specific questions for better answers
🔸 ใช้ภาษาไทยหรืออังกฤษก็ได้ / Use Thai or English
🔸 ผมจำบริบทการสนทนาได้ / I remember conversation context

🤖 ผมพร้อมช่วยเหลือตลอด 24/7!"""
    
    def run(self):
        """Main interaction loop"""
        self.display_welcome()
        
        while self.is_running:
            try:
                # Get user input
                user_input = input("\n💬 คุณ/You: ").strip()
                
                if not user_input:
                    continue
                
                # Process command
                print("\n🤖 JARVIS:", end=" ")
                response = self.process_command(user_input)
                
                if response:
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n\n👋 ขอบคุณที่ใช้งาน JARVIS!")
                print("👋 Thank you for using JARVIS!")
                break
            except EOFError:
                print("\n\n👋 ลาก่อนครับ!")
                print("👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ เกิดข้อผิดพลาด: {e}")
                print(f"❌ Error occurred: {e}")

def main():
    """Main function"""
    # Set UTF-8 encoding for proper Thai display
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    # Clear screen at start
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Create and run JARVIS
    jarvis = SimpleJarvis()
    jarvis.run()

if __name__ == "__main__":
    main()