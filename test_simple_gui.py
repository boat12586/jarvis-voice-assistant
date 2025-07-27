#!/usr/bin/env python3
"""
Simple GUI Test for JARVIS - Test core functionality without complex dependencies
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QFont, QPalette, QColor
    
    class SimpleJarvisGUI(QMainWindow):
        """Simple JARVIS GUI for testing"""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("JARVIS Voice Assistant - Test Mode")
            self.setGeometry(100, 100, 800, 600)
            
            # Set dark theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #0a0a0a;
                    color: #00ffff;
                }
                QLabel {
                    color: #00ffff;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #1a1a2e;
                    color: #00ffff;
                    border: 2px solid #00ffff;
                    border-radius: 10px;
                    padding: 10px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #16213e;
                    box-shadow: 0 0 20px #00ffff;
                }
                QPushButton:pressed {
                    background-color: #0f3460;
                }
                QTextEdit {
                    background-color: #1a1a2e;
                    color: #00ffff;
                    border: 1px solid #00ffff;
                    border-radius: 5px;
                    padding: 5px;
                }
            """)
            
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Title
            title = QLabel("🤖 J.A.R.V.I.S Voice Assistant")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Status
            self.status_label = QLabel("System Ready")
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.status_label.setFont(QFont("Arial", 14))
            layout.addWidget(self.status_label)
            
            # Test buttons
            self.test_ai_btn = QPushButton("Test AI Engine")
            self.test_ai_btn.clicked.connect(self.test_ai_engine)
            layout.addWidget(self.test_ai_btn)
            
            self.test_rag_btn = QPushButton("Test RAG System")
            self.test_rag_btn.clicked.connect(self.test_rag_system)
            layout.addWidget(self.test_rag_btn)
            
            self.test_embedding_btn = QPushButton("Test Embedding Model")
            self.test_embedding_btn.clicked.connect(self.test_embedding_model)
            layout.addWidget(self.test_embedding_btn)
            
            # Output area
            self.output_area = QTextEdit()
            self.output_area.setMaximumHeight(200)
            layout.addWidget(self.output_area)
            
            # Status updates
            self.log_message("🚀 JARVIS GUI Test Mode Initialized")
            
            # Auto-test timer
            QTimer.singleShot(1000, self.run_auto_tests)
        
        def log_message(self, message):
            """Add message to output area"""
            self.output_area.append(message)
            self.output_area.verticalScrollBar().setValue(
                self.output_area.verticalScrollBar().maximum()
            )
        
        def update_status(self, status):
            """Update status label"""
            self.status_label.setText(status)
        
        def test_ai_engine(self):
            """Test AI engine components"""
            try:
                self.update_status("Testing AI Engine...")
                self.log_message("🧠 Testing AI Engine Components...")
                
                # Test transformer imports
                from transformers import AutoTokenizer
                self.log_message("✅ Transformers library working")
                
                # Test DeepSeek tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/deepseek-r1-distill-llama-8b",
                    trust_remote_code=True
                )
                test_text = "Hello JARVIS"
                tokens = tokenizer.encode(test_text)
                self.log_message(f"✅ DeepSeek tokenizer: {len(tokens)} tokens")
                
                self.update_status("AI Engine: Ready")
                self.log_message("🎉 AI Engine test completed!")
                
            except Exception as e:
                self.log_message(f"❌ AI Engine error: {e}")
                self.update_status("AI Engine: Error")
        
        def test_rag_system(self):
            """Test RAG system"""
            try:
                self.update_status("Testing RAG System...")
                self.log_message("📚 Testing RAG System...")
                
                # Test FAISS
                import faiss
                index = faiss.IndexFlatIP(1024)
                self.log_message("✅ FAISS working")
                
                # Test sentence transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
                self.log_message("✅ Sentence transformers working")
                
                # Test embedding
                test_embedding = model.encode("Test document")
                self.log_message(f"✅ Embedding shape: {test_embedding.shape}")
                
                self.update_status("RAG System: Ready")
                self.log_message("🎉 RAG System test completed!")
                
            except Exception as e:
                self.log_message(f"❌ RAG System error: {e}")
                self.update_status("RAG System: Error")
        
        def test_embedding_model(self):
            """Test embedding model specifically"""
            try:
                self.update_status("Testing Embedding Model...")
                self.log_message("🔍 Testing mxbai-embed-large...")
                
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
                
                # Test multiple texts
                test_texts = [
                    "Hello, how are you today?",
                    "สวัสดีครับ วันนี้เป็นอย่างไรบ้าง",
                    "What is artificial intelligence?"
                ]
                
                embeddings = model.encode(test_texts)
                self.log_message(f"✅ Processed {len(test_texts)} texts")
                self.log_message(f"✅ Embedding dimensions: {embeddings.shape}")
                
                self.update_status("Embedding Model: Ready")
                self.log_message("🎉 Embedding model test completed!")
                
            except Exception as e:
                self.log_message(f"❌ Embedding model error: {e}")
                self.update_status("Embedding Model: Error")
        
        def run_auto_tests(self):
            """Run all tests automatically"""
            self.log_message("🔄 Starting automated tests...")
            
            # Schedule tests with delays
            QTimer.singleShot(500, self.test_embedding_model)
            QTimer.singleShot(2000, self.test_rag_system)
            QTimer.singleShot(4000, self.test_ai_engine)
            QTimer.singleShot(6000, self.tests_completed)
        
        def tests_completed(self):
            """All tests completed"""
            self.log_message("✨ All tests completed!")
            self.update_status("All Systems: Ready for Voice Testing")
    
    def main():
        """Main entry point"""
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("JARVIS Test GUI")
        app.setApplicationVersion("0.1.0")
        
        # Create and show window
        window = SimpleJarvisGUI()
        window.show()
        
        # Run for limited time in test mode
        QTimer.singleShot(10000, app.quit)  # Auto-quit after 10 seconds
        
        return app.exec()
    
    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    print(f"GUI test failed - missing PyQt6: {e}")
    print("Testing core functionality without GUI...")
    
    # Test core functionality without GUI
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    print("🧠 Testing AI Components...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-r1-distill-llama-8b",
            trust_remote_code=True
        )
        print("✅ AI Engine components working")
    except Exception as ai_error:
        print(f"❌ AI Engine error: {ai_error}")
    
    print("📚 Testing RAG Components...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        test_embedding = model.encode("Test")
        print(f"✅ RAG components working: {test_embedding.shape}")
    except Exception as rag_error:
        print(f"❌ RAG error: {rag_error}")
    
    print("🎉 Core functionality test completed!")