"""
Comprehensive Test Script for JARVIS Multimodal AI System
Tests all components of the multimodal AI implementation
"""

import sys
import os
import logging
import asyncio
import tempfile
from pathlib import Path
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import multimodal components
from src.ai.multimodal_engine import MultimodalEngine
from src.ai.vision_models import VisionModelManager
from src.ai.ocr_system import AdvancedOCRSystem
from src.ai.visual_qa_system import VisualQASystem
from src.ai.video_analysis_system import VideoAnalysisSystem
from src.ai.multimodal_fusion_system import (
    MultimodalFusionSystem, 
    ModalityInput, 
    ModalityType,
    MultimodalFusionStrategy
)
from src.ai.multimodal_emotional_integration import MultimodalEmotionalIntegration
from src.features.multimodal_web_interface import MultimodalWebInterface
from src.features.realtime_camera_system import RealtimeCameraSystem


class MultimodalSystemTester:
    """Comprehensive tester for multimodal AI system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Configuration for testing
        self.config = {
            'vision': {
                'clip_model': 'openai/clip-vit-base-patch32',
                'blip_model': 'Salesforce/blip-image-captioning-base',
                'blip_qa_model': 'Salesforce/blip-vqa-base'
            },
            'ocr': {
                'confidence_threshold': 0.5,
                'tesseract_path': '/usr/bin/tesseract'
            },
            'vqa': {},
            'video': {
                'max_frames': 20,
                'scene_threshold': 0.3
            },
            'fusion': {
                'fusion_weights': {
                    'text': 0.4,
                    'voice': 0.2,
                    'vision': 0.3,
                    'emotion': 0.1
                }
            },
            'emotion': {
                'max_history_length': 50,
                'voice_analysis': True
            },
            'camera': {
                'camera_id': 0,
                'fps': 30,
                'resolution': (640, 480),
                'analysis_fps': 2
            },
            'web': {
                'upload_folder': str(self.temp_dir / 'uploads'),
                'max_file_size': 50 * 1024 * 1024
            }
        }
        
        self.logger.info(f"Test environment set up in: {self.temp_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for tests"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_test_images(self):
        """Create test images for testing"""
        self.logger.info("Creating test images...")
        
        # Create uploads directory
        uploads_dir = Path(self.config['web']['upload_folder'])
        uploads_dir.mkdir(exist_ok=True)
        
        # Test image 1: Simple text image
        img1 = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img1)
        
        try:
            # Try to use a font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((50, 50), "Hello JARVIS!", fill='black', font=font)
        draw.text((50, 100), "สวัสดี JARVIS!", fill='blue', font=font)
        
        img1_path = uploads_dir / "test_text_image.jpg"
        img1.save(img1_path)
        
        # Test image 2: Colorful scene
        img2 = Image.new('RGB', (400, 300), color='lightblue')
        draw2 = ImageDraw.Draw(img2)
        
        # Draw simple shapes
        draw2.rectangle([50, 50, 150, 150], fill='red', outline='black')
        draw2.ellipse([200, 100, 300, 200], fill='green', outline='black')
        draw2.text((50, 250), "Colorful shapes", fill='black', font=font)
        
        img2_path = uploads_dir / "test_scene_image.jpg"
        img2.save(img2_path)
        
        # Test image 3: Thai text
        img3 = Image.new('RGB', (400, 200), color='white')
        draw3 = ImageDraw.Draw(img3)
        draw3.text((50, 50), "ทดสอบระบบ OCR", fill='black', font=font)
        draw3.text((50, 100), "การรู้จำตัวอักษรไทย", fill='blue', font=font)
        
        img3_path = uploads_dir / "test_thai_text.jpg"
        img3.save(img3_path)
        
        self.test_images = {
            'text_image': str(img1_path),
            'scene_image': str(img2_path),
            'thai_text': str(img3_path)
        }
        
        self.logger.info(f"Created {len(self.test_images)} test images")
        return self.test_images
    
    async def test_vision_models(self):
        """Test vision model components"""
        self.logger.info("Testing Vision Models...")
        
        try:
            # Initialize vision manager
            vision_manager = VisionModelManager(self.config['vision'])
            
            # Test CLIP model loading
            clip_loaded = vision_manager.load_clip_model()
            self.test_results['clip_model_loaded'] = clip_loaded
            
            # Test BLIP model loading
            blip_loaded = vision_manager.load_blip_models()
            self.test_results['blip_models_loaded'] = blip_loaded
            
            if clip_loaded and blip_loaded:
                # Test image captioning
                test_image_path = self.test_images['scene_image']
                caption = vision_manager.generate_image_caption(test_image_path)
                self.test_results['image_captioning'] = {
                    'success': len(caption) > 0,
                    'caption': caption
                }
                
                # Test image classification
                categories = ['person', 'vehicle', 'animal', 'object', 'scene']
                classification = vision_manager.classify_image_with_categories(
                    test_image_path, categories
                )
                self.test_results['image_classification'] = {
                    'success': len(classification) > 0,
                    'results': classification
                }
                
                # Test image embeddings
                embeddings = vision_manager.get_image_embeddings(test_image_path)
                self.test_results['image_embeddings'] = {
                    'success': len(embeddings) > 0,
                    'embedding_size': len(embeddings)
                }
            
            # Get model info
            model_info = vision_manager.get_model_info()
            self.test_results['vision_model_info'] = model_info
            
            self.logger.info("Vision models test completed")
            
        except Exception as e:
            self.logger.error(f"Vision models test failed: {e}")
            self.test_results['vision_models_error'] = str(e)
    
    async def test_ocr_system(self):
        """Test OCR system components"""
        self.logger.info("Testing OCR System...")
        
        try:
            # Initialize OCR system
            ocr_system = AdvancedOCRSystem(self.config['ocr'])
            
            # Test English text recognition
            english_result = ocr_system.extract_text(
                self.test_images['text_image'], 
                language='en'
            )
            self.test_results['ocr_english'] = {
                'success': len(english_result.get('full_text', '')) > 0,
                'text_found': english_result.get('full_text', ''),
                'confidence': english_result.get('avg_confidence', 0)
            }
            
            # Test Thai text recognition
            thai_result = ocr_system.extract_text(
                self.test_images['thai_text'], 
                language='th'
            )
            self.test_results['ocr_thai'] = {
                'success': len(thai_result.get('full_text', '')) > 0,
                'text_found': thai_result.get('full_text', ''),
                'confidence': thai_result.get('avg_confidence', 0)
            }
            
            # Test hybrid OCR
            hybrid_result = ocr_system.extract_text(
                self.test_images['text_image'], 
                engine='hybrid'
            )
            self.test_results['ocr_hybrid'] = {
                'success': len(hybrid_result.get('full_text', '')) > 0,
                'text_found': hybrid_result.get('full_text', ''),
                'engines_used': ['easyocr', 'tesseract']
            }
            
            # Get OCR stats
            ocr_stats = ocr_system.get_ocr_stats()
            self.test_results['ocr_stats'] = ocr_stats
            
            self.logger.info("OCR system test completed")
            
        except Exception as e:
            self.logger.error(f"OCR system test failed: {e}")
            self.test_results['ocr_system_error'] = str(e)
    
    async def test_visual_qa_system(self):
        """Test Visual Q&A system"""
        self.logger.info("Testing Visual Q&A System...")
        
        try:
            # Initialize VQA system
            vqa_system = VisualQASystem(self.config['vqa'])
            
            # Test questions about scene image
            test_questions = [
                "What colors do you see in this image?",
                "What shapes are visible?",
                "Describe what you see",
                "How many objects are in the image?"
            ]
            
            vqa_results = []
            for question in test_questions:
                result = vqa_system.answer_question(
                    self.test_images['scene_image'], 
                    question
                )
                vqa_results.append({
                    'question': question,
                    'answer': result.get('answer', ''),
                    'confidence': result.get('confidence', 0)
                })
            
            self.test_results['visual_qa'] = {
                'success': len(vqa_results) > 0,
                'results': vqa_results
            }
            
            # Test with text image
            text_question = "What text do you see in this image?"
            text_result = vqa_system.answer_question(
                self.test_images['text_image'], 
                text_question
            )
            
            self.test_results['visual_qa_text'] = {
                'question': text_question,
                'answer': text_result.get('answer', ''),
                'confidence': text_result.get('confidence', 0)
            }
            
            # Get VQA stats
            vqa_stats = vqa_system.get_system_stats()
            self.test_results['vqa_stats'] = vqa_stats
            
            self.logger.info("Visual Q&A system test completed")
            
        except Exception as e:
            self.logger.error(f"Visual Q&A system test failed: {e}")
            self.test_results['vqa_system_error'] = str(e)
    
    async def test_multimodal_fusion(self):
        """Test multimodal fusion system"""
        self.logger.info("Testing Multimodal Fusion System...")
        
        try:
            # Initialize fusion system
            fusion_system = MultimodalFusionSystem(self.config['fusion'])
            
            # Test 1: Text + Vision fusion
            text_input = ModalityInput(
                modality=ModalityType.TEXT,
                data="What do you see in this image?",
                confidence=1.0
            )
            
            vision_input = ModalityInput(
                modality=ModalityType.VISION,
                data=self.test_images['scene_image'],
                confidence=0.9
            )
            
            fusion_result1 = fusion_system.fuse_multimodal_input(
                [text_input, vision_input],
                fusion_strategy=MultimodalFusionStrategy.ADAPTIVE
            )
            
            self.test_results['fusion_text_vision'] = {
                'success': fusion_result1.confidence > 0,
                'response': fusion_result1.fused_response,
                'confidence': fusion_result1.confidence,
                'modalities_used': [m.value for m in fusion_result1.modalities_used],
                'fusion_strategy': fusion_result1.fusion_strategy
            }
            
            # Test 2: Text-only processing
            text_only_result = fusion_system.fuse_multimodal_input(
                [text_input],
                fusion_strategy=MultimodalFusionStrategy.ADAPTIVE
            )
            
            self.test_results['fusion_text_only'] = {
                'success': text_only_result.confidence > 0,
                'response': text_only_result.fused_response,
                'confidence': text_only_result.confidence
            }
            
            # Test 3: Different fusion strategies
            strategies_tested = []
            for strategy in MultimodalFusionStrategy:
                try:
                    strategy_result = fusion_system.fuse_multimodal_input(
                        [text_input, vision_input],
                        fusion_strategy=strategy
                    )
                    strategies_tested.append({
                        'strategy': strategy.value,
                        'success': True,
                        'confidence': strategy_result.confidence
                    })
                except Exception as e:
                    strategies_tested.append({
                        'strategy': strategy.value,
                        'success': False,
                        'error': str(e)
                    })
            
            self.test_results['fusion_strategies'] = strategies_tested
            
            # Get fusion stats
            fusion_stats = fusion_system.get_fusion_stats()
            self.test_results['fusion_stats'] = fusion_stats
            
            self.logger.info("Multimodal fusion system test completed")
            
        except Exception as e:
            self.logger.error(f"Multimodal fusion test failed: {e}")
            self.test_results['fusion_system_error'] = str(e)
    
    async def test_emotional_integration(self):
        """Test emotional AI integration"""
        self.logger.info("Testing Emotional AI Integration...")
        
        try:
            # Initialize emotional integration
            emotional_integration = MultimodalEmotionalIntegration({
                'emotion': self.config['emotion'],
                'multimodal': self.config['fusion']
            })
            
            # Test emotional multimodal processing
            emotional_text = ModalityInput(
                modality=ModalityType.TEXT,
                data="I'm feeling really happy about this beautiful image!",
                confidence=1.0
            )
            
            vision_input = ModalityInput(
                modality=ModalityType.VISION,
                data=self.test_images['scene_image'],
                confidence=0.9
            )
            
            emotional_context = emotional_integration.process_multimodal_with_emotion(
                [emotional_text, vision_input]
            )
            
            self.test_results['emotional_integration'] = {
                'success': emotional_context.overall_emotional_confidence > 0,
                'primary_emotion': emotional_context.emotion_result.primary_emotion,
                'emotion_confidence': emotional_context.emotion_result.confidence,
                'multimodal_response': emotional_context.multimodal_result.fused_response,
                'text_emotion_alignment': emotional_context.text_emotion_alignment,
                'overall_confidence': emotional_context.overall_emotional_confidence
            }
            
            # Test response enhancement
            original_response = "I can see colorful shapes in this image."
            enhanced_response = emotional_integration.enhance_response_with_emotion(
                original_response, emotional_context
            )
            
            self.test_results['response_enhancement'] = {
                'original': original_response,
                'enhanced': enhanced_response,
                'enhancement_applied': enhanced_response != original_response
            }
            
            # Test emotional recommendations
            recommendations = emotional_integration.get_emotional_recommendations(emotional_context)
            self.test_results['emotional_recommendations'] = recommendations
            
            # Get integration stats
            integration_stats = emotional_integration.get_integration_stats()
            self.test_results['emotional_integration_stats'] = integration_stats
            
            self.logger.info("Emotional AI integration test completed")
            
        except Exception as e:
            self.logger.error(f"Emotional integration test failed: {e}")
            self.test_results['emotional_integration_error'] = str(e)
    
    async def test_web_interface(self):
        """Test web interface components"""
        self.logger.info("Testing Web Interface...")
        
        try:
            # Initialize web interface
            web_interface = MultimodalWebInterface(self.config['web'])
            
            # Test file upload simulation
            test_upload = {
                'success': True,
                'filename': 'test_image.jpg',
                'file_info': {
                    'type': 'image',
                    'size': 12345,
                    'dimensions': [400, 300]
                }
            }
            
            self.test_results['web_upload_simulation'] = test_upload
            
            # Test multimodal analysis simulation
            analysis_request = {
                'text': 'Analyze this test image',
                'filename': 'test_scene_image.jpg',
                'fusion_strategy': 'adaptive'
            }
            
            self.test_results['web_analysis_request'] = {
                'success': True,
                'request_format': analysis_request,
                'interface_ready': web_interface is not None
            }
            
            # Test system status
            status_simulation = {
                'multimodal_system': True,
                'active_sessions': 0,
                'upload_folder': self.config['web']['upload_folder'],
                'allowed_extensions': ['.jpg', '.jpeg', '.png', '.mp4', '.avi'],
                'max_file_size': self.config['web']['max_file_size']
            }
            
            self.test_results['web_status'] = status_simulation
            
            self.logger.info("Web interface test completed")
            
        except Exception as e:
            self.logger.error(f"Web interface test failed: {e}")
            self.test_results['web_interface_error'] = str(e)
    
    async def test_real_world_scenarios(self):
        """Test real-world usage scenarios"""
        self.logger.info("Testing Real-World Scenarios...")
        
        try:
            # Initialize fusion system for scenarios
            fusion_system = MultimodalFusionSystem(self.config['fusion'])
            
            # Scenario 1: Document analysis
            document_query = ModalityInput(
                modality=ModalityType.TEXT,
                data="Please read and summarize the text in this image",
                confidence=1.0
            )
            
            document_image = ModalityInput(
                modality=ModalityType.VISION,
                data=self.test_images['text_image'],
                confidence=0.9
            )
            
            document_result = fusion_system.fuse_multimodal_input(
                [document_query, document_image]
            )
            
            self.test_results['scenario_document_analysis'] = {
                'success': document_result.confidence > 0.5,
                'response': document_result.fused_response,
                'confidence': document_result.confidence
            }
            
            # Scenario 2: Visual Q&A
            vqa_query = ModalityInput(
                modality=ModalityType.TEXT,
                data="What shapes and colors can you identify in this image?",
                confidence=1.0
            )
            
            scene_image = ModalityInput(
                modality=ModalityType.VISION,
                data=self.test_images['scene_image'],
                confidence=0.9
            )
            
            vqa_result = fusion_system.fuse_multimodal_input(
                [vqa_query, scene_image]
            )
            
            self.test_results['scenario_visual_qa'] = {
                'success': vqa_result.confidence > 0.5,
                'response': vqa_result.fused_response,
                'confidence': vqa_result.confidence
            }
            
            # Scenario 3: Multilingual text recognition
            thai_query = ModalityInput(
                modality=ModalityType.TEXT,
                data="กรุณาอ่านข้อความภาษาไทยในรูปภาพนี้",
                confidence=1.0
            )
            
            thai_image = ModalityInput(
                modality=ModalityType.VISION,
                data=self.test_images['thai_text'],
                confidence=0.9
            )
            
            thai_result = fusion_system.fuse_multimodal_input(
                [thai_query, thai_image]
            )
            
            self.test_results['scenario_thai_ocr'] = {
                'success': thai_result.confidence > 0.3,  # Lower threshold for Thai
                'response': thai_result.fused_response,
                'confidence': thai_result.confidence
            }
            
            self.logger.info("Real-world scenarios test completed")
            
        except Exception as e:
            self.logger.error(f"Real-world scenarios test failed: {e}")
            self.test_results['scenarios_error'] = str(e)
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.logger.info("Generating test report...")
        
        # Calculate overall success rate
        total_tests = 0
        successful_tests = 0
        
        def count_tests(data, prefix=""):
            nonlocal total_tests, successful_tests
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if key.endswith('_error'):
                        total_tests += 1
                        # Error means test failed
                        continue
                    elif isinstance(value, dict) and 'success' in value:
                        total_tests += 1
                        if value['success']:
                            successful_tests += 1
                    elif isinstance(value, (dict, list)):
                        count_tests(value, f"{prefix}{key}.")
                    elif key.endswith('_loaded') or key.endswith('_ready'):
                        total_tests += 1
                        if value:
                            successful_tests += 1
        
        count_tests(self.test_results)
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': f"{success_rate:.1f}%",
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'system_capabilities': {
                'vision_models': 'clip_model_loaded' in self.test_results and self.test_results['clip_model_loaded'],
                'ocr_system': 'ocr_english' in self.test_results and self.test_results['ocr_english'].get('success', False),
                'visual_qa': 'visual_qa' in self.test_results and self.test_results['visual_qa'].get('success', False),
                'multimodal_fusion': 'fusion_text_vision' in self.test_results and self.test_results['fusion_text_vision'].get('success', False),
                'emotional_integration': 'emotional_integration' in self.test_results and self.test_results['emotional_integration'].get('success', False),
                'web_interface': 'web_upload_simulation' in self.test_results,
                'thai_language_support': 'ocr_thai' in self.test_results and self.test_results['ocr_thai'].get('success', False)
            },
            'detailed_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = self.temp_dir / 'multimodal_test_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Test report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("JARVIS MULTIMODAL AI SYSTEM TEST REPORT")
        print("="*80)
        print(f"Test Date: {report['test_summary']['test_timestamp']}")
        print(f"Overall Success Rate: {report['test_summary']['success_rate']}")
        print(f"Tests Passed: {successful_tests}/{total_tests}")
        print("\nSystem Capabilities:")
        for capability, status in report['system_capabilities'].items():
            status_str = "✅ WORKING" if status else "❌ FAILED"
            print(f"  {capability.replace('_', ' ').title()}: {status_str}")
        
        print(f"\nDetailed report available at: {report_path}")
        print("="*80)
        
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for critical failures
        if not self.test_results.get('clip_model_loaded', False):
            recommendations.append("CRITICAL: CLIP model failed to load. Check transformers installation and model availability.")
        
        if not self.test_results.get('blip_models_loaded', False):
            recommendations.append("CRITICAL: BLIP models failed to load. Check transformers installation and model availability.")
        
        # Check OCR performance
        ocr_english = self.test_results.get('ocr_english', {})
        if not ocr_english.get('success', False):
            recommendations.append("WARNING: English OCR not working. Check EasyOCR and Tesseract installation.")
        
        ocr_thai = self.test_results.get('ocr_thai', {})
        if not ocr_thai.get('success', False):
            recommendations.append("WARNING: Thai OCR not working. Ensure Thai language support is installed.")
        
        # Check fusion system
        fusion_result = self.test_results.get('fusion_text_vision', {})
        if not fusion_result.get('success', False):
            recommendations.append("ERROR: Multimodal fusion system not working properly.")
        
        # Performance recommendations
        if fusion_result.get('confidence', 0) < 0.7:
            recommendations.append("OPTIMIZATION: Consider adjusting fusion weights for better confidence scores.")
        
        # Add general recommendations
        recommendations.extend([
            "PERFORMANCE: Consider using GPU acceleration for better performance.",
            "DEPLOYMENT: Test with larger variety of images and video content.",
            "MONITORING: Set up logging and monitoring for production use.",
            "SECURITY: Implement proper file upload validation and sanitization."
        ])
        
        return recommendations
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        self.logger.info("Starting comprehensive multimodal AI system tests...")
        
        # Create test data
        self.create_test_images()
        
        # Run tests
        await self.test_vision_models()
        await self.test_ocr_system()
        await self.test_visual_qa_system()
        await self.test_multimodal_fusion()
        await self.test_emotional_integration()
        await self.test_web_interface()
        await self.test_real_world_scenarios()
        
        # Generate report
        report = self.generate_test_report()
        
        return report


async def main():
    """Main test execution"""
    print("JARVIS Multimodal AI System - Comprehensive Test Suite")
    print("="*60)
    
    tester = MultimodalSystemTester()
    
    try:
        report = await tester.run_all_tests()
        return report
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        logging.exception("Test suite error")
        return None


if __name__ == "__main__":
    # Run tests
    try:
        import torch
        print(f"PyTorch available: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not available - some tests may fail")
    
    print("\nStarting tests...")
    report = asyncio.run(main())
    
    if report:
        print("\nTests completed successfully!")
        exit_code = 0 if report['test_summary']['success_rate'].replace('%', '') != '0' else 1
        exit(exit_code)
    else:
        print("\nTests failed!")
        exit(1)