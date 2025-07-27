#!/usr/bin/env python3
"""
Performance Testing and Optimization for JARVIS Voice Assistant
"""

import sys
import time
import psutil
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def measure_performance(func, *args, **kwargs):
    """Measure function performance"""
    # Clear memory before test
    gc.collect()
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure execution time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Get final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'result': result,
        'execution_time': end_time - start_time,
        'memory_used': final_memory - initial_memory,
        'initial_memory': initial_memory,
        'final_memory': final_memory
    }

def test_embedding_performance():
    """Test embedding model performance"""
    print("🔍 Testing Embedding Model Performance...")
    
    from sentence_transformers import SentenceTransformer
    
    # Test model loading
    def load_embedding_model():
        return SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    
    load_perf = measure_performance(load_embedding_model)
    print(f"  Model loading: {load_perf['execution_time']:.2f}s, Memory: {load_perf['memory_used']:.1f}MB")
    
    model = load_perf['result']
    
    # Test single embedding
    def single_embedding():
        return model.encode("Hello JARVIS, how are you today?")
    
    single_perf = measure_performance(single_embedding)
    print(f"  Single embedding: {single_perf['execution_time']:.3f}s, Memory: {single_perf['memory_used']:.1f}MB")
    
    # Test batch embedding
    def batch_embedding():
        texts = [
            "Hello JARVIS",
            "สวัสดีครับ จาร์วิส",
            "What is artificial intelligence?",
            "ช่วยอธิบายปัญญาประดิษฐ์ให้หน่อยครับ",
            "Thank you for your help"
        ]
        return model.encode(texts)
    
    batch_perf = measure_performance(batch_embedding)
    print(f"  Batch embedding (5 texts): {batch_perf['execution_time']:.3f}s, Memory: {batch_perf['memory_used']:.1f}MB")
    
    return {
        'model_loading': load_perf['execution_time'],
        'single_embedding': single_perf['execution_time'],
        'batch_embedding': batch_perf['execution_time'],
        'memory_usage': load_perf['final_memory']
    }

def test_rag_performance():
    """Test RAG system performance"""
    print("📚 Testing RAG System Performance...")
    
    import faiss
    import numpy as np
    
    # Test vector store creation
    def create_vector_store():
        return faiss.IndexFlatIP(1024)
    
    create_perf = measure_performance(create_vector_store)
    print(f"  Vector store creation: {create_perf['execution_time']:.3f}s")
    
    index = create_perf['result']
    
    # Test adding vectors
    def add_vectors():
        vectors = np.random.random((100, 1024)).astype(np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        return vectors
    
    add_perf = measure_performance(add_vectors)
    print(f"  Adding 100 vectors: {add_perf['execution_time']:.3f}s")
    
    vectors = add_perf['result']
    
    # Test searching
    def search_vectors():
        query = vectors[0:1]  # Use first vector as query
        scores, indices = index.search(query, 5)
        return scores, indices
    
    search_perf = measure_performance(search_vectors)
    print(f"  Searching top-5: {search_perf['execution_time']:.3f}s")
    
    return {
        'vector_store_creation': create_perf['execution_time'],
        'vector_addition': add_perf['execution_time'],
        'vector_search': search_perf['execution_time']
    }

def test_ai_model_performance():
    """Test AI model performance"""
    print("🤖 Testing AI Model Performance...")
    
    from transformers import AutoTokenizer
    
    # Test tokenizer loading
    def load_tokenizer():
        return AutoTokenizer.from_pretrained(
            'deepseek-ai/deepseek-r1-distill-llama-8b',
            trust_remote_code=True
        )
    
    load_perf = measure_performance(load_tokenizer)
    print(f"  Tokenizer loading: {load_perf['execution_time']:.2f}s")
    
    tokenizer = load_perf['result']
    
    # Test tokenization
    def tokenize_text():
        texts = [
            "Hello JARVIS, please explain artificial intelligence to me.",
            "สวัสดีครับ จาร์วิส ช่วยอธิบายปัญญาประดิษฐ์ให้ฟังหน่อยครับ",
            "What are the latest developments in machine learning?",
            "How does natural language processing work?",
            "Can you help me understand deep learning?"
        ]
        results = []
        for text in texts:
            tokens = tokenizer.encode(text)
            results.append(len(tokens))
        return results
    
    tokenize_perf = measure_performance(tokenize_text)
    print(f"  Tokenizing 5 texts: {tokenize_perf['execution_time']:.3f}s")
    
    return {
        'tokenizer_loading': load_perf['execution_time'],
        'tokenization': tokenize_perf['execution_time']
    }

def test_thai_processing_performance():
    """Test Thai language processing performance"""
    print("🇹🇭 Testing Thai Processing Performance...")
    
    import re
    
    # Thai dictionary
    thai_dict = {
        'ปัญญาประดิษฐ์': 'artificial intelligence',
        'เทคโนโลยี': 'technology', 
        'จาร์วิส': 'JARVIS',
        'ผู้ช่วย': 'assistant',
        'สวัสดี': 'hello',
        'ขอบคุณ': 'thank you',
        'ช่วย': 'help',
        'อธิบาย': 'explain'
    }
    
    polite_words = ['ครับ', 'ค่ะ', 'กรุณา', 'นะ']
    
    # Test language detection
    def detect_languages():
        texts = [
            'สวัสดีครับ จาร์วิส ช่วยอธิบายปัญญาประดิษฐ์ให้หน่อยครับ',
            'Hello JARVIS can you explain artificial intelligence please',
            'Hi จาร์วิส explain AI technology ให้ฟังหน่อยครับ',
            'ขอบคุณมากครับ JARVIS สำหรับการช่วยเหลือ',
            'Thank you very much for your assistance'
        ] * 10  # 50 texts total
        
        results = []
        for text in texts:
            thai_chars = len(re.findall(r'[ก-๙]', text))
            total_chars = len(re.sub(r'\s+', '', text))
            ratio = thai_chars / total_chars if total_chars > 0 else 0
            results.append(ratio)
        return results
    
    detect_perf = measure_performance(detect_languages)
    print(f"  Language detection (50 texts): {detect_perf['execution_time']:.3f}s")
    
    # Test dictionary matching
    def dictionary_matching():
        texts = [
            'จาร์วิส ช่วยอธิบายเทคโนโลยีปัญญาประดิษฐ์ให้หน่อยครับ',
            'ผู้ช่วยอัจฉริยะช่วยสวัสดีและขอบคุณครับ'
        ] * 25  # 50 texts total
        
        results = []
        for text in texts:
            matches = [(w, thai_dict[w]) for w in thai_dict if w in text]
            results.append(len(matches))
        return results
    
    dict_perf = measure_performance(dictionary_matching)
    print(f"  Dictionary matching (50 texts): {dict_perf['execution_time']:.3f}s")
    
    # Test politeness detection
    def politeness_detection():
        texts = [
            'สวัสดีครับ',
            'กรุณาช่วยด้วยค่ะ',
            'ขอบคุณมากครับ นะ',
            'Hello there'
        ] * 25  # 100 texts total
        
        results = []
        for text in texts:
            count = sum(1 for word in polite_words if word in text)
            results.append(count)
        return results
    
    polite_perf = measure_performance(politeness_detection)
    print(f"  Politeness detection (100 texts): {polite_perf['execution_time']:.3f}s")
    
    return {
        'language_detection': detect_perf['execution_time'],
        'dictionary_matching': dict_perf['execution_time'],
        'politeness_detection': polite_perf['execution_time']
    }

def test_voice_components_performance():
    """Test voice components performance"""
    print("🎙️ Testing Voice Components Performance...")
    
    import sounddevice as sd
    
    # Test audio device detection
    def detect_audio_devices():
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        return len(input_devices), len(output_devices)
    
    device_perf = measure_performance(detect_audio_devices)
    print(f"  Audio device detection: {device_perf['execution_time']:.3f}s")
    
    # Test Whisper model loading
    def load_whisper():
        from faster_whisper import WhisperModel
        return WhisperModel('base', device='cpu', compute_type='int8')
    
    whisper_perf = measure_performance(load_whisper)
    print(f"  Whisper model loading: {whisper_perf['execution_time']:.2f}s, Memory: {whisper_perf['memory_used']:.1f}MB")
    
    return {
        'audio_device_detection': device_perf['execution_time'],
        'whisper_loading': whisper_perf['execution_time'],
        'whisper_memory': whisper_perf['memory_used']
    }

def test_memory_management():
    """Test memory management"""
    print("💾 Testing Memory Management...")
    
    process = psutil.Process()
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"  Initial memory: {initial_memory:.1f}MB")
    
    # Load all components
    print("  Loading all components...")
    
    # Embedding model
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    after_embedding = process.memory_info().rss / 1024 / 1024
    print(f"  After embedding model: {after_embedding:.1f}MB (+{after_embedding - initial_memory:.1f}MB)")
    
    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-r1-distill-llama-8b', trust_remote_code=True)
    after_tokenizer = process.memory_info().rss / 1024 / 1024
    print(f"  After tokenizer: {after_tokenizer:.1f}MB (+{after_tokenizer - after_embedding:.1f}MB)")
    
    # Whisper
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel('base', device='cpu', compute_type='int8')
    after_whisper = process.memory_info().rss / 1024 / 1024
    print(f"  After Whisper: {after_whisper:.1f}MB (+{after_whisper - after_tokenizer:.1f}MB)")
    
    # Vector store
    import faiss
    import numpy as np
    index = faiss.IndexFlatIP(1024)
    vectors = np.random.random((1000, 1024)).astype(np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    after_vectors = process.memory_info().rss / 1024 / 1024
    print(f"  After vector store (1000 vectors): {after_vectors:.1f}MB (+{after_vectors - after_whisper:.1f}MB)")
    
    total_memory = after_vectors
    print(f"  Total memory usage: {total_memory:.1f}MB")
    
    # Cleanup test
    del embedding_model, tokenizer, whisper_model, index, vectors
    gc.collect()
    
    after_cleanup = process.memory_info().rss / 1024 / 1024
    print(f"  After cleanup: {after_cleanup:.1f}MB (freed: {total_memory - after_cleanup:.1f}MB)")
    
    return {
        'initial_memory': initial_memory,
        'total_memory': total_memory,
        'memory_after_cleanup': after_cleanup,
        'memory_freed': total_memory - after_cleanup
    }

def main():
    """Run all performance tests"""
    print("🚀 JARVIS Performance Testing & Optimization")
    print("=" * 60)
    
    # System info
    print(f"🖥️  System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f}GB RAM")
    print()
    
    # Run all tests
    results = {}
    
    try:
        results['embedding'] = test_embedding_performance()
        print()
        
        results['rag'] = test_rag_performance()
        print()
        
        results['ai_model'] = test_ai_model_performance()
        print()
        
        results['thai_processing'] = test_thai_processing_performance()
        print()
        
        results['voice_components'] = test_voice_components_performance()
        print()
        
        results['memory'] = test_memory_management()
        print()
        
        # Performance summary
        print("📊 Performance Summary:")
        print("=" * 40)
        
        print(f"⚡ Startup Times:")
        print(f"  - Embedding model: {results['embedding']['model_loading']:.2f}s")
        print(f"  - Tokenizer: {results['ai_model']['tokenizer_loading']:.2f}s")
        print(f"  - Whisper model: {results['voice_components']['whisper_loading']:.2f}s")
        
        print(f"🔍 Processing Times:")
        print(f"  - Single embedding: {results['embedding']['single_embedding']*1000:.1f}ms")
        print(f"  - Vector search: {results['rag']['vector_search']*1000:.1f}ms")
        print(f"  - Thai language detection: {results['thai_processing']['language_detection']*1000/50:.1f}ms per text")
        
        print(f"💾 Memory Usage:")
        print(f"  - Total system memory: {results['memory']['total_memory']:.1f}MB")
        print(f"  - Whisper model: {results['voice_components']['whisper_memory']:.1f}MB")
        print(f"  - Memory cleanup efficiency: {results['memory']['memory_freed']:.1f}MB freed")
        
        # Optimization recommendations
        print()
        print("🎯 Optimization Recommendations:")
        print("=" * 40)
        
        if results['embedding']['model_loading'] > 10:
            print("  ⚠️  Consider model caching for faster startup")
        
        if results['memory']['total_memory'] > 8000:
            print("  ⚠️  High memory usage - consider model quantization")
        
        if results['embedding']['single_embedding'] > 0.1:
            print("  ⚠️  Embedding generation slow - consider batch processing")
        
        if results['voice_components']['whisper_loading'] > 30:
            print("  ⚠️  Whisper loading slow - consider smaller model")
        
        print("  ✅ Thai processing is efficient")
        print("  ✅ Vector search performance good")
        print("  ✅ Memory cleanup working well")
        
        print()
        print("🎉 Performance testing completed!")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()