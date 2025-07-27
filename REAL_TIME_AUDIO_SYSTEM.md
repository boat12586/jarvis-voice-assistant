# ระบบเสียงแบบเรียลไทม์สำหรับ Live AI Agent

## ภาพรวม

ระบบเสียงแบบเรียลไทม์เป็นองค์ประกอบสำคัญของการปรับปรุง Live AI Agent ที่ช่วยให้การสนทนามีความเป็นธรรมชาติและต่อเนื่อง คล้ายกับ Gemini Live แต่มีความสามารถเฉพาะทางมากกว่า ระบบนี้ออกแบบมาเพื่อให้มีความหน่วงต่ำ (<200ms) และรองรับการสนทนาแบบต่อเนื่องโดยไม่มีการหยุดชะงัก

## องค์ประกอบหลัก

### 1. ไปป์ไลน์การสตรีมเสียงต่อเนื่อง

ไปป์ไลน์การสตรีมเสียงต่อเนื่องจัดการการรับและส่งเสียงแบบเรียลไทม์ โดยมีความหน่วงต่ำและคุณภาพเสียงที่ดี

#### คุณสมบัติหลัก:
- **การรับเสียงต่อเนื่อง**: รับเสียงจากไมโครโฟนอย่างต่อเนื่องโดยไม่มีการหยุดชะงัก
- **การส่งเสียงแบบสตรีม**: ส่งเสียงตอบกลับขณะที่กำลังสร้าง ไม่ต้องรอให้สร้างเสร็จสมบูรณ์
- **การปรับขนาดบัฟเฟอร์แบบไดนามิก**: ปรับขนาดบัฟเฟอร์เสียงตามสภาพเครือข่ายและฮาร์ดแวร์
- **การปรับแต่งรูปแบบเสียง**: ปรับแต่งรูปแบบเสียงเพื่อประสิทธิภาพที่ดีที่สุด
- **การปรับแต่งเฉพาะฮาร์ดแวร์**: ปรับแต่งการประมวลผลเสียงตามฮาร์ดแวร์ที่ใช้งาน

#### การพัฒนา:
```python
class AudioStreamManager:
    def __init__(self):
        self.input_stream = ContinuousAudioInput()
        self.output_stream = StreamingAudioOutput()
        self.latency_optimizer = LatencyOptimizer()
        self.audio_format = AudioFormatConverter()
        
    async def start_live_mode(self):
        # ตั้งค่าพารามิเตอร์การสตรีมเสียง
        input_params = {
            'sample_rate': 48000,
            'channels': 1,
            'format': 'float32',
            'frames_per_buffer': 480  # 10ms ที่ 48kHz
        }
        
        output_params = {
            'sample_rate': 48000,
            'channels': 1,
            'format': 'float32',
            'frames_per_buffer': 960  # 20ms ที่ 48kHz
        }
        
        # เริ่มสตรีมเสียงต่อเนื่อง
        await self.input_stream.start(input_params)
        await self.output_stream.start(output_params)
        
        # เริ่มการตรวจจับกิจกรรมเสียง
        self.vad = VoiceActivityDetector()
        self.vad.start_detection(self.input_stream)
        
        # เริ่มการประมวลผลเสียงแบบเรียลไทม์
        self.audio_processor = RealTimeAudioProcessor()
        self.audio_processor.start_processing(self.input_stream, self.output_stream)
        
    def optimize_latency(self):
        # วัดความหน่วงปัจจุบัน
        current_latency = self.measure_end_to_end_latency()
        
        if current_latency > 200:  # เป้าหมาย <200ms
            # ปรับขนาดบัฟเฟอร์เพื่อลดความหน่วง
            new_buffer_size = max(240, self.input_stream.frames_per_buffer // 2)
            self.input_stream.set_frames_per_buffer(new_buffer_size)
            
            # ปรับรูปแบบเสียงเพื่อประสิทธิภาพ
            self.audio_format.optimize_for_latency()
            
            # ปรับแต่งเฉพาะฮาร์ดแวร์
            if self.is_high_performance_hardware():
                self.enable_hardware_acceleration()
            else:
                self.reduce_processing_quality()
                
        # ตรวจสอบความหน่วงอีกครั้งหลังการปรับแต่ง
        new_latency = self.measure_end_to_end_latency()
        print(f"Latency optimized: {current_latency}ms -> {new_latency}ms")
```

### 2. การรู้จำเสียงแบบเรียลไทม์

ระบบการรู้จำเสียงแบบเรียลไทม์แปลงเสียงพูดเป็นข้อความทันทีขณะที่ผู้ใช้พูด โดยไม่ต้องรอให้ผู้ใช้หยุดพูด

#### คุณสมบัติหลัก:
- **การรู้จำเสียงแบบสตรีม**: แปลงเสียงพูดเป็นข้อความแบบเรียลไทม์ขณะที่ผู้ใช้พูด
- **การตรวจจับกิจกรรมเสียง**: ตรวจจับเมื่อผู้ใช้เริ่มและหยุดพูด
- **บัฟเฟอร์การถอดความแบบสตรีม**: เก็บและอัปเดตข้อความที่ถอดความแบบเรียลไทม์
- **การตรวจจับภาษา**: ตรวจจับและสลับระหว่างภาษาไทยและอังกฤษโดยอัตโนมัติ

#### การพัฒนา:
```python
class RealTimeAudioProcessor:
    def __init__(self):
        self.vad = VoiceActivityDetector()
        self.streaming_whisper = StreamingWhisperProcessor()
        self.transcription_buffer = StreamingTranscriptionBuffer()
        self.language_detector = LanguageDetector()
        
    async def process_audio_stream(self, audio_stream):
        # ตั้งค่า Faster-Whisper สำหรับการประมวลผลแบบสตรีม
        self.streaming_whisper.initialize(
            model_size="medium",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )
        
        # เริ่มการประมวลผลเสียงแบบสตรีม
        async for audio_chunk in audio_stream:
            # ตรวจสอบว่ามีเสียงพูดหรือไม่
            if self.vad.is_speech(audio_chunk):
                # ส่งชิ้นส่วนเสียงไปยัง Whisper สำหรับการถอดความแบบสตรีม
                partial_text = await self.streaming_whisper.process_chunk(audio_chunk)
                
                if partial_text:
                    # ตรวจจับภาษาถ้าเป็นข้อความใหม่
                    if self.transcription_buffer.is_empty():
                        detected_language = self.language_detector.detect(partial_text)
                        self.streaming_whisper.set_language(detected_language)
                    
                    # อัปเดตบัฟเฟอร์การถอดความ
                    self.transcription_buffer.update(partial_text)
                    
                    # ส่งการถอดความแบบเรียลไทม์ไปยังระบบ AI
                    await self.send_realtime_transcription(self.transcription_buffer.get_text())
            
            # ตรวจสอบการหยุดพูด
            elif self.vad.is_silence(audio_chunk) and self.vad.was_speaking():
                # ส่งการถอดความสุดท้ายและรีเซ็ตบัฟเฟอร์
                final_text = self.transcription_buffer.finalize()
                await self.send_final_transcription(final_text)
                self.transcription_buffer.reset()
```

### 3. การจัดการการไหลของการสนทนา

ระบบการจัดการการไหลของการสนทนาควบคุมการโต้ตอบระหว่างผู้ใช้และ AI เพื่อให้การสนทนาเป็นธรรมชาติและมีการขัดจังหวะที่เหมาะสม

#### คุณสมบัติหลัก:
- **การจัดการการขัดจังหวะ**: ตรวจจับและจัดการเมื่อผู้ใช้ขัดจังหวะ AI หรือในทางกลับกัน
- **ระบบการผลัดกันพูด**: จัดการลำดับการพูดระหว่างผู้ใช้และ AI
- **การสร้างการตอบสนองแบบเรียลไทม์**: สร้างการตอบสนองของ AI แบบเรียลไทม์ขณะที่ได้รับข้อความ
- **การจัดการสถานะการสนทนา**: ติดตามและจัดการสถานะของการสนทนา

#### การพัฒนา:
```python
class ConversationFlowManager:
    def __init__(self):
        self.interrupt_handler = ConversationInterruptHandler()
        self.turn_taking_system = TurnTakingSystem()
        self.response_generator = RealTimeResponseGenerator()
        self.conversation_state = ConversationStateManager()
        
    async def manage_conversation_flow(self, user_stream, ai_stream):
        # ตั้งค่าการจัดการการไหลของการสนทนา
        self.turn_taking_system.initialize()
        self.conversation_state.set_state("waiting_for_user")
        
        # เริ่มการติดตามสตรีมผู้ใช้และ AI
        user_task = asyncio.create_task(self.process_user_stream(user_stream))
        ai_task = asyncio.create_task(self.process_ai_stream(ai_stream))
        
        # รอทั้งสองงานเสร็จสิ้น
        await asyncio.gather(user_task, ai_task)
        
    async def process_user_stream(self, user_stream):
        async for user_text in user_stream:
            # ตรวจสอบว่า AI กำลังพูดอยู่หรือไม่
            if self.conversation_state.get_state() == "ai_speaking":
                # จัดการการขัดจังหวะถ้าผู้ใช้เริ่มพูดขณะที่ AI กำลังพูด
                if self.interrupt_handler.should_interrupt_ai(user_text):
                    await self.interrupt_handler.interrupt_ai()
                    self.conversation_state.set_state("user_speaking")
            else:
                # ตั้งค่าสถานะเป็นผู้ใช้กำลังพูด
                self.conversation_state.set_state("user_speaking")
            
            # ประมวลผลข้อความผู้ใช้
            await self.process_user_input(user_text)
            
            # ตรวจสอบการจบประโยคหรือการหยุดพูด
            if self.turn_taking_system.is_user_turn_complete(user_text):
                self.conversation_state.set_state("ai_turn")
                self.turn_taking_system.switch_to_ai()
                
    async def process_ai_stream(self, ai_stream):
        async for ai_text in ai_stream:
            # ตรวจสอบว่าผู้ใช้ขัดจังหวะหรือไม่
            if self.conversation_state.get_state() == "user_speaking":
                # หยุดการพูดของ AI ถ้าผู้ใช้ขัดจังหวะ
                continue
                
            # ตั้งค่าสถานะเป็น AI กำลังพูด
            self.conversation_state.set_state("ai_speaking")
            
            # ส่งข้อความ AI ไปยังระบบ TTS แบบสตรีม
            await self.stream_ai_response(ai_text)
            
            # ตรวจสอบการจบประโยคหรือการหยุดพูด
            if self.turn_taking_system.is_ai_turn_complete(ai_text):
                self.conversation_state.set_state("user_turn")
                self.turn_taking_system.switch_to_user()
```

### 4. การบูรณาการ F5-TTS แบบสตรีม

การบูรณาการ F5-TTS แบบสตรีมแปลงข้อความเป็นเสียงพูดแบบเรียลไทม์ด้วยเสียง J.A.R.V.I.S ที่สมจริงและมีเอฟเฟกต์เสียงที่เหมาะสม

#### คุณสมบัติหลัก:
- **การสังเคราะห์เสียงแบบสตรีม**: สร้างเสียงพูดแบบเรียลไทม์ขณะที่ได้รับข้อความ
- **การประมวลผลเอฟเฟกต์เสียง**: เพิ่มเอฟเฟกต์เสียงเรเวิร์บและโทนโลหะแบบ J.A.R.V.I.S
- **ระบบผสมเสียง**: จัดการการตอบสนองเสียงที่ซ้อนทับกัน
- **การปรับเปลี่ยนอารมณ์เสียง**: ปรับเปลี่ยนอารมณ์ของเสียงตามบริบท

#### การพัฒนา:
```python
class StreamingTTSSystem:
    def __init__(self):
        self.tts_pipeline = F5TTSPipeline()
        self.voice_effects = VoiceEffectProcessor()
        self.audio_mixer = AudioMixingSystem()
        self.emotion_modulator = VoiceEmotionModulator()
        
    async def initialize(self):
        # โหลดโมเดล F5-TTS และเสียง J.A.R.V.I.S
        await self.tts_pipeline.load_model("f5-tts-jarvis")
        
        # ตั้งค่าเอฟเฟกต์เสียง
        self.voice_effects.set_reverb(0.2)  # เรเวิร์บเล็กน้อย
        self.voice_effects.set_metallic_tone(0.3)  # โทนโลหะปานกลาง
        
        # ตั้งค่าระบบผสมเสียง
        self.audio_mixer.initialize(
            channels=1,
            sample_rate=48000,
            buffer_size=1024
        )
        
    async def generate_streaming_speech(self, text_stream):
        # ตั้งค่าบัฟเฟอร์สำหรับการสร้างเสียงแบบสตรีม
        text_buffer = ""
        sentence_end_pattern = re.compile(r'[.!?।\n]')
        
        async for text_chunk in text_stream:
            # เพิ่มชิ้นส่วนข้อความใหม่ลงในบัฟเฟอร์
            text_buffer += text_chunk
            
            # ตรวจสอบว่ามีประโยคที่สมบูรณ์หรือไม่
            match = sentence_end_pattern.search(text_buffer)
            
            if match or len(text_buffer) > 50:  # ประมวลผลเมื่อพบจุดสิ้นสุดประโยคหรือข้อความยาวพอ
                # แยกข้อความที่จะประมวลผล
                if match:
                    process_text = text_buffer[:match.end()]
                    text_buffer = text_buffer[match.end():]
                else:
                    process_text = text_buffer
                    text_buffer = ""
                
                # ตรวจจับอารมณ์จากข้อความ
                emotion = self.detect_emotion(process_text)
                self.emotion_modulator.set_emotion(emotion)
                
                # สร้างเสียงพูดสำหรับข้อความ
                audio_chunk = await self.tts_pipeline.generate_speech(process_text)
                
                # ใช้เอฟเฟกต์เสียง
                processed_audio = self.voice_effects.process(audio_chunk)
                modulated_audio = self.emotion_modulator.modulate(processed_audio)
                
                # ส่งชิ้นส่วนเสียงไปยังระบบเอาต์พุต
                await self.audio_mixer.add_audio(modulated_audio)
                await self.output_audio_chunk(modulated_audio)
                
    def detect_emotion(self, text):
        # ตรวจจับอารมณ์พื้นฐานจากข้อความ
        if re.search(r'urgent|emergency|critical|immediately', text, re.IGNORECASE):
            return "urgent"
        elif re.search(r'happy|glad|excellent|amazing', text, re.IGNORECASE):
            return "happy"
        elif re.search(r'sorry|unfortunate|regret', text, re.IGNORECASE):
            return "apologetic"
        else:
            return "neutral"
```

## การเพิ่มประสิทธิภาพและการทดสอบ

### การเพิ่มประสิทธิภาพความหน่วง

การเพิ่มประสิทธิภาพความหน่วงเป็นสิ่งสำคัญสำหรับประสบการณ์การสนทนาที่เป็นธรรมชาติ โดยมีเป้าหมายความหน่วงน้อยกว่า 200ms

#### เทคนิคการเพิ่มประสิทธิภาพ:
- **การประมวลผลแบบขนาน**: ใช้การประมวลผลแบบขนานสำหรับการรู้จำเสียงและการสังเคราะห์เสียง
- **การปรับขนาดบัฟเฟอร์แบบไดนามิก**: ปรับขนาดบัฟเฟอร์เสียงตามสภาพเครือข่ายและฮาร์ดแวร์
- **การใช้ GPU เร่งความเร็ว**: ใช้ GPU สำหรับการประมวลผลโมเดล AI เพื่อลดความหน่วง
- **การบีบอัดเสียงที่มีประสิทธิภาพ**: ใช้โคเด็กเสียงที่มีประสิทธิภาพเพื่อลดแบนด์วิดท์
- **การเพิ่มประสิทธิภาพการโหลดโมเดล**: โหลดโมเดลล่วงหน้าและใช้การ quantization เพื่อลดเวลาโหลด

### การทดสอบประสิทธิภาพ

การทดสอบประสิทธิภาพอย่างครอบคลุมเป็นสิ่งสำคัญเพื่อให้แน่ใจว่าระบบเสียงแบบเรียลไทม์ทำงานได้อย่างมีประสิทธิภาพและน่าเชื่อถือ

#### แผนการทดสอบ:
- **การทดสอบความหน่วง**: วัดความหน่วงจากต้นจนจบในสถานการณ์ต่างๆ
- **การทดสอบภายใต้โหลด**: ทดสอบประสิทธิภาพภายใต้โหลดระบบต่างๆ
- **การทดสอบการขัดจังหวะ**: ตรวจสอบว่าระบบจัดการการขัดจังหวะได้อย่างถูกต้อง
- **การทดสอบการตรวจจับภาษา**: ตรวจสอบความแม่นยำในการตรวจจับและสลับภาษา
- **การทดสอบคุณภาพเสียง**: ประเมินคุณภาพเสียงที่สังเคราะห์ในสภาพแวดล้อมต่างๆ

## ข้อควรพิจารณาในการพัฒนา

### ความต้องการของระบบ
- **CPU**: แนะนำ 8 คอร์ขึ้นไปสำหรับการประมวลผลเสียงแบบเรียลไทม์
- **GPU**: NVIDIA RTX 2050+ สำหรับการประมวลผลโมเดล AI
- **RAM**: 16GB+ สำหรับการโหลดโมเดลและบัฟเฟอร์เสียง
- **เครือข่าย**: การเชื่อมต่อที่เสถียรสำหรับการอัปเดตโมเดลและข้อมูล
- **อุปกรณ์เสียง**: ไมโครโฟนและลำโพงคุณภาพดีสำหรับประสบการณ์ที่ดีที่สุด

### การจัดการข้อผิดพลาด
- **การฟื้นฟูจากการหยุดชะงักของเสียง**: กลไกการฟื้นฟูเมื่อเกิดการหยุดชะงักของเสียง
- **การจัดการความล้มเหลวของโมเดล**: กลยุทธ์สำรองเมื่อโมเดล AI ล้มเหลว
- **การปรับตัวต่อสภาพแวดล้อมที่มีเสียงรบกวน**: การปรับตัวต่อสภาพแวดล้อมที่มีเสียงรบกวนสูง
- **การจัดการการสูญเสียการเชื่อมต่อ**: การจัดการเมื่อการเชื่อมต่อเครือข่ายไม่เสถียร
- **การแจ้งเตือนผู้ใช้**: การแจ้งเตือนผู้ใช้เมื่อเกิดปัญหาและวิธีแก้ไข

## สรุป

ระบบเสียงแบบเรียลไทม์เป็นองค์ประกอบสำคัญของการปรับปรุง Live AI Agent ที่ช่วยให้การสนทนามีความเป็นธรรมชาติและต่อเนื่อง ด้วยการบูรณาการเทคโนโลยีล่าสุด เช่น Faster-Whisper และ F5-TTS พร้อมการเพิ่มประสิทธิภาพความหน่วงและการจัดการการไหลของการสนทนา ระบบนี้จะมอบประสบการณ์การสนทนาที่ราบรื่นและเป็นธรรมชาติกับ AI Agent เฉพาะทางต่างๆ