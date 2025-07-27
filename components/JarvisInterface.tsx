'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface VoiceVisualizerProps {
  isActive: boolean;
  quality: 'excellent' | 'good' | 'poor' | 'low';
}

const VoiceVisualizer: React.FC<VoiceVisualizerProps> = ({ isActive, quality }) => {
  const bars = Array.from({ length: 10 }, (_, i) => i);
  
  const getQualityColor = () => {
    switch (quality) {
      case 'excellent': return 'from-green-400 to-emerald-500';
      case 'good': return 'from-blue-400 to-cyan-500';
      case 'poor': return 'from-yellow-400 to-orange-500';
      case 'low': return 'from-red-400 to-red-600';
      default: return 'from-blue-400 to-cyan-500';
    }
  };

  return (
    <div className={`h-16 bg-gradient-to-r ${getQualityColor()} rounded-2xl flex items-center justify-center gap-1 px-4 backdrop-blur-sm border border-white/20 ${isActive ? 'shadow-lg' : ''}`}>
      {bars.map((_, index) => (
        <motion.div
          key={index}
          className="w-1 bg-white/80 rounded-full"
          animate={{
            height: isActive ? [8, 32, 16, 28, 12] : 8,
            opacity: isActive ? [0.3, 1, 0.6, 0.9, 0.4] : 0.3,
          }}
          transition={{
            duration: isActive ? 1.5 : 0.3,
            repeat: isActive ? Infinity : 0,
            delay: index * 0.1,
            ease: 'easeInOut',
          }}
        />
      ))}
      <div className="absolute text-white font-medium text-sm">
        {isActive ? '🎙️ Listening...' : '🎙️ Voice Ready'}
      </div>
    </div>
  );
};

const JarvisInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceQuality, setVoiceQuality] = useState<'excellent' | 'good' | 'poor' | 'low'>('good');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Simulate connection
    setTimeout(() => setIsConnected(true), 1000);
  }, []);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `สวัสดีครับ! ผมเข้าใจคำถามของคุณเกี่ยวกับ "${inputText}" แล้ว ✨ ผมสามารถช่วยอธิบายและให้ข้อมูลได้ครับ`,
        isUser: false,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 2000);
  };

  const toggleVoiceRecording = () => {
    setIsRecording(!isRecording);
    // Simulate voice quality changes
    const qualities: ('excellent' | 'good' | 'poor' | 'low')[] = ['excellent', 'good', 'poor', 'low'];
    setVoiceQuality(qualities[Math.floor(Math.random() * qualities.length)]);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-pink-500 bg-[length:400%_400%] animate-gradient-shift relative overflow-hidden flex items-center justify-center p-4">
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute text-2xl"
          animate={{
            x: [0, 100, -50, 0],
            y: [0, -50, 100, 0],
            rotate: [0, 180, 360],
          }}
          transition={{ duration: 20, repeat: Infinity }}
          style={{ top: '10%', left: '8%' }}
        >
          ✨
        </motion.div>
        <motion.div
          className="absolute text-2xl"
          animate={{
            x: [0, -80, 60, 0],
            y: [0, 80, -40, 0],
            rotate: [0, -180, -360],
          }}
          transition={{ duration: 25, repeat: Infinity, delay: -5 }}
          style={{ top: '15%', right: '12%' }}
        >
          🌟
        </motion.div>
        <motion.div
          className="absolute text-2xl"
          animate={{
            x: [0, 120, -80, 0],
            y: [0, -80, 40, 0],
            rotate: [0, 270, 540],
          }}
          transition={{ duration: 30, repeat: Infinity, delay: -10 }}
          style={{ bottom: '20%', left: '15%' }}
        >
          💫
        </motion.div>
        <motion.div
          className="absolute text-2xl"
          animate={{
            x: [0, -60, 90, 0],
            y: [0, 60, -70, 0],
            rotate: [0, -90, -180],
          }}
          transition={{ duration: 18, repeat: Infinity, delay: -8 }}
          style={{ bottom: '25%', right: '8%' }}
        >
          ⚡
        </motion.div>
      </div>

      {/* Main container */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-4xl h-[90vh] bg-white/10 backdrop-blur-2xl rounded-3xl border border-white/20 shadow-2xl p-6 flex flex-col relative"
        style={{
          boxShadow: '0 25px 45px rgba(0, 0, 0, 0.1), 0 0 30px rgba(103, 126, 234, 0.2)',
        }}
      >
        {/* Header */}
        <motion.div
          className="text-center mb-6 p-6 bg-gradient-to-r from-white/15 to-white/5 backdrop-blur-lg rounded-2xl border border-white/10 relative overflow-hidden"
          animate={{
            boxShadow: [
              '0 25px 45px rgba(0, 0, 0, 0.1), 0 0 30px rgba(103, 126, 234, 0.2)',
              '0 25px 45px rgba(0, 0, 0, 0.1), 0 0 40px rgba(240, 147, 251, 0.3)',
            ],
          }}
          transition={{ duration: 8, repeat: Infinity, repeatType: 'reverse' }}
        >
          <motion.div
            className="absolute top-2 right-4 text-2xl"
            animate={{
              rotate: [0, 90, 180, 270, 360],
              scale: [1, 1.2, 1, 1.2, 1],
            }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            ✨
          </motion.div>
          
          <motion.h1
            className="text-5xl font-bold text-white mb-2 bg-gradient-to-r from-white via-gray-100 to-white bg-clip-text text-transparent"
            animate={{
              y: [0, -5, 0],
            }}
            transition={{ duration: 2, repeat: Infinity }}
            style={{
              fontFamily: 'Poppins, sans-serif',
              textShadow: '0 4px 8px rgba(0,0,0,0.3)',
              backgroundSize: '200% 200%',
            }}
          >
            JARVIS
          </motion.h1>
          <p className="text-white/90 text-lg font-medium">
            ผู้ช่วยอัจฉริยะด้วย AI - Thai Language Supported
          </p>
        </motion.div>

        {/* Status bar */}
        <div className="flex justify-between items-center bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-2xl p-4 mb-4 border border-white/10 backdrop-blur-lg">
          <div className="flex items-center gap-3">
            <motion.div
              className="w-3 h-3 bg-green-400 rounded-full shadow-lg"
              animate={{
                scale: [1, 1.2, 1],
                boxShadow: [
                  '0 0 20px rgba(74, 222, 128, 0.6)',
                  '0 0 30px rgba(74, 222, 128, 0.8)',
                  '0 0 20px rgba(74, 222, 128, 0.6)',
                ],
              }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <span className="text-white font-medium">
              {isConnected ? 'เชื่อมต่อแล้ว' : 'กำลังเชื่อมต่อ...'}
            </span>
          </div>
          <div className="text-white/80">
            ✨ Ready to Help
          </div>
        </div>

        {/* Chat container */}
        <div className="flex-1 bg-white/5 rounded-2xl p-6 mb-4 overflow-y-auto backdrop-blur-sm border border-white/10 relative">
          {/* Welcome message */}
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center text-white/80 mb-6"
            >
              <h3 className="text-2xl font-semibold mb-2">🎉 ยินดีต้อนรับสู่ JARVIS!</h3>
              <p className="text-lg">
                พิมพ์ข้อความหรือใช้เสียงเพื่อเริ่มสนทนา เช่น "สวัสดี" หรือ "ปัญญาประดิษฐ์คืออะไร"
              </p>
            </motion.div>
          )}

          {/* Messages */}
          <div className="space-y-4">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.95 }}
                  className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] p-4 rounded-3xl backdrop-blur-lg border border-white/20 shadow-lg relative ${
                      message.isUser
                        ? 'bg-gradient-to-r from-purple-500/80 to-pink-500/80 text-white'
                        : 'bg-gradient-to-r from-green-500/80 to-emerald-500/80 text-white'
                    }`}
                  >
                    <p className="text-base leading-relaxed">{message.text}</p>
                    
                    {/* Chat tail */}
                    <div
                      className={`absolute bottom-[-8px] w-0 h-0 ${
                        message.isUser
                          ? 'right-5 border-l-[15px] border-l-transparent border-t-[15px] border-t-purple-500/80'
                          : 'left-5 border-r-[15px] border-r-transparent border-t-[15px] border-t-green-500/80'
                      }`}
                    />
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Typing indicator */}
            <AnimatePresence>
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="flex justify-start"
                >
                  <div className="max-w-[80%] p-4 rounded-3xl bg-white/10 backdrop-blur-lg border border-white/20 flex items-center gap-3">
                    <div className="flex gap-1">
                      {[0, 1, 2].map((i) => (
                        <motion.div
                          key={i}
                          className="w-2 h-2 bg-white/60 rounded-full"
                          animate={{
                            scale: [1, 1.2, 1],
                            opacity: [0.5, 1, 0.5],
                          }}
                          transition={{
                            duration: 1.4,
                            repeat: Infinity,
                            delay: i * 0.16,
                          }}
                        />
                      ))}
                    </div>
                    <span className="text-white/80 text-sm">JARVIS กำลังคิด...</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <div ref={messagesEndRef} />
        </div>

        {/* Voice visualizer */}
        <div className="mb-4">
          <VoiceVisualizer isActive={isRecording} quality={voiceQuality} />
        </div>

        {/* Input section */}
        <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-2xl rounded-2xl p-6 border border-white/20">
          {/* Voice controls */}
          <div className="flex justify-center gap-4 mb-4">
            <motion.button
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleVoiceRecording}
              className={`px-6 py-3 rounded-full font-semibold text-white border-none cursor-pointer transition-all duration-300 relative overflow-hidden ${
                isRecording
                  ? 'bg-gradient-to-r from-red-500 to-red-600 shadow-lg shadow-red-500/30'
                  : 'bg-gradient-to-r from-red-500 to-red-600 shadow-lg shadow-red-500/30'
              }`}
              animate={isRecording ? { scale: [1, 1.05, 1] } : {}}
              transition={isRecording ? { duration: 1, repeat: Infinity } : {}}
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                initial={{ x: '-100%' }}
                whileHover={{ x: '100%' }}
                transition={{ duration: 0.5 }}
              />
              <span className="relative z-10">
                {isRecording ? '🔴 Recording...' : '🎤 เริ่มพูด'}
              </span>
            </motion.button>
          </div>

          {/* Text input */}
          <div className="flex gap-3 items-center">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="พิมพ์ข้อความหรือพูด... ✨"
              className="flex-1 px-6 py-4 bg-white/10 backdrop-blur-lg border-2 border-white/20 rounded-3xl text-white placeholder-white/60 text-base font-medium transition-all duration-300 focus:outline-none focus:border-white/40 focus:bg-white/15 focus:shadow-lg focus:shadow-white/10"
            />
            <motion.button
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={sendMessage}
              disabled={!inputText.trim()}
              className="px-6 py-4 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-3xl font-semibold cursor-pointer transition-all duration-300 border-none shadow-lg shadow-purple-500/30 relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <motion.div
                className="absolute inset-0 bg-white/20 rounded-full"
                initial={{ width: 0, height: 0 }}
                whileHover={{ width: 300, height: 300 }}
                transition={{ duration: 0.6 }}
                style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}
              />
              <span className="relative z-10">ส่ง</span>
            </motion.button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default JarvisInterface;