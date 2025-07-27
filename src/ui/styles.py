"""
Styles and themes for Jarvis Voice Assistant UI
Implements glassmorphic design with J.A.R.V.I.S-inspired colors
"""

from typing import Dict, Any


class JarvisStyles:
    """Style definitions for the Jarvis UI"""
    
    @staticmethod
    def get_main_window_style(config: Dict[str, Any]) -> str:
        """Get main window stylesheet"""
        colors = config.get('colors', {})
        opacity = config.get('opacity', 0.9)
        
        primary = colors.get('primary', '#00d4ff')
        secondary = colors.get('secondary', '#0099cc')
        accent = colors.get('accent', '#ff6b35')
        background = colors.get('background', '#1a1a1a')
        text = colors.get('text', '#ffffff')
        
        return f"""
        QMainWindow {{
            background: transparent;
        }}
        
        #mainContainer {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(26, 26, 26, {opacity * 0.8}),
                stop:1 rgba(0, 212, 255, {opacity * 0.1}));
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }}
        
        #titleLabel {{
            color: {primary};
            font-size: 32px;
            font-weight: bold;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }}
        
        #closeButton {{
            background: rgba(255, 107, 53, 0.8);
            border: 1px solid rgba(255, 107, 53, 0.5);
            border-radius: 15px;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }}
        
        #closeButton:hover {{
            background: rgba(255, 107, 53, 1.0);
            border: 1px solid rgba(255, 107, 53, 0.8);
        }}
        
        #closeButton:pressed {{
            background: rgba(255, 107, 53, 0.6);
        }}
        """
    
    @staticmethod
    def get_action_button_style(config: Dict[str, Any]) -> str:
        """Get action button stylesheet"""
        colors = config.get('colors', {})
        
        primary = colors.get('primary', '#00d4ff')
        secondary = colors.get('secondary', '#0099cc')
        accent = colors.get('accent', '#ff6b35')
        
        return f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(0, 212, 255, 0.2),
                stop:1 rgba(0, 153, 204, 0.2));
            border: 2px solid rgba(0, 212, 255, 0.4);
            border-radius: 15px;
            color: white;
            font-size: 14px;
            font-weight: bold;
            padding: 10px 20px;
            min-width: 120px;
            min-height: 50px;
        }}
        
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(0, 212, 255, 0.4),
                stop:1 rgba(0, 153, 204, 0.4));
            border: 2px solid rgba(0, 212, 255, 0.6);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }}
        
        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(0, 212, 255, 0.6),
                stop:1 rgba(0, 153, 204, 0.6));
            border: 2px solid rgba(0, 212, 255, 0.8);
        }}
        
        QPushButton:disabled {{
            background: rgba(100, 100, 100, 0.3);
            border: 2px solid rgba(100, 100, 100, 0.5);
            color: rgba(255, 255, 255, 0.5);
        }}
        
        QPushButton[active="true"] {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(255, 107, 53, 0.6),
                stop:1 rgba(255, 107, 53, 0.4));
            border: 2px solid rgba(255, 107, 53, 0.8);
            box-shadow: 0 0 25px rgba(255, 107, 53, 0.6);
        }}
        """
    
    @staticmethod
    def get_status_bar_style(config: Dict[str, Any]) -> str:
        """Get status bar stylesheet"""
        colors = config.get('colors', {})
        
        primary = colors.get('primary', '#00d4ff')
        text = colors.get('text', '#ffffff')
        
        return f"""
        QWidget {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 10px;
            padding: 5px 10px;
        }}
        
        QLabel {{
            color: {text};
            font-size: 12px;
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }}
        
        QLabel[status="error"] {{
            color: #ff4444;
        }}
        
        QLabel[status="warning"] {{
            color: #ffaa00;
        }}
        
        QLabel[status="success"] {{
            color: #00ff88;
        }}
        """
    
    @staticmethod
    def get_voice_visualizer_style(config: Dict[str, Any]) -> str:
        """Get voice visualizer stylesheet"""
        colors = config.get('colors', {})
        
        primary = colors.get('primary', '#00d4ff')
        secondary = colors.get('secondary', '#0099cc')
        
        return f"""
        QWidget {{
            background: transparent;
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 50px;
        }}
        
        QWidget[active="true"] {{
            border: 2px solid rgba(0, 212, 255, 0.8);
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
        }}
        
        QWidget[listening="true"] {{
            border: 2px solid rgba(255, 107, 53, 0.8);
            box-shadow: 0 0 30px rgba(255, 107, 53, 0.6);
        }}
        """
    
    @staticmethod
    def get_overlay_style(config: Dict[str, Any]) -> str:
        """Get overlay window stylesheet"""
        colors = config.get('colors', {})
        opacity = config.get('opacity', 0.9)
        
        background = colors.get('background', '#1a1a1a')
        primary = colors.get('primary', '#00d4ff')
        text = colors.get('text', '#ffffff')
        
        return f"""
        QWidget {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(26, 26, 26, {opacity * 0.9}),
                stop:1 rgba(0, 212, 255, {opacity * 0.1}));
            border: 2px solid rgba(0, 212, 255, 0.5);
            border-radius: 15px;
            color: {text};
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }}
        
        QLabel {{
            color: {text};
            background: transparent;
            border: none;
        }}
        
        QLabel[role="title"] {{
            color: {primary};
            font-size: 18px;
            font-weight: bold;
        }}
        
        QLabel[role="content"] {{
            font-size: 14px;
            line-height: 1.4;
        }}
        """