from setuptools import setup, find_packages

setup(
    name="jarvis-voice-assistant",
    version="0.1.0",
    description="A J.A.R.V.I.S-inspired local voice assistant",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "PyQt6>=6.6.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "openai-whisper>=20231117",
        "TTS>=0.21.3",
        "sounddevice>=0.4.6",
        "numpy>=1.24.3",
        "scipy>=1.11.4",
        "librosa>=0.10.1",
        "pydub>=0.25.1",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "websockets>=12.0",
        "PyYAML>=6.0.1",
        "python-dotenv>=1.0.0",
        "pillow>=10.1.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "jarvis=jarvis.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)