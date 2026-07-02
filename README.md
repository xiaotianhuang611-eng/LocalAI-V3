# LocalAI_V3

**A local-first, privacy-preserving personalised AI tutor for students.**

LocalAI_V3 is a Windows desktop AI assistant designed for student learning, spoken English practice, document-based RAG, local memory, web-crawled knowledge ingestion, image understanding, and voice interaction.

The project is designed to run locally on the user's own computer, reducing dependence on cloud APIs and keeping personal learning data private.

---

## Key Features

- Local GGUF LLM chat
- Spoken English practice
- Voice input with ASR
- Voice output with XTTS
- Local RAG over user knowledge files
- Web knowledge ingestion into local RAG
- Local memory system
- Image understanding with Qwen-VL
- Streaming live response mode
- Kawaii-style PySide6 desktop UI
- Windows easy-start scripts

---

## Project Positioning

LocalAI_V3 is not just a chatbot. It is designed as a local AI tutor that can:

- listen to students
- answer questions
- use personal study materials
- remember learning context locally
- support spoken practice
- analyse images and screenshots
- ingest web knowledge into a local knowledge base

The long-term goal is to explore how privacy-preserving local AI systems can support personalised learning.

---

## Quick Start on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/xiaotianhuang611-eng/LocalAI-V3.git
cd LocalAI-V3
```

### 2. Run setup

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

This will:

- create a Python virtual environment
- install dependencies from `requirements.txt`
- run a system check
- report missing models or runtime files

### 3. Add model files

Model files are not included in this repository.

Place them like this:

```text
models/
  google_gemma-4-E4B-it-Q5_K_M.gguf

models/qwen_vl/
  Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
  mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
```

See:

```text
docs/MODEL_SETUP.md
```

### 4. Add voice reference file

For XTTS voice output, add:

```text
data/reference.wav
```

Recommended:

- WAV format
- 7 to 12 seconds
- one speaker
- clear voice
- low background noise

### 5. Build RAG index

```powershell
.\.venv\Scripts\python.exe .\tools\build_rag_index.py
```

### 6. Start the app

```powershell
.\run_localai.bat
```

Alternative:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

---

## System Check

To check whether your environment is ready:

```powershell
.\.venv\Scripts\python.exe .\tools\check_system.py
```

The system check reports:

- Python version
- project files
- required directories
- model files
- voice reference file
- RAG index
- Python dependencies
- CUDA / GPU status

---

## Hardware Requirements

### Recommended

- Windows 10 / Windows 11
- NVIDIA RTX GPU with around 8GB VRAM or above
- Python 3.10 or 3.11
- 16GB RAM or above
- 20GB free disk space or above

### Minimum

- NVIDIA RTX GPU with around 6GB VRAM
- Lite settings may be required
- Vision and long voice generation may be limited

### Not Recommended

- CPU-only machines for the full voice assistant pipeline
- Non-NVIDIA GPU systems
- Very low VRAM laptops

---

## Repository Structure

```text
LocalAI_V3/
  main.py
  config.py
  requirements.txt
  setup_windows.ps1
  run_localai.bat
  QUICKSTART.md

  core/
    gemma_chat.py
    qwen_vision.py
    model_runtime.py
    xtts_fast.py
    asr.py
    rag_store.py
    memory_store.py
    web_knowledge.py
    sentence_buffer.py
    tts_queue.py

  ui/
    main_window.py

  tools/
    check_system.py
    build_rag_index.py
    web_search_to_rag.py

  data/
    knowledge/

  docs/
    MODEL_SETUP.md
```

---

## What Is Not Included

This repository intentionally excludes large, private, or generated files:

```text
models/
.venv/
data/memory/
data/rag/
data/temp/
data/reference.wav
*.gguf
*.wav
```

Reasons:

- model files are too large for GitHub
- model licenses may differ from the project license
- personal memory should remain private
- voice samples should remain private
- RAG indexes are generated locally

---

## Documentation

Start here:

```text
QUICKSTART.md
```

Model setup:

```text
docs/MODEL_SETUP.md
```

System check:

```powershell
.\.venv\Scripts\python.exe .\tools\check_system.py
```

---

## Development Status

Current version direction:

```text
V4.9 Easy Start Pack
```

Completed:

- GitHub open-source repository
- Windows setup script
- one-click start script
- system check script
- quick start guide
- model setup guide

Planned:

- English Speaking Coach Mode
- Math Teacher Mode
- Formula Display
- Better Lite Mode
- Desktop release packaging

---

## Privacy Design

LocalAI_V3 follows a local-first design:

- user documents stay local
- memory data stays local
- voice samples stay local
- generated RAG indexes stay local
- models run on the user's own machine

This makes the project suitable for exploring privacy-preserving AI learning assistants.

---

## License

This project is released under the MIT License.

See:

```text
LICENSE
```

---

## Disclaimer

This project is an educational and research prototype. It is not a commercial medical, legal, or safety-critical system.

Users are responsible for downloading models according to the relevant model licenses.
