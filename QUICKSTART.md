# LocalAI_V3 Quick Start

LocalAI_V3 is a local-first, privacy-preserving personalised AI tutor for students. It supports local LLM chat, voice interaction, RAG, memory, web knowledge ingestion, and vision understanding.

This guide explains how to run the project on Windows.

---

## 1. Hardware Requirements

### Recommended

- Windows 10 / Windows 11
- NVIDIA RTX GPU with 8GB VRAM or above
- Python 3.10 or 3.11
- 16GB RAM or above
- At least 20GB free disk space

### Minimum

- NVIDIA RTX GPU with 6GB VRAM
- Some features may need Lite Mode
- Vision and long voice generation may be limited

### Not Recommended

- CPU-only machines for the full voice assistant pipeline
- Non-NVIDIA GPU environments
- Very low VRAM laptops

---

## 2. Clone the Repository

```powershell
git clone https://github.com/xiaotianhuang611-eng/LocalAI-V3.git
cd LocalAI-V3
```

---

## 3. Run Windows Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

This script will:

- create `.venv`
- upgrade pip
- install Python dependencies
- run the system check script

If you only want to check the system without installing dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipInstall
```

---

## 4. Download Model Files

Model files are not included in this GitHub repository because they are large and may have separate licenses.

You must place the model files manually.

Required structure:

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

---

## 5. Add Voice Reference File

For XTTS voice output, prepare a clean WAV file:

```text
data/reference.wav
```

Recommended voice sample:

- 7 to 12 seconds
- clear speech
- no background music
- no heavy noise
- one speaker only

If `data/reference.wav` is missing, some voice features may not work.

---

## 6. Build RAG Index

LocalAI_V3 uses local knowledge files from:

```text
data/knowledge/
```

To build the RAG index:

```powershell
.\.venv\Scripts\python.exe .\tools\build_rag_index.py
```

The generated index will be stored in:

```text
data/rag/
```

This folder is ignored by Git because it is generated locally.

---

## 7. Start the App

Recommended:

```powershell
.\run_localai.bat
```

Alternative:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

---

## 8. Run System Check

If something does not work, run:

```powershell
.\.venv\Scripts\python.exe .\tools\check_system.py
```

The system check will report:

- Python version
- project files
- required folders
- model files
- reference voice file
- RAG index
- Python dependencies
- CUDA / GPU status

---

## 9. Typical First-Time Workflow

```powershell
git clone https://github.com/xiaotianhuang611-eng/LocalAI-V3.git
cd LocalAI-V3
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

Then manually add:

```text
models/
data/reference.wav
```

Then build RAG:

```powershell
.\.venv\Scripts\python.exe .\tools\build_rag_index.py
```

Then start:

```powershell
.\run_localai.bat
```

---

## 10. Important Notes

This project is designed for local-first AI experimentation and student learning.

The repository does not include:

- GGUF model files
- Python virtual environment
- private memory data
- generated RAG index
- personal voice samples
- temporary audio files

These files must stay local for privacy, licensing, and file size reasons.
