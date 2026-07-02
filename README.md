# LocalAI_V3

## A Local-First Privacy-Preserving Personalised AI Tutor

**LocalAI_V3** is a local-first, privacy-preserving personalised AI tutor designed for students, English learners, and users who need private AI-assisted learning on their own device.

It supports real-time voice interaction, English speaking practice, mathematical learning, document-based RAG, local memory, image understanding, web knowledge ingestion, and local GPU inference.

> **Speak · Learn · Remember · Grow**

---

## 1. Project Overview

LocalAI_V3 is not a cloud chatbot. It is a desktop AI tutor that runs the main AI pipeline locally on the user's own computer.

The system integrates multiple local AI modules:

- **VAD** for automatic speech start/stop detection
- **faster-whisper ASR** for speech recognition
- **Gemma GGUF** for local language reasoning
- **XTTS** for voice output and voice cloning
- **Qwen-VL** for image understanding
- **Local Memory** for personalised long-term context
- **Local RAG** for document-grounded answering
- **Web Knowledge Ingestion** for adding public web content into the local RAG knowledge base
- **PySide6 Desktop UI** for user interaction

The project is designed as a local AI tutor for spoken English practice, presentation rehearsal, academic Q&A, mathematical learning, personalised study support, and local document-based question answering.

---

## 2. Key Features

### Real-Time Voice Chat

LocalAI_V3 supports live voice interaction using:

```text
VAD -> ASR -> Gemma -> XTTS
```

The system listens to the user, transcribes speech into text, generates a response locally, and speaks back using XTTS.

### Streaming Live Mode

The latest version supports sentence-level streaming speech playback.

Instead of waiting for the full LLM response before speaking, the system can:

```text
Generate response -> detect complete sentence -> send to XTTS -> speak sentence by sentence
```

This reduces perceived latency and makes the interaction feel closer to a real AI tutor.

### English Speaking Coach

The system can be used as a speaking practice partner for students and English learners. It can support daily conversation practice, presentation practice, classroom discussion practice, natural expression correction, and personalised feedback.

### Math Teacher Mode

The project is designed to support mathematical learning through step-by-step explanation, formula-based reasoning, final answer generation, scoring, correction, and personalised advice.

### Local RAG

Users can place their own documents into the local knowledge folder and build a RAG index:

```text
data/knowledge/
```

The assistant can then answer questions based on the user's own study materials, project notes, or documents.

### Web Knowledge Ingestion

The system includes a web knowledge module that can search public web pages, crawl a direct URL, extract readable content, save cleaned text as local Markdown files, and add the content into the local RAG knowledge base.

### Local Memory

The memory module stores stable facts, learning context, project information, and user preferences locally.

### Image Understanding

The system can use Qwen-VL to analyse images. To manage limited GPU memory, the runtime safely unloads and reloads models when switching into image mode.

---

## 3. Target Users

LocalAI_V3 is designed for:

- international students
- university students
- high school students
- English learners
- students preparing presentations or viva-style explanations
- users who want private AI tutoring on their own device
- users with NVIDIA RTX GPUs who want local AI inference

---

## 4. Why Local-First?

Many AI assistants rely on cloud APIs. LocalAI_V3 takes a different approach.

The main assistant pipeline runs locally, which means:

- voice data stays on the user's device
- documents stay on the user's device
- memory stays on the user's device
- RAG knowledge files remain local
- the user controls the models and data
- no cloud API is required for the core assistant workflow

This makes the system suitable for privacy-sensitive learning and personal study workflows.

---

## 5. System Architecture

High-level architecture:

```text
User
 ↓
PySide6 Desktop UI
 ↓
ModelRuntime
 ↓
Core AI Modules
 ├── Energy VAD
 ├── faster-whisper ASR
 ├── Gemma local LLM
 ├── XTTS voice output
 ├── Qwen-VL image understanding
 ├── Local Memory
 ├── Local RAG
 └── Web Knowledge Ingestion
```

Text interaction:

```text
User text
 ↓
PromptBuilder
 ↓
Gemma GGUF via llama-cpp-python
 ↓
English response
 ↓
XTTS voice output
```

Voice interaction:

```text
Microphone
 ↓
VAD auto-stop recording
 ↓
faster-whisper ASR
 ↓
Gemma response generation
 ↓
XTTS speech synthesis
```

Streaming Live Mode:

```text
ASR text
 ↓
Gemma streaming generation
 ↓
SentenceBuffer
 ↓
TTSQueue
 ↓
XTTS sentence-level playback
```

Web-Augmented RAG:

```text
Search query / URL
 ↓
Web crawler
 ↓
Cleaned Markdown document
 ↓
data/knowledge/web_crawled/
 ↓
RAG index rebuild
 ↓
Gemma answers with retrieved context
```

---

## 6. Hardware Requirements

### Minimum Target

```text
NVIDIA RTX 2060 6GB VRAM or above
```

This supports reduced configuration and core local inference.

### Recommended

```text
RTX 3060 12GB
RTX 4060 8GB
RTX 5060 8GB
or higher
```

### Best Experience

```text
12GB VRAM or higher
```

RTX 3050 4GB may run a lightweight configuration, but full multimodal use is not recommended on 4GB VRAM.

---

## 7. Software Requirements

Recommended environment:

```text
Windows 10 / Windows 11
Python virtual environment
NVIDIA GPU
CUDA-compatible PyTorch
llama-cpp-python with CUDA support
PySide6
faster-whisper
Coqui TTS / XTTS
```

---

## 8. Project Structure

```text
LocalAI_V3/
  main.py
  config.py
  README.md
  LICENSE
  requirements.txt

  core/
    model_runtime.py
    gemma_chat.py
    prompt_builder.py
    xtts_fast.py
    asr.py
    vad_recorder.py
    qwen_vision.py
    memory_store.py
    rag_store.py
    web_knowledge.py
    sentence_buffer.py
    tts_queue.py
    persona.py
    emotion_engine.py

  ui/
    main_window.py

  tools/
    build_rag_index.py
    test_rag_search.py
    web_search_to_rag.py
    benchmark_v4_2.py

  data/
    knowledge/
      README.md
      sample_project_note.md

  docs/
    index.html
    style.css
```

---

## 9. Model Files

Model weights are **not included** in this repository.

Do not upload:

```text
models/
*.gguf
*.safetensors
*.bin
*.pt
*.pth
*.onnx
```

Users should download compatible models separately and place them in the required local folders.

Example expected model structure:

```text
models/
  google_gemma-4-E4B-it-Q5_K_M.gguf

  qwen_vl/
    Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
    mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
```

Model licenses must be checked and followed separately.

---

## 10. Quick Start

Run:

```powershell
cd LocalAI_V3
.\.venv\Scripts\python.exe main.py
```

If running from the original project path:

```powershell
cd C:\Users\111\Desktop\LocalAI_V3
.\.venv\Scripts\python.exe main.py
```

---

## 11. Build the RAG Index

Place local study materials or project notes into:

```text
data/knowledge/
```

Then run:

```powershell
.\.venv\Scripts\python.exe .\tools\build_rag_index.py
```

After building the index, enable the **RAG** checkbox in the UI.

---

## 12. Web Knowledge Ingestion

### Search and add web content into RAG

```powershell
.\.venv\Scripts\python.exe .\tools\web_search_to_rag.py --query "local AI speech assistant RAG" --max-results 5 --rebuild
```

### Crawl a direct URL

```powershell
.\.venv\Scripts\python.exe .\tools\web_search_to_rag.py --url "https://example.com/article" --rebuild
```

Crawled web pages are saved to:

```text
data/knowledge/web_crawled/
```

The crawler is intended only for publicly accessible pages. Users should respect website terms, copyright restrictions, robots.txt, and rate limits.

---

## 13. Privacy Notice

The repository should not include private user data.

Do not upload:

```text
data/reference.wav
data/output.wav
data/temp/
data/memory/
data/rag/
private documents
model weights
```

This project is designed to keep user data local.

---

## 14. GitHub Pages Website

A static project landing page can be placed in:

```text
docs/
  index.html
  style.css
```

GitHub Pages can be enabled from:

```text
Repository Settings -> Pages -> Deploy from branch -> main -> /docs
```

The website introduces the project and links users to GitHub for download.

---

## 15. Current Development Status

LocalAI_V3 is currently a functional local AI software prototype.

It includes:

- desktop UI
- local LLM inference
- voice input and output
- streaming live mode
- local RAG
- local memory
- image understanding
- web knowledge ingestion
- benchmark tooling
- open-source release preparation

It is best described as:

```text
Research Prototype
Engineering Prototype
Local Personalised AI Tutor
```

It is not yet a production-level commercial SaaS platform.

---

## 16. Roadmap

```text
V4.7 Streaming Live Mode
- sentence-level streaming response
- TTS queue
- lower perceived latency

V4.8 Web-Augmented RAG
- web search
- direct URL crawl
- local knowledge ingestion

V4.9 Tutor Modes
- English Speaking Coach
- Math Teacher Mode
- scoring and feedback

V5.0 Desktop Tutor Prototype
- software stability layer
- launcher
- better documentation
- open-source release
```

---

## 17. Academic Relevance

This project can support research in:

- edge AI
- privacy-preserving AI
- multimodal AI assistants
- local RAG
- speech-to-speech interaction
- AI-assisted learning
- personalised tutoring systems

A suitable academic description:

> LocalAI_V3 demonstrates how a privacy-preserving multimodal AI tutor can be built using local ASR, LLM inference, TTS, RAG, memory, image understanding, and web-augmented knowledge ingestion on consumer hardware.

---

## 18. Limitations

Current limitations include:

- requires NVIDIA GPU for best experience
- model setup is manual
- XTTS can be slow for long responses
- web crawling is not reliable for all websites
- Qwen-VL requires careful GPU memory management
- full real-time interruption is not yet implemented
- not designed for high-concurrency cloud deployment

---

## 19. License

This project is released under the MIT License.

Model weights are not included and may have separate licenses.

---

## 20. Disclaimer

This project is intended for educational, research, and personal learning use. Users are responsible for complying with model licenses, website terms of service, copyright rules, and local regulations.
