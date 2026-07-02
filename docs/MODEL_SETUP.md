# Model Setup Guide

LocalAI_V3 requires local GGUF model files. These files are not included in the GitHub repository because they are large and may have separate model licenses.

You need to download the required models manually and place them in the correct folders.

---

## 1. Required Model Folder Structure

Your local project should look like this:

```text
LocalAI_V3/
  models/
    google_gemma-4-E4B-it-Q5_K_M.gguf

    qwen_vl/
      Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
      mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
```

The file names must match the paths expected by the project configuration.

---

## 2. Main Language Model

Required file:

```text
models/google_gemma-4-E4B-it-Q5_K_M.gguf
```

Purpose:

- local text chat
- tutor response generation
- RAG-based answer generation
- memory-aware conversation

Recommended quantisation:

```text
Q5_K_M
```

This is a practical balance between quality, speed, and VRAM usage.

---

## 3. Vision Model

Required files:

```text
models/qwen_vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
models/qwen_vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
```

Purpose:

- local image understanding
- screenshot analysis
- visual question answering

The `mmproj` file is required for multimodal projection. Without it, the vision model will not work correctly.

---

## 4. Voice Model Dependencies

XTTS is installed through Python dependencies. The project also requires a user-provided reference voice file:

```text
data/reference.wav
```

Recommended format:

- WAV
- 7 to 12 seconds
- clear speech
- one speaker
- low background noise

---

## 5. Why Models Are Not Included

The GitHub repository intentionally excludes:

```text
models/
*.gguf
*.bin
*.safetensors
*.pt
*.pth
```

Reasons:

1. model files are too large for normal GitHub repositories
2. model licenses may be different from the project license
3. users may want to choose different quantised models
4. keeping models local supports the privacy-first design

---

## 6. Recommended Hardware

### Standard Mode

Recommended:

```text
NVIDIA RTX GPU with around 8GB VRAM or above
```

Examples:

```text
RTX 3060 12GB
RTX 4060 8GB
RTX 5060 Laptop GPU 8GB
RTX 4070 Laptop GPU 8GB
```

### Minimum Mode

Possible but limited:

```text
RTX GPU with around 6GB VRAM
```

Limitations:

- shorter context
- slower generation
- vision model may need to be disabled
- XTTS may need shorter responses

### Lite Mode

For 4GB VRAM GPUs:

```text
RTX 3050 Laptop GPU 4GB
```

Suggested limitations:

- disable vision
- use shorter responses
- avoid long XTTS output
- reduce RAG context size

---

## 7. Verify Model Installation

After placing the model files, run:

```powershell
.\.venv\Scripts\python.exe .\tools\check_system.py
```

Expected output should include:

```text
[OK] Found models/google_gemma-4-E4B-it-Q5_K_M.gguf
[OK] Found models/qwen_vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
[OK] Found models/qwen_vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
```

If any model shows `[MISSING]`, check the file name and folder path carefully.

---

## 8. Troubleshooting

### Problem: model file missing

Check:

```text
models/
models/qwen_vl/
```

Make sure the file names are exactly the same as required.

### Problem: CUDA not available

Run:

```powershell
.\.venv\Scripts\python.exe .\tools\check_system.py
```

Check whether torch can detect CUDA.

### Problem: app starts but voice does not work

Check:

```text
data/reference.wav
```

Make sure the file exists and contains a clean voice sample.

### Problem: RAG has no useful answer

Add documents to:

```text
data/knowledge/
```

Then rebuild:

```powershell
.\.venv\Scripts\python.exe .\tools\build_rag_index.py
```
