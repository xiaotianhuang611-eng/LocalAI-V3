from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def missing(message: str) -> None:
    print(f"[MISSING] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def check_python() -> None:
    print("\n=== Python ===")
    version = sys.version.split()[0]
    info(f"Python version: {version}")
    info(f"Python executable: {sys.executable}")

    major, minor = sys.version_info[:2]

    if major == 3 and minor >= 10:
        ok("Python version is suitable.")
    else:
        warn("Python 3.10 or newer is recommended.")


def check_project_files() -> None:
    print("\n=== Project Files ===")

    required_files = [
        "main.py",
        "config.py",
        "requirements.txt",
        "core/model_runtime.py",
        "core/gemma_chat.py",
        "core/xtts_fast.py",
        "core/asr.py",
        "core/rag_store.py",
        "core/web_knowledge.py",
        "ui/main_window.py",
        "tools/build_rag_index.py",
        "tools/web_search_to_rag.py",
    ]

    for relative_path in required_files:
        path = ROOT_DIR / relative_path

        if path.exists():
            ok(f"Found {relative_path}")
        else:
            missing(f"Missing {relative_path}")


def check_directories() -> None:
    print("\n=== Directories ===")

    required_dirs = [
        "core",
        "ui",
        "tools",
        "data",
        "data/knowledge",
        "models",
    ]

    for relative_path in required_dirs:
        path = ROOT_DIR / relative_path

        if path.exists():
            ok(f"Found {relative_path}/")
        else:
            warn(f"Missing {relative_path}/")


def check_models() -> None:
    print("\n=== Model Files ===")

    model_paths = [
        "models/google_gemma-4-E4B-it-Q5_K_M.gguf",
        "models/qwen_vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "models/qwen_vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
    ]

    for relative_path in model_paths:
        path = ROOT_DIR / relative_path

        if path.exists():
            size_gb = path.stat().st_size / (1024 ** 3)
            ok(f"Found {relative_path} ({size_gb:.2f} GB)")
        else:
            missing(f"Missing {relative_path}")

    print("")
    info("Model files are not included in GitHub.")
    info("Users must download models manually and place them in the paths above.")


def check_runtime_data() -> None:
    print("\n=== Runtime Data ===")

    reference_path = ROOT_DIR / "data" / "reference.wav"

    if reference_path.exists():
        ok("Found data/reference.wav")
    else:
        warn("Missing data/reference.wav")
        info("XTTS voice cloning needs a clean reference.wav file.")

    rag_index_path = ROOT_DIR / "data" / "rag" / "rag_index.pkl"

    if rag_index_path.exists():
        ok("Found data/rag/rag_index.pkl")
    else:
        warn("RAG index not found.")
        info("Run: .\\.venv\\Scripts\\python.exe .\\tools\\build_rag_index.py")

    knowledge_dir = ROOT_DIR / "data" / "knowledge"

    if knowledge_dir.exists():
        files = [p for p in knowledge_dir.rglob("*") if p.is_file()]
        ok(f"Knowledge folder exists with {len(files)} file(s).")
    else:
        warn("Knowledge folder not found.")


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_dependencies() -> None:
    print("\n=== Python Dependencies ===")

    modules = [
        ("PySide6", "PySide6"),
        ("llama_cpp", "llama-cpp-python"),
        ("torch", "torch"),
        ("TTS", "TTS"),
        ("faster_whisper", "faster-whisper"),
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
    ]

    for module_name, package_name in modules:
        if module_available(module_name):
            ok(f"{package_name} is installed.")
        else:
            missing(f"{package_name} is not installed.")


def check_cuda() -> None:
    print("\n=== CUDA / GPU ===")

    if not module_available("torch"):
        warn("torch is not installed, so CUDA cannot be checked.")
        return

    try:
        import torch

        info(f"torch version: {torch.__version__}")

        if torch.cuda.is_available():
            ok("CUDA is available.")
            device_count = torch.cuda.device_count()
            info(f"CUDA device count: {device_count}")

            for index in range(device_count):
                name = torch.cuda.get_device_name(index)
                props = torch.cuda.get_device_properties(index)
                vram_gb = props.total_memory / (1024 ** 3)
                info(f"GPU {index}: {name} | VRAM: {vram_gb:.2f} GB")

                if vram_gb >= 7.5:
                    ok("This GPU is suitable for Standard Mode.")
                elif vram_gb >= 6:
                    warn("This GPU can run Minimum Mode. Vision and long XTTS may be limited.")
                elif vram_gb >= 4:
                    warn("This GPU should use Lite Mode. Disable vision and heavy settings.")
                else:
                    warn("VRAM is very limited for this project.")
        else:
            warn("CUDA is not available.")
            info("CPU-only mode is not recommended for the full voice assistant pipeline.")

    except Exception as exc:
        warn(f"CUDA check failed: {exc}")


def print_next_steps() -> None:
    print("\n=== Suggested Next Steps ===")
    print("1. If dependencies are missing, run:")
    print("   .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    print("")
    print("2. If model files are missing, download them and place them in:")
    print("   models/")
    print("   models/qwen_vl/")
    print("")
    print("3. If reference.wav is missing, prepare a clean 7-12 second WAV file:")
    print("   data/reference.wav")
    print("")
    print("4. If RAG index is missing, run:")
    print("   .\\.venv\\Scripts\\python.exe .\\tools\\build_rag_index.py")
    print("")
    print("5. To start LocalAI_V3, run:")
    print("   .\\.venv\\Scripts\\python.exe .\\main.py")


def main() -> int:
    print("LocalAI_V3 System Check")
    print("=======================")
    info(f"Project root: {ROOT_DIR}")
    info(f"Operating system: {platform.platform()}")

    check_python()
    check_project_files()
    check_directories()
    check_models()
    check_runtime_data()
    check_dependencies()
    check_cuda()
    print_next_steps()

    print("\nSystem check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
