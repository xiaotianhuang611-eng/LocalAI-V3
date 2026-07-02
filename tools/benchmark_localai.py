from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.model_runtime import ModelRuntime


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def measure(name: str, func):
    print(f"\n[BENCH] {name}...")
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    print(f"[BENCH] {name}: {elapsed:.3f}s")
    return elapsed, result


def main() -> int:
    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    benchmark = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(ROOT_DIR),
        "results": {},
    }

    runtime = ModelRuntime(root_dir=ROOT_DIR)

    load_time, _ = measure("Load Gemma + XTTS", runtime.load)
    benchmark["results"]["load_gemma_xtts_seconds"] = load_time

    asr_load_time, _ = measure("Load ASR", runtime.load_asr)
    benchmark["results"]["load_asr_seconds"] = asr_load_time

    prompts = [
        "What can you do for me?",
        "Explain your main features in one short paragraph.",
        "Give me one useful study tip for today.",
    ]

    gemma_times = []
    gemma_outputs = []

    for i, prompt in enumerate(prompts, start=1):
        elapsed, response = measure(
            f"Gemma response {i}",
            lambda p=prompt: runtime.ask(p),
        )
        gemma_times.append(elapsed)
        gemma_outputs.append(
            {
                "prompt": prompt,
                "response": response,
                "seconds": elapsed,
                "response_chars": len(response),
                "chars_per_second": len(response) / elapsed if elapsed > 0 else 0,
            }
        )

    benchmark["results"]["gemma"] = {
        "runs": gemma_outputs,
        "avg_seconds": statistics.mean(gemma_times),
        "min_seconds": min(gemma_times),
        "max_seconds": max(gemma_times),
    }

    tts_texts = [
        "Hello, I am ready.",
        "This is a local voice assistant running on your computer.",
    ]

    tts_times = []

    for i, text in enumerate(tts_texts, start=1):
        elapsed, _ = measure(
            f"XTTS speak {i}",
            lambda t=text: runtime.speak(t),
        )
        tts_times.append(elapsed)

    benchmark["results"]["xtts_total_wall_time"] = {
        "note": "This includes synthesis plus audio playback waiting time.",
        "avg_seconds": statistics.mean(tts_times),
        "min_seconds": min(tts_times),
        "max_seconds": max(tts_times),
    }

    end_to_end_times = []
    end_to_end_prompts = [
        "Say hello in one sentence.",
        "Tell me one reason why local AI is useful.",
    ]

    for i, prompt in enumerate(end_to_end_prompts, start=1):
        elapsed, response = measure(
            f"End-to-end Ask + Speak {i}",
            lambda p=prompt: runtime.ask_and_speak(p),
        )
        end_to_end_times.append(elapsed)

        benchmark["results"][f"end_to_end_{i}"] = {
            "prompt": prompt,
            "response": response,
            "seconds": elapsed,
        }

    benchmark["results"]["end_to_end_summary"] = {
        "avg_seconds": statistics.mean(end_to_end_times),
        "min_seconds": min(end_to_end_times),
        "max_seconds": max(end_to_end_times),
    }

    output_json = logs_dir / f"benchmark_{now_stamp()}.json"
    output_txt = logs_dir / f"benchmark_{now_stamp()}.txt"

    output_json.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("LocalAI V3 Benchmark Report")
    lines.append("=" * 40)
    lines.append(f"Created at: {benchmark['created_at']}")
    lines.append("")
    lines.append(f"Load Gemma + XTTS: {load_time:.3f}s")
    lines.append(f"Load ASR: {asr_load_time:.3f}s")
    lines.append("")
    lines.append(
        f"Gemma avg: {benchmark['results']['gemma']['avg_seconds']:.3f}s "
        f"| min: {benchmark['results']['gemma']['min_seconds']:.3f}s "
        f"| max: {benchmark['results']['gemma']['max_seconds']:.3f}s"
    )
    lines.append(
        f"XTTS avg wall time: {benchmark['results']['xtts_total_wall_time']['avg_seconds']:.3f}s "
        f"| min: {benchmark['results']['xtts_total_wall_time']['min_seconds']:.3f}s "
        f"| max: {benchmark['results']['xtts_total_wall_time']['max_seconds']:.3f}s"
    )
    lines.append(
        f"End-to-end avg: {benchmark['results']['end_to_end_summary']['avg_seconds']:.3f}s "
        f"| min: {benchmark['results']['end_to_end_summary']['min_seconds']:.3f}s "
        f"| max: {benchmark['results']['end_to_end_summary']['max_seconds']:.3f}s"
    )
    lines.append("")
    lines.append(f"JSON saved: {output_json}")
    lines.append(f"TXT saved: {output_txt}")

    output_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())