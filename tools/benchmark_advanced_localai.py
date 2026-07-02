from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.model_runtime import ModelRuntime


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def perf_seconds() -> float:
    return time.perf_counter()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)

    if low == high:
        return values[int(rank)]

    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def summarize_latency(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if v is not None]

    if not clean:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "std": 0.0,
        }

    return {
        "count": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "std": statistics.stdev(clean) if len(clean) >= 2 else 0.0,
    }


def measure(name: str, func):
    print(f"\n[BENCH] {name}")
    start = perf_seconds()

    ok = True
    error = ""
    result = None

    try:
        result = func()
    except Exception as exc:
        ok = False
        error = str(exc)

    seconds = perf_seconds() - start

    if ok:
        print(f"[OK] {name}: {seconds:.4f}s")
    else:
        print(f"[ERROR] {name}: {seconds:.4f}s | {error}")

    return {
        "name": name,
        "ok": ok,
        "seconds": seconds,
        "error": error,
        "result": result,
    }


def get_memory_snapshot() -> dict[str, Any]:
    data: dict[str, Any] = {
        "rss_mb": None,
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
        "cuda_max_allocated_mb": None,
    }

    try:
        import psutil

        process = psutil.Process()
        data["rss_mb"] = process.memory_info().rss / 1024 / 1024
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            data["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            data["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            data["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass

    return data


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    fields: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def save_bar(plt, labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111)

    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)

    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_line(plt, labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111)

    x = list(range(1, len(values) + 1))
    ax.plot(x, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_scatter(
    plt,
    x_values: list[float],
    y_values: list[float],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111)

    ax.scatter(x_values, y_values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for x, y, label in zip(x_values, y_values, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_multi_line(
    plt,
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111)

    x = list(range(1, len(labels) + 1))

    for name, values in series.items():
        ax.plot(x, values, marker="o", label=name)

    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def benchmark_xtts_synthesis_only(runtime: ModelRuntime, text: str) -> dict[str, Any]:
    """
    Measures XTTS synthesis only, without audio playback.
    This gives cleaner numbers than runtime.speak(), because speak() includes sd.play + sd.wait.
    """
    if runtime.xtts.tts is None:
        runtime.xtts.load()

    if not runtime.xtts.reference_path.exists():
        raise FileNotFoundError(
            f"Reference voice not found: {runtime.xtts.reference_path}. "
            "Record Reference first before running advanced XTTS benchmark."
        )

    sanitize_start = perf_seconds()
    clean_text = runtime.xtts._sanitize_for_tts(text)
    sanitize_seconds = perf_seconds() - sanitize_start

    speaker_wav = runtime.xtts._prepare_temp_reference()

    synth_start = perf_seconds()

    try:
        wav = runtime.xtts.tts.tts(
            text=clean_text,
            speaker_wav=str(speaker_wav),
            language=runtime.xtts.language,
            enable_text_splitting=False,
        )
    except TypeError:
        wav = runtime.xtts.tts.tts(
            text=clean_text,
            speaker_wav=str(speaker_wav),
            language=runtime.xtts.language,
            split_sentences=False,
        )

    synth_seconds = perf_seconds() - synth_start

    audio = runtime.xtts._to_numpy_audio(wav)
    audio = runtime.xtts._safe_normalize(audio)

    audio_seconds = len(audio) / float(runtime.xtts.sample_rate)
    rtf = synth_seconds / audio_seconds if audio_seconds > 0 else 0.0

    chars_per_second = len(clean_text) / synth_seconds if synth_seconds > 0 else 0.0

    return {
        "raw_text": text,
        "clean_text": clean_text,
        "raw_chars": len(text),
        "clean_chars": len(clean_text),
        "sanitize_seconds": sanitize_seconds,
        "synthesis_seconds": synth_seconds,
        "audio_seconds": audio_seconds,
        "rtf": rtf,
        "tts_chars_per_second": chars_per_second,
    }


def run_advanced_benchmark() -> int:
    run_id = now_stamp()
    output_dir = ROOT_DIR / "logs" / f"benchmark_advanced_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LocalAI V3 Advanced Benchmark Suite")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    results: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(ROOT_DIR),
        "output_dir": str(output_dir),
        "load": {},
        "memory": {},
        "gemma_repeated_latency": [],
        "gemma_prompt_scaling": [],
        "xtts_synthesis": [],
        "xtts_wall_playback": [],
        "end_to_end_synthesis_only": [],
        "end_to_end_with_playback": [],
        "stability": [],
        "summary": {},
    }

    flat_rows: list[dict[str, Any]] = []

    runtime = ModelRuntime(root_dir=ROOT_DIR)

    results["memory"]["before_load"] = get_memory_snapshot()

    load_result = measure("Cold load Gemma + XTTS", runtime.load)
    results["load"]["gemma_xtts_seconds"] = load_result["seconds"]
    results["load"]["gemma_xtts_ok"] = load_result["ok"]

    flat_rows.append(
        {
            "category": "load",
            "name": "Gemma + XTTS",
            "seconds": load_result["seconds"],
            "ok": load_result["ok"],
            "error": load_result["error"],
        }
    )

    results["memory"]["after_gemma_xtts_load"] = get_memory_snapshot()

    asr_result = measure("Cold load ASR", runtime.load_asr)
    results["load"]["asr_seconds"] = asr_result["seconds"]
    results["load"]["asr_ok"] = asr_result["ok"]

    flat_rows.append(
        {
            "category": "load",
            "name": "ASR",
            "seconds": asr_result["seconds"],
            "ok": asr_result["ok"],
            "error": asr_result["error"],
        }
    )

    results["memory"]["after_asr_load"] = get_memory_snapshot()

    warmup_prompts = [
        "Say hello in one short sentence.",
        "Tell me one simple fact about local AI.",
    ]

    for i, prompt in enumerate(warmup_prompts, start=1):
        measure(f"Warm-up Gemma {i}", lambda p=prompt: runtime.ask(p))

    repeated_prompt = "Answer in one short sentence: what can you do?"

    for i in range(1, 11):
        measured = measure(
            f"Gemma repeated latency run {i}",
            lambda p=repeated_prompt: runtime.ask(p),
        )

        response = measured["result"] or ""

        row = {
            "run": i,
            "prompt": repeated_prompt,
            "seconds": measured["seconds"],
            "ok": measured["ok"],
            "error": measured["error"],
            "response": response,
            "response_chars": len(response),
            "chars_per_second": len(response) / measured["seconds"] if measured["seconds"] > 0 else 0.0,
        }

        results["gemma_repeated_latency"].append(row)

        flat_rows.append(
            {
                "category": "gemma_repeated_latency",
                "name": f"run_{i}",
                "seconds": row["seconds"],
                "ok": row["ok"],
                "error": row["error"],
                "response_chars": row["response_chars"],
                "chars_per_second": row["chars_per_second"],
            }
        )

    prompt_scaling_cases = [
        {
            "name": "tiny_prompt",
            "prompt": "Hi.",
        },
        {
            "name": "short_prompt",
            "prompt": "What can you do?",
        },
        {
            "name": "medium_prompt",
            "prompt": "Explain your main features as a local AI assistant in two short sentences.",
        },
        {
            "name": "long_prompt",
            "prompt": (
                "I am building a local AI voice assistant with ASR, Gemma, XTTS, and Qwen-VL. "
                "Explain how this system works, why it is private, and what the main limitations are. "
                "Keep the answer concise."
            ),
        },
        {
            "name": "symbol_prompt",
            "prompt": (
                "Explain this pipeline briefly: microphone -> ASR -> Gemma -> XTTS -> speaker. "
                "Also mention C:\\Users\\111\\Desktop\\LocalAI_V3."
            ),
        },
    ]

    for item in prompt_scaling_cases:
        name = item["name"]
        prompt = item["prompt"]

        measured = measure(
            f"Gemma prompt scaling {name}",
            lambda p=prompt: runtime.ask(p),
        )

        response = measured["result"] or ""

        row = {
            "name": name,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "seconds": measured["seconds"],
            "ok": measured["ok"],
            "error": measured["error"],
            "response": response,
            "response_chars": len(response),
            "chars_per_second": len(response) / measured["seconds"] if measured["seconds"] > 0 else 0.0,
        }

        results["gemma_prompt_scaling"].append(row)

        flat_rows.append(
            {
                "category": "gemma_prompt_scaling",
                "name": name,
                "prompt_chars": row["prompt_chars"],
                "seconds": row["seconds"],
                "ok": row["ok"],
                "error": row["error"],
                "response_chars": row["response_chars"],
                "chars_per_second": row["chars_per_second"],
            }
        )

    xtts_cases = [
        {
            "name": "tiny",
            "text": "Hello.",
        },
        {
            "name": "short",
            "text": "Hello, I am ready to help you.",
        },
        {
            "name": "medium",
            "text": "This is a local voice assistant running on your computer with private speech recognition and voice cloning.",
        },
        {
            "name": "long",
            "text": (
                "Local AI is useful because it protects privacy, reduces dependence on cloud APIs, "
                "and keeps the assistant available even when the network is unavailable. "
                "It can process speech, text, and images locally on your computer."
            ),
        },
        {
            "name": "symbol_heavy",
            "text": "**Sure!** The path is C:\\Users\\111\\Desktop\\LocalAI_V3 -> models/qwen_vl 😊.",
        },
        {
            "name": "url_email",
            "text": "Visit https://example.com or email test@example.com for more information.",
        },
    ]

    for item in xtts_cases:
        name = item["name"]
        text = item["text"]

        measured = measure(
            f"XTTS synthesis-only {name}",
            lambda t=text: benchmark_xtts_synthesis_only(runtime, t),
        )

        data = measured["result"] or {}

        row = {
            "name": name,
            "ok": measured["ok"],
            "error": measured["error"],
            "total_function_seconds": measured["seconds"],
            "raw_chars": data.get("raw_chars", len(text)),
            "clean_chars": data.get("clean_chars", 0),
            "sanitize_seconds": data.get("sanitize_seconds", 0.0),
            "synthesis_seconds": data.get("synthesis_seconds", 0.0),
            "audio_seconds": data.get("audio_seconds", 0.0),
            "rtf": data.get("rtf", 0.0),
            "tts_chars_per_second": data.get("tts_chars_per_second", 0.0),
            "clean_text_preview": str(data.get("clean_text", ""))[:200],
        }

        results["xtts_synthesis"].append(row)

        flat_rows.append(
            {
                "category": "xtts_synthesis",
                "name": name,
                "ok": row["ok"],
                "error": row["error"],
                "sanitize_seconds": row["sanitize_seconds"],
                "synthesis_seconds": row["synthesis_seconds"],
                "audio_seconds": row["audio_seconds"],
                "rtf": row["rtf"],
                "raw_chars": row["raw_chars"],
                "clean_chars": row["clean_chars"],
                "tts_chars_per_second": row["tts_chars_per_second"],
            }
        )

    playback_cases = [
        {
            "name": "playback_short",
            "text": "This benchmark includes synthesis plus audio playback.",
        },
        {
            "name": "playback_symbol",
            "text": "ASR -> Gemma -> XTTS, with emoji 😊 and markdown **removed**.",
        },
    ]

    for item in playback_cases:
        name = item["name"]
        text = item["text"]

        measured = measure(
            f"XTTS wall playback {name}",
            lambda t=text: runtime.speak(t),
        )

        row = {
            "name": name,
            "text": text,
            "text_chars": len(text),
            "seconds": measured["seconds"],
            "ok": measured["ok"],
            "error": measured["error"],
        }

        results["xtts_wall_playback"].append(row)

        flat_rows.append(
            {
                "category": "xtts_wall_playback",
                "name": name,
                "seconds": row["seconds"],
                "text_chars": row["text_chars"],
                "ok": row["ok"],
                "error": row["error"],
            }
        )

    e2e_cases = [
        {
            "name": "e2e_short",
            "prompt": "Say hello in one short sentence.",
        },
        {
            "name": "e2e_privacy",
            "prompt": "Give one short reason why local AI protects privacy.",
        },
        {
            "name": "e2e_study",
            "prompt": "Give me one short English speaking practice task.",
        },
    ]

    for item in e2e_cases:
        name = item["name"]
        prompt = item["prompt"]

        print(f"\n[BENCH] End-to-end synthesis-only {name}")

        total_start = perf_seconds()

        ask = measure(
            f"E2E Gemma part {name}",
            lambda p=prompt: runtime.ask(p),
        )

        response = ask["result"] or ""

        tts = measure(
            f"E2E XTTS synthesis part {name}",
            lambda r=response: benchmark_xtts_synthesis_only(runtime, r),
        )

        total_seconds = perf_seconds() - total_start

        tts_data = tts["result"] or {}

        row = {
            "name": name,
            "prompt": prompt,
            "ok": ask["ok"] and tts["ok"],
            "error": ask["error"] or tts["error"],
            "gemma_seconds": ask["seconds"],
            "xtts_synthesis_seconds": tts_data.get("synthesis_seconds", 0.0),
            "xtts_sanitize_seconds": tts_data.get("sanitize_seconds", 0.0),
            "audio_seconds": tts_data.get("audio_seconds", 0.0),
            "rtf": tts_data.get("rtf", 0.0),
            "total_seconds": total_seconds,
            "response": response,
            "response_chars": len(response),
        }

        results["end_to_end_synthesis_only"].append(row)

        flat_rows.append(
            {
                "category": "end_to_end_synthesis_only",
                "name": name,
                "ok": row["ok"],
                "error": row["error"],
                "gemma_seconds": row["gemma_seconds"],
                "xtts_synthesis_seconds": row["xtts_synthesis_seconds"],
                "xtts_sanitize_seconds": row["xtts_sanitize_seconds"],
                "audio_seconds": row["audio_seconds"],
                "rtf": row["rtf"],
                "total_seconds": row["total_seconds"],
                "response_chars": row["response_chars"],
            }
        )

    playback_e2e_cases = [
        {
            "name": "e2e_playback_short",
            "prompt": "Say one sentence about your speed.",
        },
        {
            "name": "e2e_playback_local",
            "prompt": "Say one sentence about local AI.",
        },
    ]

    for item in playback_e2e_cases:
        name = item["name"]
        prompt = item["prompt"]

        measured = measure(
            f"End-to-end with playback {name}",
            lambda p=prompt: runtime.ask_and_speak(p),
        )

        response = measured["result"] or ""

        row = {
            "name": name,
            "prompt": prompt,
            "response": response,
            "response_chars": len(response),
            "seconds": measured["seconds"],
            "ok": measured["ok"],
            "error": measured["error"],
        }

        results["end_to_end_with_playback"].append(row)

        flat_rows.append(
            {
                "category": "end_to_end_with_playback",
                "name": name,
                "seconds": row["seconds"],
                "ok": row["ok"],
                "error": row["error"],
                "response_chars": row["response_chars"],
            }
        )

    stability_cases = [
        {
            "name": "markdown_code",
            "text": "Here is code: ```python\nprint('hello')\n``` Done.",
        },
        {
            "name": "url_email",
            "text": "Visit https://openai.com or email test@example.com.",
        },
        {
            "name": "windows_path",
            "text": "The folder is C:\\Users\\111\\Desktop\\LocalAI_V3\\models\\qwen_vl.",
        },
        {
            "name": "math_symbols",
            "text": "A+B=C, x >= 10, y != 3, input -> output.",
        },
        {
            "name": "emoji_heavy",
            "text": "Good job! 😊😊😊🔥🔥🔥✅✅✅",
        },
        {
            "name": "mixed",
            "text": "**Result:** ASR -> LLM -> TTS. URL: https://test.com. Path: C:\\test\\file.txt 😊",
        },
    ]

    for item in stability_cases:
        name = item["name"]
        text = item["text"]

        measured = measure(
            f"Stability sanitizer + XTTS synthesis {name}",
            lambda t=text: benchmark_xtts_synthesis_only(runtime, t),
        )

        data = measured["result"] or {}

        row = {
            "name": name,
            "ok": measured["ok"],
            "error": measured["error"],
            "seconds": measured["seconds"],
            "raw_chars": data.get("raw_chars", len(text)),
            "clean_chars": data.get("clean_chars", 0),
            "sanitize_seconds": data.get("sanitize_seconds", 0.0),
            "synthesis_seconds": data.get("synthesis_seconds", 0.0),
            "rtf": data.get("rtf", 0.0),
            "clean_text_preview": str(data.get("clean_text", ""))[:200],
        }

        results["stability"].append(row)

        flat_rows.append(
            {
                "category": "stability",
                "name": name,
                "ok": row["ok"],
                "error": row["error"],
                "seconds": row["seconds"],
                "sanitize_seconds": row["sanitize_seconds"],
                "synthesis_seconds": row["synthesis_seconds"],
                "rtf": row["rtf"],
            }
        )

    results["memory"]["after_all"] = get_memory_snapshot()

    gemma_repeated = [x["seconds"] for x in results["gemma_repeated_latency"] if x["ok"]]
    gemma_scaling = [x["seconds"] for x in results["gemma_prompt_scaling"] if x["ok"]]
    xtts_synth = [x["synthesis_seconds"] for x in results["xtts_synthesis"] if x["ok"]]
    xtts_rtf = [x["rtf"] for x in results["xtts_synthesis"] if x["ok"]]
    e2e_synth = [x["total_seconds"] for x in results["end_to_end_synthesis_only"] if x["ok"]]
    e2e_playback = [x["seconds"] for x in results["end_to_end_with_playback"] if x["ok"]]
    sanitizer = [x["sanitize_seconds"] for x in results["xtts_synthesis"] if x["ok"]]
    stability_passed = sum(1 for x in results["stability"] if x["ok"])
    stability_total = len(results["stability"])

    results["summary"] = {
        "gemma_repeated": summarize_latency(gemma_repeated),
        "gemma_prompt_scaling": summarize_latency(gemma_scaling),
        "xtts_synthesis": summarize_latency(xtts_synth),
        "xtts_rtf": summarize_latency(xtts_rtf),
        "end_to_end_synthesis_only": summarize_latency(e2e_synth),
        "end_to_end_with_playback": summarize_latency(e2e_playback),
        "sanitizer": summarize_latency(sanitizer),
        "stability_passed": stability_passed,
        "stability_total": stability_total,
        "stability_pass_rate": stability_passed / stability_total if stability_total else 0.0,
    }

    target_e2e = results["summary"]["end_to_end_synthesis_only"]["mean"]
    target_rtf = results["summary"]["xtts_rtf"]["mean"]
    gemma_p95 = results["summary"]["gemma_repeated"]["p95"]

    latency_score = 100.0
    latency_score -= max(0.0, target_e2e - 2.0) * 8.0
    latency_score -= max(0.0, target_rtf - 0.5) * 20.0
    latency_score -= max(0.0, gemma_p95 - 1.5) * 10.0
    latency_score = max(0.0, min(100.0, latency_score))

    results["summary"]["latency_score_0_100"] = latency_score

    json_path = output_dir / "benchmark_advanced_results.json"
    csv_path = output_dir / "benchmark_advanced_results.csv"
    report_path = output_dir / "benchmark_advanced_report.txt"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(flat_rows, csv_path)

    report_lines = []
    report_lines.append("LocalAI V3 Advanced Benchmark Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Created at: {results['created_at']}")
    report_lines.append(f"Output dir: {output_dir}")
    report_lines.append("")
    report_lines.append("[Load]")
    report_lines.append(f"Gemma + XTTS cold load: {results['load']['gemma_xtts_seconds']:.3f}s")
    report_lines.append(f"ASR cold load: {results['load']['asr_seconds']:.3f}s")
    report_lines.append("")
    report_lines.append("[Gemma repeated latency]")
    for key, value in results["summary"]["gemma_repeated"].items():
        report_lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    report_lines.append("")
    report_lines.append("[XTTS synthesis]")
    for key, value in results["summary"]["xtts_synthesis"].items():
        report_lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    report_lines.append("")
    report_lines.append("[XTTS RTF]")
    for key, value in results["summary"]["xtts_rtf"].items():
        report_lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    report_lines.append("")
    report_lines.append("[End-to-end synthesis-only]")
    for key, value in results["summary"]["end_to_end_synthesis_only"].items():
        report_lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    report_lines.append("")
    report_lines.append("[End-to-end with playback]")
    for key, value in results["summary"]["end_to_end_with_playback"].items():
        report_lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    report_lines.append("")
    report_lines.append("[Stability]")
    report_lines.append(f"Passed: {stability_passed}/{stability_total}")
    report_lines.append(f"Pass rate: {results['summary']['stability_pass_rate']:.2%}")
    report_lines.append("")
    report_lines.append("[Score]")
    report_lines.append(f"Latency score: {latency_score:.1f}/100")
    report_lines.append("")
    report_lines.append("[Files]")
    report_lines.append(f"JSON: {json_path}")
    report_lines.append(f"CSV: {csv_path}")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    plt = try_import_matplotlib()

    if plt is None:
        print("[WARN] matplotlib not installed. Charts skipped.")
        print(r"Install: .\.venv\Scripts\python.exe -m pip install matplotlib")
    else:
        save_bar(
            plt,
            labels=["Gemma+XTTS", "ASR"],
            values=[
                results["load"]["gemma_xtts_seconds"],
                results["load"]["asr_seconds"],
            ],
            title="Cold Load Latency",
            ylabel="Seconds",
            path=output_dir / "01_cold_load_latency.png",
        )

        save_line(
            plt,
            labels=[str(x["run"]) for x in results["gemma_repeated_latency"]],
            values=[x["seconds"] for x in results["gemma_repeated_latency"]],
            title="Gemma Repeated Latency",
            ylabel="Seconds",
            path=output_dir / "02_gemma_repeated_latency_line.png",
        )

        summary = results["summary"]["gemma_repeated"]
        save_bar(
            plt,
            labels=["p50", "p90", "p95", "p99", "max"],
            values=[summary["p50"], summary["p90"], summary["p95"], summary["p99"], summary["max"]],
            title="Gemma Latency Percentiles",
            ylabel="Seconds",
            path=output_dir / "03_gemma_latency_percentiles.png",
        )

        save_bar(
            plt,
            labels=[x["name"] for x in results["gemma_prompt_scaling"]],
            values=[x["seconds"] for x in results["gemma_prompt_scaling"]],
            title="Gemma Prompt Scaling Latency",
            ylabel="Seconds",
            path=output_dir / "04_gemma_prompt_scaling_latency.png",
        )

        save_scatter(
            plt,
            x_values=[x["prompt_chars"] for x in results["gemma_prompt_scaling"]],
            y_values=[x["seconds"] for x in results["gemma_prompt_scaling"]],
            labels=[x["name"] for x in results["gemma_prompt_scaling"]],
            title="Prompt Length vs Gemma Latency",
            xlabel="Prompt characters",
            ylabel="Seconds",
            path=output_dir / "05_prompt_length_vs_gemma_latency.png",
        )

        save_bar(
            plt,
            labels=[x["name"] for x in results["xtts_synthesis"]],
            values=[x["synthesis_seconds"] for x in results["xtts_synthesis"]],
            title="XTTS Synthesis Latency",
            ylabel="Seconds",
            path=output_dir / "06_xtts_synthesis_latency.png",
        )

        save_bar(
            plt,
            labels=[x["name"] for x in results["xtts_synthesis"]],
            values=[x["rtf"] for x in results["xtts_synthesis"]],
            title="XTTS Real-Time Factor",
            ylabel="RTF lower is better",
            path=output_dir / "07_xtts_rtf.png",
        )

        save_bar(
            plt,
            labels=[x["name"] for x in results["xtts_synthesis"]],
            values=[x["sanitize_seconds"] * 1000 for x in results["xtts_synthesis"]],
            title="Sanitizer Latency",
            ylabel="Milliseconds",
            path=output_dir / "08_sanitizer_latency_ms.png",
        )

        save_scatter(
            plt,
            x_values=[x["clean_chars"] for x in results["xtts_synthesis"]],
            y_values=[x["synthesis_seconds"] for x in results["xtts_synthesis"]],
            labels=[x["name"] for x in results["xtts_synthesis"]],
            title="Clean Text Length vs XTTS Synthesis Latency",
            xlabel="Clean text characters",
            ylabel="Seconds",
            path=output_dir / "09_clean_text_length_vs_xtts_latency.png",
        )

        save_multi_line(
            plt,
            labels=[x["name"] for x in results["end_to_end_synthesis_only"]],
            series={
                "Gemma": [x["gemma_seconds"] for x in results["end_to_end_synthesis_only"]],
                "XTTS synthesis": [x["xtts_synthesis_seconds"] for x in results["end_to_end_synthesis_only"]],
                "Total": [x["total_seconds"] for x in results["end_to_end_synthesis_only"]],
            },
            title="End-to-End Latency Breakdown",
            ylabel="Seconds",
            path=output_dir / "10_end_to_end_latency_breakdown.png",
        )

        save_bar(
            plt,
            labels=[x["name"] for x in results["end_to_end_with_playback"]],
            values=[x["seconds"] for x in results["end_to_end_with_playback"]],
            title="End-to-End Latency With Audio Playback",
            ylabel="Seconds",
            path=output_dir / "11_end_to_end_with_playback.png",
        )

        save_bar(
            plt,
            labels=[x["name"] for x in results["stability"]],
            values=[1 if x["ok"] else 0 for x in results["stability"]],
            title="Special Text Stability",
            ylabel="Pass = 1, fail = 0",
            path=output_dir / "12_stability_pass_fail.png",
        )

        save_bar(
            plt,
            labels=["Latency score"],
            values=[latency_score],
            title="Overall Latency Score",
            ylabel="Score out of 100",
            path=output_dir / "13_latency_score.png",
        )

        memory_labels = []
        rss_values = []

        for name, snapshot in results["memory"].items():
            if snapshot.get("rss_mb") is not None:
                memory_labels.append(name.replace("_", " "))
                rss_values.append(float(snapshot["rss_mb"]))

        if rss_values:
            save_bar(
                plt,
                labels=memory_labels,
                values=rss_values,
                title="Process Memory RSS",
                ylabel="MB",
                path=output_dir / "14_process_memory_rss.png",
            )

        cuda_labels = []
        cuda_values = []

        for name, snapshot in results["memory"].items():
            if snapshot.get("cuda_reserved_mb") is not None:
                cuda_labels.append(name.replace("_", " "))
                cuda_values.append(float(snapshot["cuda_reserved_mb"]))

        if cuda_values:
            save_bar(
                plt,
                labels=cuda_labels,
                values=cuda_values,
                title="CUDA Reserved Memory",
                ylabel="MB",
                path=output_dir / "15_cuda_reserved_memory.png",
            )

    print("\n" + "\n".join(report_lines))
    print("\n[DONE] Advanced benchmark finished.")
    print(f"[DONE] Open folder: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_advanced_benchmark())