from __future__ import annotations

import base64
import gc
import mimetypes
import time
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


class QwenVision:
    def __init__(
        self,
        root_dir: Path,
        model_name: str = "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        mmproj_name: str = "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.model_dir = self.root_dir / "models" / "qwen_vl"

        self.model_path = self.model_dir / model_name
        self.mmproj_path = self.model_dir / mmproj_name

        self.llm: Llama | None = None
        self.chat_handler: Llava15ChatHandler | None = None

    def load(self) -> None:
        if self.llm is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen-VL model not found: {self.model_path}")

        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"Qwen-VL mmproj not found: {self.mmproj_path}")

        print(f"[QwenVision] Loading mmproj: {self.mmproj_path}")
        print(f"[QwenVision] Loading model: {self.model_path}")

        start = time.perf_counter()

        self.chat_handler = Llava15ChatHandler(
            clip_model_path=str(self.mmproj_path),
            verbose=False,
        )

        self.llm = Llama(
            model_path=str(self.model_path),
            chat_handler=self.chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,
            n_batch=256,
            flash_attn=True,
            verbose=False,
        )

        elapsed = time.perf_counter() - start

        print(f"[QwenVision] Loaded in {elapsed:.2f}s")

    def describe_image(
        self,
        image_path: Path,
        question: str = "Describe this image clearly and concisely.",
    ) -> str:
        if self.llm is None:
            raise RuntimeError("QwenVision is not loaded. Call load() first.")

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_data_url = self._image_to_data_url(image_path)

        prompt = (
            f"{question}\n"
            "Answer in concise natural English. "
            "Do not use markdown unless necessary."
        )

        print(f"[QwenVision] Image: {image_path}")
        print(f"[QwenVision] Question: {question!r}")

        start = time.perf_counter()

        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=220,
            temperature=0.2,
            top_p=0.9,
        )

        elapsed = time.perf_counter() - start

        answer = output["choices"][0]["message"]["content"].strip()
        answer = self._clean(answer)

        print(f"[QwenVision] Answer in {elapsed:.2f}s: {answer!r}")

        return answer

    def unload(self) -> None:
        print("[QwenVision] Unloading model...")

        self.llm = None
        self.chat_handler = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        print("[QwenVision] Unloaded.")

    def _image_to_data_url(self, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))

        if not mime_type:
            suffix = image_path.suffix.lower()

            if suffix in {".jpg", ".jpeg"}:
                mime_type = "image/jpeg"
            elif suffix == ".png":
                mime_type = "image/png"
            elif suffix == ".webp":
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"

        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")

        return f"data:{mime_type};base64,{b64}"

    def _clean(self, text: str) -> str:
        text = str(text or "").strip()

        for prefix in ["Assistant:", "Answer:", "The image shows"]:
            if text.startswith(prefix):
                if prefix == "The image shows":
                    return text
                text = text[len(prefix) :].strip()

        if not text:
            return "I cannot clearly understand the image."

        return text