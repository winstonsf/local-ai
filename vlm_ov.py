"""
vlm_ov.py — run a vision-language model on the Intel iGPU via OpenVINO GenAI.

Wraps openvino_genai.VLMPipeline (e.g. Qwen2-VL) so an image becomes text. Used
for the recognition side: turn a page image into clean markdown. A VL model runs
as a single graph, so it targets the iGPU ("GPU") rather than splitting the
vision encoder onto the NPU. Falls back to CPU if the GPU pipeline won't build.

    from vlm_ov import VLM
    vlm = VLM("models/qwen2-vl-2b-ov", device="GPU")
    md = vlm.recognize(pil_image, "Extract all text as clean markdown.")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov


def pil_to_tensor(img) -> ov.Tensor:
    """PIL image -> ov.Tensor of shape [1, H, W, 3] uint8 (what VLMPipeline expects)."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)[None]  # add batch dim
    return ov.Tensor(arr)


class VLM:
    def __init__(self, model_dir: str = "models/qwen2-vl-2b-ov", device: str = "GPU",
                 fallback: bool = True):
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"model dir '{model_dir}' not found — run the optimum-cli export first")
        import openvino_genai as og

        self._og = og
        order = list(dict.fromkeys([device] + (["CPU"] if fallback else [])))
        last_err = None
        for dev in order:
            try:
                self.pipe = og.VLMPipeline(model_dir, dev)
                self.device = dev
                break
            except Exception as e:
                last_err = e
        else:
            raise RuntimeError(f"no device worked (tried {order}): {last_err}")

    def recognize(self, img, prompt: str = "Extract all text from this image as clean markdown.",
                  max_new_tokens: int = 1024) -> str:
        cfg = self._og.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.do_sample = False  # deterministic OCR
        tensor = pil_to_tensor(img)
        # Param name changed across versions: try plural, fall back to singular.
        try:
            out = self.pipe.generate(prompt, images=[tensor], generation_config=cfg)
        except TypeError:
            out = self.pipe.generate(prompt, image=tensor, generation_config=cfg)
        return str(out)
