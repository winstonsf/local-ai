"""
llm_ov.py — run an instruct LLM on the Intel iGPU via OpenVINO GenAI.

Thin wrapper over openvino_genai.LLMPipeline targeting the Arc iGPU ("GPU"),
with a Qwen-style chat formatter and optional token streaming. Falls back to CPU
if the GPU pipeline can't be created.

    from llm_ov import LLM
    llm = LLM("models/qwen2.5-3b-ov", device="GPU")
    print(llm.chat(system="You are concise.", user="Say hi in 3 words."))
"""

from __future__ import annotations

from pathlib import Path


def qwen_prompt(system: str, user: str) -> str:
    """Qwen2.5 chat format -> a single prompt string for LLMPipeline.generate."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class LLM:
    def __init__(self, model_dir: str = "models/qwen2.5-3b-ov", device: str = "GPU",
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
                self.pipe = og.LLMPipeline(model_dir, dev)
                self.device = dev
                break
            except Exception as e:
                last_err = e
        else:
            raise RuntimeError(f"no device worked (tried {order}): {last_err}")

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.0, stream: bool = False) -> str:
        cfg = self._og.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.do_sample = True
            cfg.temperature = temperature
        else:
            cfg.do_sample = False  # greedy, deterministic

        if stream:
            def _printer(chunk: str) -> bool:
                print(chunk, end="", flush=True)
                return False  # keep going
            out = self.pipe.generate(prompt, cfg, _printer)
            print()
            return str(out)
        return str(self.pipe.generate(prompt, cfg))

    def chat(self, system: str, user: str, **kw) -> str:
        return self.generate(qwen_prompt(system, user), **kw)
