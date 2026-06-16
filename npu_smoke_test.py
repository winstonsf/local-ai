"""
npu_smoke_test.py — confirm OpenVINO can see and run on the Intel NPU (and iGPU).

Builds a trivial model (elementwise add), then compiles + runs it on every
available device. If NPU shows a correct result, the NPU path works end-to-end.

    python npu_smoke_test.py
"""

import numpy as np
import openvino as ov
from openvino import opset13 as ops

core = ov.Core()

print("OpenVINO:", ov.get_version())
print("Available devices:")
for d in core.available_devices:
    try:
        name = core.get_property(d, "FULL_DEVICE_NAME")
    except Exception as e:  # some devices don't expose every property
        name = f"(no FULL_DEVICE_NAME: {e})"
    print(f"  {d:6} -> {name}")


def tiny_model() -> ov.Model:
    """y = a + b, with static shapes (NPU prefers static)."""
    a = ops.parameter([1, 8], ov.Type.f32, name="a")
    b = ops.parameter([1, 8], ov.Type.f32, name="b")
    return ov.Model([ops.add(a, b)], [a, b], "add")


a = np.arange(8, dtype=np.float32).reshape(1, 8)
b = np.ones((1, 8), dtype=np.float32)
expected = a + b

print("\nPer-device run (expected:", expected.ravel().tolist(), "):")
for dev in core.available_devices:
    try:
        compiled = core.compile_model(tiny_model(), dev)
        out = compiled([a, b])[compiled.output(0)]
        ok = np.allclose(out, expected)
        print(f"  {dev:6} -> {'OK ' if ok else 'WRONG'} {out.ravel().tolist()}")
    except Exception as e:
        print(f"  {dev:6} -> FAILED: {type(e).__name__}: {e}")
