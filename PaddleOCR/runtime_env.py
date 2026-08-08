"""Device / precision / model-class resolution shared by train_ocr and infer_ocr.

Written so the same scripts run unchanged on an NVIDIA box, on a domestic
accelerator (Ascend NPU, Cambricon MLU, Hygon DCU via the ROCm-style CUDA API),
on Apple MPS, or on a CPU-only CentOS 7 machine.
"""
from __future__ import annotations

import os
from typing import Any

import torch


def _has_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def detect_device() -> str:
    """Return the torch device string for whatever accelerator is present."""
    # Hygon DCU and AMD ROCm builds both report through the cuda API.
    if torch.cuda.is_available():
        return "cuda"
    if _has_module("torch_npu"):                      # Ascend 910/310
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            return "npu"
    if _has_module("torch_mlu"):                      # Cambricon
        mlu = getattr(torch, "mlu", None)
        if mlu is not None and mlu.is_available():
            return "mlu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(device: str, requested: str = "auto") -> str:
    """Pick a dtype name. CPU stays fp32: fp16 on CPU is slow and often unsupported."""
    if requested != "auto":
        if requested == "fp16" and device == "cpu":
            raise ValueError("fp16 on CPU is not supported; use fp32 (or bf16 on AMX CPUs)")
        return requested
    if device == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if device in ("npu", "mlu"):
        return "bf16"
    if device == "mps":
        return "fp16"
    return "fp32"


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_attn(device: str, requested: str = "auto") -> str:
    """flash_attention_2 only exists on recent NVIDIA; sdpa is the safe default."""
    if requested != "auto":
        return requested
    return "eager" if device in ("npu", "mlu") else "sdpa"


def configure_threads(num_threads: int | None = None) -> int:
    """Set intra-op threads. Matters a lot on a 48-core CPU-only run."""
    if num_threads is None:
        num_threads = int(os.environ.get("OMP_NUM_THREADS", 0)) or (os.cpu_count() or 8)
    torch.set_num_threads(num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    return num_threads


def enable_tf32(device: str) -> bool:
    """TF32 is an NVIDIA Ampere+ feature; skip it elsewhere."""
    if device != "cuda":
        return False
    try:
        ok = torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:
        return False
    torch.backends.cuda.matmul.allow_tf32 = ok
    torch.backends.cudnn.allow_tf32 = ok
    return ok


def resolve_model_class() -> Any:
    """AutoModelForMultimodalLM does not exist in every transformers release."""
    import transformers

    for name in (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    return transformers.AutoModel


def load_model(
    model_path: str,
    *,
    dtype_name: str,
    attn: str,
    device: str,
    use_device_map: bool = True,
    trust_remote_code: bool = True,
) -> Any:
    """Load the VLM, tolerating the dtype/torch_dtype kwarg rename in transformers 4.56."""
    model_cls = resolve_model_class()
    torch_dtype = DTYPES[dtype_name]

    kwargs: dict[str, Any] = {
        "attn_implementation": attn,
        "trust_remote_code": trust_remote_code,
    }
    # device_map needs accelerate, and on CPU it buys nothing.
    if use_device_map and device != "cpu" and _has_module("accelerate"):
        kwargs["device_map"] = "auto"

    try:
        model = model_cls.from_pretrained(model_path, dtype=torch_dtype, **kwargs)
    except TypeError:
        model = model_cls.from_pretrained(model_path, torch_dtype=torch_dtype, **kwargs)

    if "device_map" not in kwargs:
        model = model.to(device)
    return model


def describe(device: str, dtype_name: str, threads: int) -> str:
    bits = [f"device={device}", f"dtype={dtype_name}", f"threads={threads}"]
    if device == "cuda":
        try:
            bits.append(f"gpu={torch.cuda.get_device_name(0)}")
            bits.append(f"n_gpu={torch.cuda.device_count()}")
        except Exception:
            pass
    return "  ".join(bits)
