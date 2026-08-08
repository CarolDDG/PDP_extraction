#!/usr/bin/env python3
"""
LoRA Fine-Tuning entry point for cosmetic-poster OCR with Qwen3.5-0.8B.
Includes WSL2 memory patch and hardware resource monitoring.
"""

from __future__ import annotations

import os
# 开启显存分配优化，解决 WSL2 驱动在分配大块显存时触发的虚拟化 OOM Bug
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import psutil
from datasets import Dataset, Image, Sequence
from peft import LoraConfig
from transformers import AutoModelForMultimodalLM, AutoProcessor, set_seed
from trl import SFTConfig, SFTTrainer

# 注入针对 WSL2 驱动上报 16 Exabytes 虚拟内存导致的 OOM Bug 修复补丁
import transformers.modeling_utils
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


DEFAULT_PROMPT = (
    "请逐字识别这张化妆品海报中的所有可见文字。同时辨别并提取其中属于Product Detail Page Claim（商品详情页宣称）的文字。"
    "Product Detail Page Claim包括：商品的核心功效承诺（如“淡化细纹”、“深层补水”）、核心成分说明（如“高浓度玻尿酸”、“烟酰胺”）、实验或测试数据支持（如“细纹减少20%”、“真人测试认证”）、适用肤质及具体的使用体感描述。"
    "Product Detail Page Claim排除：禁止提取品牌Logo名称、价格/净含量/规格、简单实验条件（“使用前”、“使用后”）、证书/论文照片内文字或无关背景字符。"
    "若出现角标, 则如实保留并统一添加全角括号包裹，示例：`¹ ³ VI X *` → `（1） （3） （VI） （X） （*）`。"
    "保持原有阅读顺序和换行，只输出识别到的纯文本，不要解释，不要使用Markdown。"
    "如果图片中没有符合上述定义的文字，只输出<EMPTY>。"
)


class HardwareMonitor:
    """轻量级硬件监视器，默认每 5 秒监测 CPU、RAM 及 NVIDIA GPU Util/VRAM 占用。"""
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_usages: list[float] = []
        self.peak_cpu = 0.0
        self.peak_ram = 0.0
        self.gpu_utils: list[float] = []
        self.peak_gpu_mem = 0.0
        self.thread: threading.Thread | None = None

    def start(self):
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _monitor(self):
        has_nvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            has_nvml = True
        except Exception:
            pass

        process = psutil.Process()

        while not self.stop_event.is_set():
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                if cpu_pct > 0.0:
                    self.cpu_usages.append(cpu_pct)
                    self.peak_cpu = max(self.peak_cpu, cpu_pct)

                ram_mb = process.memory_info().rss / (1024 ** 2)
                self.peak_ram = max(self.peak_ram, ram_mb)

                if has_nvml:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        self.gpu_utils.append(float(util))
                    except Exception:
                        pass

                if torch.cuda.is_available():
                    gpu_mem_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
                    self.peak_gpu_mem = max(self.peak_gpu_mem, gpu_mem_mb)
            except Exception:
                pass

            self.stop_event.wait(self.interval)

        if has_nvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_stats(self) -> dict[str, float]:
        avg_cpu = sum(self.cpu_usages) / len(self.cpu_usages) if self.cpu_usages else 0.0
        avg_gpu = sum(self.gpu_utils) / len(self.gpu_utils) if self.gpu_utils else 0.0
        
        if torch.cuda.is_available():
            self.peak_gpu_mem = max(self.peak_gpu_mem, torch.cuda.max_memory_allocated(0) / (1024 ** 2))

        return {
            "cpu_percent_avg": round(avg_cpu, 2),
            "cpu_percent_peak": round(self.peak_cpu, 2),
            "memory_peak_mb": round(self.peak_ram, 2),
            "gpu_util_avg": round(avg_gpu, 2),
            "gpu_memory_peak_mb": round(self.peak_gpu_mem, 2),
        }


@dataclass
class Qwen35TrainConfig:
    train_file: str
    image_root: str
    output_dir: str = "outputs/qwen3.5-0.8b-cosmetic-ocr-lora"
    eval_file: str | None = None
    model_name: str = "Qwen/Qwen3.5-0.8B"
    prompt: str = DEFAULT_PROMPT
    epochs: float = 3.0
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_pixels: int = 1048576
    logging_steps: int = 10
    save_strategy: str = "epoch"
    seed: int = 42
    precision: str = "auto"
    num_workers: int = 0
    resume_from_checkpoint: str | None = None


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = json.load(handle)

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list or JSONL records")
    return rows


def _build_dataset(annotation_file: str, image_root: str, prompt: str) -> Dataset:
    annotation_path = Path(annotation_file).expanduser().resolve()
    root = Path(image_root).expanduser().resolve()
    records: list[dict[str, Any]] = []

    for index, row in enumerate(_read_annotations(annotation_path), start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} must be a JSON object")

        if "image" in row and "text" in row:
            image_value = row["image"]
            text_value = row["text"]
        elif "file_name" in row and "lines" in row:
            image_value = row["file_name"]
            if not isinstance(row["lines"], list) or not all(isinstance(line, str) for line in row["lines"]):
                raise TypeError(f"Record {index}: 'lines' must be a list of strings")
            text_value = "\n".join(line for line in row["lines"] if line.strip())
        else:
            raise ValueError(f"Record {index} must contain either 'image'/'text' or 'file_name'/'lines'")

        if not isinstance(image_value, str) or not isinstance(text_value, str):
            raise TypeError(f"Record {index}: image path and OCR text must both be strings")
        if not text_value.strip():
            text_value = "<EMPTY>"

        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = root / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Record {index}: image does not exist: {image_path}")

        records.append({
            "images": [str(image_path)],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text_value}],
                },
            ],
        })

    return Dataset.from_list(records).cast_column("images", Sequence(Image(decode=True)))


def train_qwen35(config: Qwen35TrainConfig) -> dict[str, float]:
    """Fine-tune Qwen3.5-0.8B for cosmetic poster OCR."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training")

    if config.precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        precision = config.precision
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but GPU does not support BF16")
    model_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    set_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.cuda.get_device_capability(0)[0] >= 8
    torch.backends.cudnn.allow_tf32 = torch.cuda.get_device_capability(0)[0] >= 8

    monitor = HardwareMonitor(interval=5.0)
    monitor.start()

    train_dataset = _build_dataset(config.train_file, config.image_root, config.prompt)
    eval_dataset = (
        _build_dataset(config.eval_file, config.image_root, config.prompt)
        if config.eval_file
        else None
    )

    processor = AutoProcessor.from_pretrained(config.model_name, max_pixels=config.max_pixels)
    model = AutoModelForMultimodalLM.from_pretrained(
        config.model_name,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules="all-linear",
        exclude_modules=r".*visual.*",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        tf32=torch.cuda.get_device_capability(0)[0] >= 8,
        optim="adamw_torch_fused",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        max_length=None,
        remove_unused_columns=False,
        report_to="tensorboard",
        seed=config.seed,
        dataloader_num_workers=config.num_workers,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    trainer.save_state()

    monitor.stop()
    hw_stats = monitor.get_stats()

    metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
    metrics.update(hw_stats)
    return metrics


def _parse_args() -> Qwen35TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-0.8B with LoRA")
    parser.add_argument("--train-file", required=True, help="Path to train_manifest.json")
    parser.add_argument("--image-root", required=True, help="Root directory containing images")
    parser.add_argument("--output-dir", default="outputs/qwen3.5-0.8b-cosmetic-ocr-lora")
    parser.add_argument("--eval-file", help="Path to eval_manifest.json")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "no"], default="epoch")
    parser.add_argument("--resume-from-checkpoint", help="Path to checkpoint folder")
    args = parser.parse_args()
    return Qwen35TrainConfig(**vars(args))


if __name__ == "__main__":
    metrics = train_qwen35(_parse_args())
    print("=" * 60)
    print("🎉 Qwen3.5-0.8B 微调已完成！统计指标与硬件资源负载如下:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("=" * 60)
