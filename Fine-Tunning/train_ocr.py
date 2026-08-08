#!/usr/bin/env python3
"""
Unified Multi-Model LoRA Fine-Tuning Entry Point for Cosmetic Poster OCR.
Supports Qwen3.5-0.8B and GLM-OCR with auto-detection of model architecture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train_qwen35 import train_qwen35, Qwen35TrainConfig, DEFAULT_PROMPT
from train_glm_ocr import train_glm_ocr, GLMOCRTrainConfig


def main():
    parser = argparse.ArgumentParser(description="Unified LoRA Fine-Tuner for Qwen3.5 & GLM-OCR")
    parser.add_argument("--train-file", required=True, help="Path to train_manifest.json")
    parser.add_argument("--image-root", required=True, help="Root directory containing images")
    parser.add_argument(
        "--model-type",
        choices=["auto", "qwen", "glm"],
        default="auto",
        help="Model architecture type: qwen, glm, or auto-detect from --model-name",
    )
    parser.add_argument("--model-name", required=True, help="HuggingFace model ID or local directory path")
    parser.add_argument("--output-dir", help="Output directory for LoRA weights")
    parser.add_argument("--eval-file", help="Path to eval_manifest.json")
    parser.add_argument("--epochs", type=float, help="Number of training epochs (default: 3.0 for Qwen, 10.0 for GLM)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, help="Max image pixels constraint")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "no"], default="epoch")
    parser.add_argument("--resume-from-checkpoint", help="Path to checkpoint folder")
    args = parser.parse_args()

    # 自动识别模型类型
    model_type = args.model_type
    if model_type == "auto":
        model_name_lower = args.model_name.lower()
        if "glm" in model_name_lower:
            model_type = "glm"
        elif "qwen" in model_name_lower:
            model_type = "qwen"
        else:
            print(f"⚠️ 无法从模型路径 '{args.model_name}' 自动推导类型，默认使用 Qwen 架构。")
            model_type = "qwen"

    print("=" * 60)
    print(f"🚀 启动 [{model_type.upper()}] LoRA 微调任务")
    print(f"🤗 模型路径: {args.model_name}")
    print("=" * 60)

    if model_type == "glm":
        output_dir = args.output_dir or "outputs/glm-ocr-cosmetic-ocr-lora"
        epochs = args.epochs if args.epochs is not None else 10.0
        max_pixels = args.max_pixels if args.max_pixels is not None else 401760

        config = GLMOCRTrainConfig(
            train_file=args.train_file,
            image_root=args.image_root,
            output_dir=output_dir,
            eval_file=args.eval_file,
            model_name=args.model_name,
            prompt=DEFAULT_PROMPT,
            epochs=epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_pixels=max_pixels,
            precision=args.precision,
            num_workers=args.num_workers,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        metrics = train_glm_ocr(config)
    else:
        output_dir = args.output_dir or "outputs/qwen3.5-0.8b-cosmetic-ocr-lora"
        epochs = args.epochs if args.epochs is not None else 3.0
        max_pixels = args.max_pixels if args.max_pixels is not None else 1048576

        config = Qwen35TrainConfig(
            train_file=args.train_file,
            image_root=args.image_root,
            output_dir=output_dir,
            eval_file=args.eval_file,
            model_name=args.model_name,
            prompt=DEFAULT_PROMPT,
            epochs=epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_pixels=max_pixels,
            precision=args.precision,
            num_workers=args.num_workers,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        metrics = train_qwen35(config)

    print("🎉 训练完成！结果统计指标:")
    import json
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
