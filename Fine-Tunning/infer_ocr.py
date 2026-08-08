#!/usr/bin/env python3
"""
Unified Inference & Evaluation Tool for Cosmetic Poster OCR.
Supports both Raw Base Models and LoRA Checkpoints for Qwen3.5-0.8B & GLM-OCR.
Generates comprehensive Markdown comparison reports with latency statistics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from PIL import Image as PILImage
import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
)

DEFAULT_PROMPT = (
    "请逐字识别这张化妆品海报中的所有可见文字。同时辨别并提取其中属于Product Detail Page Claim（商品详情页宣称）的文字。"
    "Product Detail Page Claim包括：商品的核心功效承诺（如“淡化细纹”、“深层补水”）、核心成分说明（如“高浓度玻尿酸”、“烟酰胺”）、实验或测试数据支持（如“细纹减少20%”、“真人测试认证”）、适用肤质及具体的使用体感描述。"
    "Product Detail Page Claim排除：禁止提取品牌Logo名称、价格/净含量/规格、简单实验条件（“使用前”、“使用后”）、证书/论文照片内文字或无关背景字符。"
    "若出现角标, 则如实保留并统一添加全角括号包裹，示例：`¹ ³ VI X *` → `（1） （3） （VI） （X） （*）`。"
    "保持原有阅读顺序和换行，只输出识别到的纯文本，不要解释，不要使用Markdown。"
    "如果图片中没有符合上述定义的文字，只输出<EMPTY>。"
)


def resolve_lora_path(lora_path: str | None) -> str | None:
    if not lora_path:
        return None
    p = Path(lora_path).expanduser().resolve()
    if (p / "adapter_config.json").is_file():
        return str(p)

    candidates = [p, p.parent] if p.parent.exists() else [p]
    if p.exists() and p.is_dir():
        candidates.extend(list(p.glob("checkpoint-*")))
    if p.parent.exists():
        candidates.extend(list(p.parent.glob("checkpoint-*")))

    for cand in candidates:
        if cand.is_dir() and (cand / "adapter_config.json").is_file():
            print(f"💡 自动查找到包含 adapter_config.json 的 Checkpoint 目录: {cand}")
            return str(cand)

    print(f"⚠️ 在 {lora_path} 及其关联路径下均未找到 adapter_config.json！")
    return str(p)


def load_model_and_processor(
    model_name: str,
    model_type: str,
    lora_dir: str | None,
    max_pixels: int
):
    # is_bf16_supported() 需要先确认有 CUDA 设备，否则在纯 CPU / MPS 机器上
    # 会报错或给出无意义的结果。train_qwen35.py 里是同样的判断顺序。
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    print(f"🤗 正在加载 [{model_type.upper()}] 基础模型与处理器: {model_name}...")

    if model_type == "glm":
        try:
            processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
            )
        except Exception:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
            )
    else:
        processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )

    resolved_lora = resolve_lora_path(lora_dir)
    if resolved_lora and (Path(resolved_lora) / "adapter_config.json").is_file():
        print(f"💾 正在加载 LoRA 适配器: {resolved_lora}")
        model = PeftModel.from_pretrained(model, resolved_lora)
    else:
        print("⚡ 未指定或未找到有效 LoRA 适配器，将直接使用原始基座模型 (Raw Base Model) 进行推理！")

    model.eval()
    return model, processor


def run_inference(model, processor, image_path: Path, prompt: str) -> str:
    image = PILImage.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def evaluate(args):
    start_total_time = time.time()

    # 自动识别模型类型
    model_type = args.model_type
    if model_type == "auto":
        model_name_lower = args.model_name.lower()
        if "glm" in model_name_lower:
            model_type = "glm"
        else:
            model_type = "qwen"

    model, processor = load_model_and_processor(
        model_name=args.model_name,
        model_type=model_type,
        lora_dir=args.lora_dir,
        max_pixels=args.max_pixels,
    )

    manifest_path = Path(args.test_file).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"📊 开始在测试集上运行推理评估（共 {len(samples)} 个样本）...")

    results = []

    for idx, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            continue

        image_name = (
            sample.get("source_file")
            or sample.get("file_name")
            or sample.get("image")
            or sample.get("filename")
            or sample.get("image_path")
        )
        if not image_name:
            print(f"⚠️ 样本 {idx} 缺少图片键: {list(sample.keys())}")
            continue

        if "claim_text" in sample:
            claim_val = sample["claim_text"]
            ground_truth = "\n".join(claim_val) if isinstance(claim_val, list) else str(claim_val)
        elif "lines" in sample:
            lines_val = sample["lines"]
            ground_truth = "\n".join(lines_val) if isinstance(lines_val, list) else str(lines_val)
        elif "text" in sample:
            ground_truth = str(sample["text"])
        else:
            ground_truth = "<EMPTY>"

        if not ground_truth.strip():
            ground_truth = "<EMPTY>"

        image_path = Path(image_name).expanduser()
        if not image_path.is_absolute():
            if (image_root / image_path).is_file():
                image_path = image_root / image_path
            elif (image_root / "images" / image_path).is_file():
                image_path = image_root / "images" / image_path
            elif (image_root / "images6" / image_path).is_file():
                image_path = image_root / "images6" / image_path
            else:
                image_path = image_root / image_path
        image_path = image_path.resolve()

        if not image_path.is_file():
            print(f"❌ 样本 {idx}: 找不到图片文件: {image_path}")
            continue

        print(f"\n🔍 [{idx}/{len(samples)}] 正在推理: {image_name}")

        try:
            start_sample_time = time.time()
            prediction = run_inference(model, processor, image_path, args.prompt)
            sample_duration = time.time() - start_sample_time

            print(f"--- 真实标注 (Ground Truth) ---\n{ground_truth}")
            print(f"--- 模型预测 (Prediction) ---\n{prediction}")
            print(f"⏱️ 单图推理耗时: {sample_duration:.2f} 秒")
            print("-" * 40)

            results.append({
                "image": image_name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "duration": sample_duration
            })
        except Exception as e:
            print(f"❌ 推理失败: {image_name}, 错误原因: {str(e)}")

    total_duration = time.time() - start_total_time
    avg_duration = total_duration / len(results) if results else 0.0

    base_model_name = Path(args.model_name).name
    checkpoint_name = Path(args.lora_dir).name if (args.lora_dir and Path(args.lora_dir).exists()) else "Raw-Base-Model"
    report_path = Path(args.output_report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# 🧪 [{base_model_name}] ({checkpoint_name}) 测试集评估报告\n\n")
        f.write(f"- **评估模型 (Model Name)**: `{base_model_name}`\n")
        f.write(f"- **模型架构 (Architecture)**: `{model_type.upper()}`\n")
        f.write(f"- **基础模型路径 (Base Model Path)**: `{args.model_name}`\n")
        f.write(f"- **模型状态 (Model Status)**: `{'LoRA Fine-tuned (' + checkpoint_name + ')' if checkpoint_name != 'Raw-Base-Model' else '原始未经微调基座模型 (Raw Base Model)'}`\n")
        f.write(f"- **评估测试集文件 (Test File)**: `{manifest_path.name}`\n")
        f.write(f"- **成功评估样本数 (Evaluated Samples)**: {len(results)} / {len(samples)}\n")
        f.write(f"- **⏱️ 推理总耗时 (Total Duration)**: `{total_duration:.2f} 秒` ({total_duration / 60:.2f} 分钟)\n")
        f.write(f"- **⚡ 单图平均耗时 (Avg Time / Image)**: `{avg_duration:.2f} 秒/张`\n\n")

        for idx, res in enumerate(results, start=1):
            f.write(f"## 📷 样本 {idx}: {res['image']}\n\n")
            f.write(f"⏱️ *单图推理耗时: {res['duration']:.2f} 秒*\n\n")
            f.write("| 真实标注 (Ground Truth) | 模型预测 (Model Prediction) |\n")
            f.write("| :--- | :--- |\n")

            gt_html = res['ground_truth'].replace("\n", "<br>")
            pred_html = res['prediction'].replace("\n", "<br>")
            f.write(f"| {gt_html} | {pred_html} |\n\n")

    print(f"\n🎉 评估完成！总耗时: {total_duration:.2f}s, 单图平均: {avg_duration:.2f}s。测试报告已保存至: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Inference & Evaluation Script")
    parser.add_argument("--test-file", required=True, help="Path to test_manifest.json or ground truth JSON")
    parser.add_argument("--image-root", required=True, help="Root directory containing images")
    parser.add_argument("--model-name", required=True, help="HuggingFace model ID or local directory path")
    parser.add_argument(
        "--model-type",
        choices=["auto", "qwen", "glm"],
        default="auto",
        help="Model architecture: qwen, glm, or auto-detect"
    )
    parser.add_argument("--lora-dir", "--lora-adapter", dest="lora_dir", help="Path to LoRA checkpoint directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-pixels", type=int, default=401760)
    parser.add_argument(
        "--output-report",
        required=True,
        help="Output Markdown report path (e.g. outputs/evaluation_report.md)"
    )
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
