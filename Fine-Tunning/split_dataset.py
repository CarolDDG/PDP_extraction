#!/usr/bin/env python3
"""
Dataset Split & Manifest Generator for Cosmetic Poster OCR.
Supports 3-way split (80% Train, 10% Eval, 10% Test) or 2-way split (90% Train, 10% Eval).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_dataset(
    dataset_dir: str | Path,
    annotation_filename: str = "PDP_groundTruth_0805.json",
    image_dir_name: str = "images",
    split_mode: str = "3way",
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    annotation_file = dataset_dir / annotation_filename

    if not annotation_file.exists():
        raise FileNotFoundError(f"❌ 找不到源标注文件: {annotation_file}")

    print(f"📖 正在读取标注文件: {annotation_file}")
    with open(annotation_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_records = []
    skipped_count = 0

    for idx, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            continue
        source_file = item.get("source_file") or item.get("file_name") or item.get("image")
        claim_lines = item.get("claim_text") or item.get("lines") or []

        if not source_file:
            print(f"⚠️ 跳过第 {idx} 条记录: 缺少图片文件名/source_file")
            skipped_count += 1
            continue

        rel_image_path = f"{image_dir_name}/{source_file}" if not source_file.startswith(f"{image_dir_name}/") else source_file
        abs_image_path = dataset_dir / rel_image_path

        if not abs_image_path.exists():
            # 兼容软链接或不同图片目录名
            alt_path = dataset_dir / source_file
            if alt_path.exists():
                rel_image_path = source_file
                abs_image_path = alt_path
            else:
                print(f"⚠️ 图片不存在，跳过: {abs_image_path}")
                skipped_count += 1
                continue

        clean_lines = [line.strip() for line in claim_lines if line.strip()] if isinstance(claim_lines, list) else [str(claim_lines).strip()]

        processed_records.append({
            "file_name": rel_image_path,
            "lines": clean_lines if clean_lines else ["<EMPTY>"]
        })

    total_valid = len(processed_records)
    print(f"✅ 数据校验完成！有效样本数: {total_valid} | 跳过样本数: {skipped_count}")

    random.seed(seed)
    random.shuffle(processed_records)

    out_train_file = dataset_dir / "train_manifest.json"
    out_eval_file = dataset_dir / "eval_manifest.json"
    out_test_file = dataset_dir / "test_manifest.json"

    if split_mode == "3way":
        test_size = int(total_valid * test_ratio)
        eval_size = int(total_valid * val_ratio)

        test_data = processed_records[:test_size]
        eval_data = processed_records[test_size:test_size + eval_size]
        train_data = processed_records[test_size + eval_size:]

        with open(out_train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(out_eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        with open(out_test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        print("=" * 60)
        print("🎉 数据集 3-Way (80% Train / 10% Eval / 10% Test) 划分割完成！")
        print(f"  - 训练集: {out_train_file} ({len(train_data)} 条)")
        print(f"  - 验证集: {out_eval_file} ({len(eval_data)} 条)")
        print(f"  - 测试集: {out_test_file} ({len(test_data)} 条)")
        print("=" * 60)
    else:
        eval_size = int(total_valid * val_ratio)
        eval_data = processed_records[:eval_size]
        train_data = processed_records[eval_size:]

        with open(out_train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(out_eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        print("=" * 60)
        print("🎉 数据集 2-Way (90% Train / 10% Eval) 划分割完成！")
        print(f"  - 训练集: {out_train_file} ({len(train_data)} 条)")
        print(f"  - 验证集: {out_eval_file} ({len(eval_data)} 条)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Cosmetic Poster OCR Dataset Splitter")
    parser.add_argument(
        "--dataset-dir",
        default="/mnt/d/LLM_OCR/dataset",
        help="Path to dataset folder (default: /mnt/d/LLM_OCR/dataset)",
    )
    parser.add_argument(
        "--annotation-filename",
        default="PDP_groundTruth_0805.json",
        help="Ground truth annotation JSON file name",
    )
    parser.add_argument(
        "--image-dir-name",
        default="images",
        help="Subfolder containing images (default: images)",
    )
    parser.add_argument(
        "--split-mode",
        choices=["3way", "2way"],
        default="3way",
        help="Split mode: 3way (80/10/10) or 2way (90/10)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio (default: 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio for 3way split (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_dataset(
        dataset_dir=args.dataset_dir,
        annotation_filename=args.annotation_filename,
        image_dir_name=args.image_dir_name,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
