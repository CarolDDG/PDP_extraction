#!/bin/bash
set -e

# ==============================================================================
# Windows WSL + RTX 4070 微调一键启动脚本
# 支持 Qwen3.5-0.8B 与 GLM-OCR LoRA 微调
# ==============================================================================

MODEL_CHOICE="${1:-qwen}" # 可选参数: qwen (默认), glm, all

BASE_DIR="/mnt/d/LLM_OCR"
DATASET_DIR="${BASE_DIR}/dataset"
TRAIN_FILE="${DATASET_DIR}/train_manifest.json"
EVAL_FILE="${DATASET_DIR}/eval_manifest.json"
TEST_FILE="${DATASET_DIR}/test_manifest.json"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================="
echo "🚀 WSL RTX4070 化妆品海报 OCR 微调任务启动"
echo "📂 数据集目录: ${DATASET_DIR}"
echo "=================================================="

# 1. 检查并生成数据集划分清单
if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${TEST_FILE}" ]; then
    echo "📄 正在自动生成 80% + 10% + 10% 数据集划分清单..."
    python3 "${SCRIPT_DIR}/split_dataset.py" \
        --dataset-dir "${DATASET_DIR}" \
        --split-mode 3way
fi

# 2. 根据选定模型启动微调
if [ "${MODEL_CHOICE}" = "qwen" ] || [ "${MODEL_CHOICE}" = "all" ]; then
    QWEN_MODEL_PATH="${BASE_DIR}/models/Qwen3.5-0.8B"
    if [ ! -d "${QWEN_MODEL_PATH}" ]; then
        QWEN_MODEL_PATH="Qwen/Qwen3.5-0.8B"
    fi
    QWEN_OUTPUT_DIR="${BASE_DIR}/outputs/qwen3.5-0.8b-cosmetic-ocr-lora"

    echo ""
    echo "📌 启动 Qwen3.5-0.8B 微调..."
    python3 "${SCRIPT_DIR}/train_qwen35.py" \
        --train-file "${TRAIN_FILE}" \
        --eval-file "${EVAL_FILE}" \
        --image-root "${DATASET_DIR}" \
        --model-name "${QWEN_MODEL_PATH}" \
        --output-dir "${QWEN_OUTPUT_DIR}" \
        --epochs 3 \
        --batch-size 1 \
        --gradient-accumulation-steps 8 \
        --precision bf16
fi

if [ "${MODEL_CHOICE}" = "glm" ] || [ "${MODEL_CHOICE}" = "all" ]; then
    GLM_MODEL_PATH="${BASE_DIR}/models/GLM-OCR"
    if [ ! -d "${GLM_MODEL_PATH}" ]; then
        GLM_MODEL_PATH="zai-org/GLM-OCR"
    fi
    GLM_OUTPUT_DIR="${BASE_DIR}/outputs/glm-ocr-cosmetic-ocr-lora"

    echo ""
    echo "📌 启动 GLM-OCR 微调..."
    python3 "${SCRIPT_DIR}/train_glm_ocr.py" \
        --train-file "${TRAIN_FILE}" \
        --eval-file "${EVAL_FILE}" \
        --image-root "${DATASET_DIR}" \
        --model-name "${GLM_MODEL_PATH}" \
        --output-dir "${GLM_OUTPUT_DIR}" \
        --epochs 10 \
        --batch-size 1 \
        --gradient-accumulation-steps 8 \
        --precision bf16
fi

echo "=================================================="
echo "🎉 所有指定的微调任务已全部完成！"
echo "=================================================="
