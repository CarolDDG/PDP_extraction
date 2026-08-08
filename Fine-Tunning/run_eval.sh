#!/bin/bash
set -e

# ==============================================================================
# Windows WSL + RTX 4070 评估与基准推理一键启动脚本
# 支持 Qwen3.5-0.8B 与 GLM-OCR (Raw Base Model vs LoRA Fine-tuned)
# ==============================================================================

MODE="${1:-finetuned}" # 可选参数: finetuned (默认), raw, all

BASE_DIR="/mnt/d/LLM_OCR"
DATASET_DIR="${BASE_DIR}/dataset"
TEST_FILE="${DATASET_DIR}/test_manifest.json"
OUTPUT_DIR="${BASE_DIR}/outputs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "${OUTPUT_DIR}"

QWEN_BASE="${BASE_DIR}/models/Qwen3.5-0.8B"
if [ ! -d "${QWEN_BASE}" ]; then QWEN_BASE="Qwen/Qwen3.5-0.8B"; fi

GLM_BASE="${BASE_DIR}/models/GLM-OCR"
if [ ! -d "${GLM_BASE}" ]; then GLM_BASE="zai-org/GLM-OCR"; fi

QWEN_LORA="${OUTPUT_DIR}/qwen3.5-0.8b-cosmetic-ocr-lora"
GLM_LORA="${OUTPUT_DIR}/glm-ocr-cosmetic-ocr-lora"

echo "=================================================="
echo "🚀 启动模型推理与测试集评估 (模式: ${MODE})"
echo "=================================================="

# 1. 运行原始基座模型评测 (Raw Base Models)
if [ "${MODE}" = "raw" ] || [ "${MODE}" = "all" ]; then
    echo ""
    echo "📌 [1/2] 运行原始 Qwen3.5-0.8B 基座模型推理..."
    python3 "${SCRIPT_DIR}/infer_ocr.py" \
        --test-file "${TEST_FILE}" \
        --image-root "${DATASET_DIR}" \
        --model-name "${QWEN_BASE}" \
        --model-type qwen \
        --max-pixels 401760 \
        --output-report "${OUTPUT_DIR}/evaluation_report_qwen3.5_raw_testset.md"

    echo ""
    echo "📌 [2/2] 运行原始 GLM-OCR 基座模型推理..."
    python3 "${SCRIPT_DIR}/infer_ocr.py" \
        --test-file "${TEST_FILE}" \
        --image-root "${DATASET_DIR}" \
        --model-name "${GLM_BASE}" \
        --model-type glm \
        --max-pixels 401760 \
        --output-report "${OUTPUT_DIR}/evaluation_report_glm_ocr_raw_testset.md"
fi

# 2. 运行微调模型评测 (Fine-tuned LoRA Models)
if [ "${MODE}" = "finetuned" ] || [ "${MODE}" = "all" ]; then
    echo ""
    echo "📌 [1/2] 运行 Qwen3.5-0.8B (LoRA) 微调模型推理..."
    python3 "${SCRIPT_DIR}/infer_ocr.py" \
        --test-file "${TEST_FILE}" \
        --image-root "${DATASET_DIR}" \
        --model-name "${QWEN_BASE}" \
        --model-type qwen \
        --lora-dir "${QWEN_LORA}" \
        --max-pixels 401760 \
        --output-report "${OUTPUT_DIR}/evaluation_report_qwen3.5_finetuned_testset.md"

    echo ""
    echo "📌 [2/2] 运行 GLM-OCR (LoRA) 微调模型推理..."
    python3 "${SCRIPT_DIR}/infer_ocr.py" \
        --test-file "${TEST_FILE}" \
        --image-root "${DATASET_DIR}" \
        --model-name "${GLM_BASE}" \
        --model-type glm \
        --lora-dir "${GLM_LORA}" \
        --max-pixels 401760 \
        --output-report "${OUTPUT_DIR}/evaluation_report_glm_ocr_finetuned_testset.md"
fi

echo "=================================================="
echo "🎉 评估运行完毕！对比报告已存至: ${OUTPUT_DIR}"
echo "=================================================="
