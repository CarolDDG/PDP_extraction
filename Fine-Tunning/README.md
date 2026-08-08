# Cosmetic Poster OCR Fine-Tuning & Evaluation Suite (WSL2 + RTX 4070)

An end-to-end, high-performance toolkit for LoRA fine-tuning and benchmark inference of Vision-Language Models (**Qwen3.5-0.8B** and **GLM-OCR**) on cosmetic poster OCR and **Product Detail Page (PDP) Claim Extraction**.

This repository is optimized for **Windows WSL2 + NVIDIA RTX 4070 (12GB VRAM)** environments, featuring memory patch workarounds for WSL2 GPU drivers, real-time background hardware monitoring, modular dataset splitting, and automated Markdown evaluation reporting.

---

## 🛠️ Environment & Prerequisites

### 1. System Requirements
- **OS**: Windows 11 with WSL2 (Ubuntu 22.04 LTS / 24.04 LTS)
- **GPU**: NVIDIA GeForce RTX 4070 (12GB VRAM) or higher
- **CUDA Driver**: NVIDIA CUDA 12.x driver installed on Windows host

### 2. Conda Environment Setup
Activate your dedicated Conda environment (e.g., `sft_gpu`):

```bash
conda activate sft_gpu
```

Required Python packages:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl datasets psutil pynvml pillow
```

### 3. WSL2 CUDA Driver Memory Bug Workarounds
Running PyTorch models under WSL2 GPU virtualization can trigger catastrophic Out-Of-Memory (OOM) errors due to driver bugs in reported virtual memory (e.g., driver reporting 16 Exabytes of virtual RAM). This codebase embeds two critical patches in all training scripts:
1. **PyTorch Memory Segment Optimization**:
   ```python
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```
2. **Transformers Caching Allocator Warmup Patch**:
   ```python
   import transformers.modeling_utils
   transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
   ```

---

## 📁 Repository Directory Structure

```text
fine-tunning/
├── README.md              # Detailed documentation & usage guide (English)
├── prompt0805.txt         # Standardised PDP Claim extraction prompt template
├── split_dataset.py       # Dataset splitter (3-way 80/10/10 or 2-way 90/10)
├── train_qwen35.py        # Dedicated LoRA SFT fine-tuner for Qwen3.5-0.8B
├── train_glm_ocr.py       # Dedicated LoRA SFT fine-tuner for GLM-OCR
├── train_ocr.py           # Unified multi-architecture fine-tuning entry point
├── infer_ocr.py           # Unified inference & evaluation benchmark tool
├── run_train.sh           # One-click WSL2 fine-tuning launcher script
└── run_eval.sh            # One-click WSL2 inference benchmark launcher script
```

---

## 🔍 Script Architecture & In-Depth Logic

### 1. Data Preprocessing & Splitting (`split_dataset.py`)

#### **Logic & Workflow**:
1. **Annotation Parsing**: Reads raw ground-truth JSON files (e.g., `PDP_groundTruth_0805.json`), extracting image filenames (`source_file`) and target OCR text (`claim_text` or `lines`).
2. **Path Validation**: Verifies that every referenced image file physically exists under the dataset directory (e.g., `images/filename.jpg`). Missing images are gracefully reported and skipped.
3. **Data Normalisation**: Trims whitespace line by line. If a ground-truth entry contains no text, it is tagged as `<EMPTY>`.
4. **Deterministic Shuffling**: Shuffles processed entries using a fixed random seed (`--seed 42`) for 100% reproducible dataset splits.
5. **Split Ratios**:
   - **`3way` Mode (Default)**: 80% Training (`train_manifest.json`), 10% Validation (`eval_manifest.json`), and 10% Testing (`test_manifest.json`).
   - **`2way` Mode**: 90% Training (`train_manifest.json`) and 10% Validation (`eval_manifest.json`).

#### **Manifest JSON Schema**:
```json
[
  {
    "file_name": "images/poster_001.jpg",
    "lines": [
      "深层补水 紧致肌肤",
      "高浓度玻脉酸（1）"
    ]
  }
]
```

---

### 2. Fine-Tuning Architecture (`train_qwen35.py`, `train_glm_ocr.py`, `train_ocr.py`)

#### **Data Pipeline**:
- Annotations from `train_manifest.json` are loaded into a HuggingFace `Dataset` object.
- Each record is converted into standard Chat ML multi-modal format:
  - `user`: Image + System Prompt (`prompt0805.txt`).
  - `assistant`: Ground truth OCR text (joined by newline `\n`).

#### **LoRA Strategy & Target Modules**:
- **Qwen3.5-0.8B (`train_qwen35.py`)**:
  - Target Modules: `all-linear` (adapts self-attention and MLP layers in language backbone).
  - Exclude Pattern: `exclude_modules=r".*visual.*"` (keeps vision encoder frozen to conserve VRAM).
  - Default Hyperparameters: `bf16`, `epochs=3.0`, `batch_size=1`, `gradient_accumulation_steps=8` (effective batch size = 8), `lr=1e-4`, `max_pixels=1048576`.
- **GLM-OCR (`train_glm_ocr.py`)**:
  - Model Loading: Uses `AutoModelForImageTextToText` (or fallback `AutoModelForMultimodalLM`) with `trust_remote_code=True`.
  - Exclude Pattern: `exclude_modules=r".*visual.*|.*vision.*"`.
  - Default Hyperparameters: `bf16`, `epochs=10.0`, `batch_size=1`, `gradient_accumulation_steps=8`, `max_pixels=401760`.
- **Unified Entry Point (`train_ocr.py`)**:
  - Inspects `--model-name` or `--model-type`. Auto-detects whether the target model belongs to `qwen` or `glm` family and automatically applies optimal resolution bounds and LoRA configurations.

#### **Real-Time Hardware Monitor (`HardwareMonitor`)**:
A daemon background thread runs during training, querying system state every 5 seconds via `psutil` and `pynvml`:
- Average & Peak CPU Utilization (%)
- Peak Host RAM Consumption (MB)
- Average GPU Utilization (%)
- Peak VRAM Allocation (MB) via `torch.cuda.max_memory_allocated(0)`
Hardware statistics are printed alongside evaluation loss upon training completion.

---

### 3. Inference & Evaluation Engine (`infer_ocr.py`)

#### **Logic & Workflow**:
1. **Model & Adapter Resolution**:
   - Loads base model weights (local folder or HuggingFace ID).
   - Searches for `adapter_config.json` inside `--lora-dir` or parent checkpoint folders (e.g., `checkpoint-96`, `checkpoint-160`). If found, wraps the base model with `PeftModel`. If not found, falls back to evaluating the **Raw Base Model**.
2. **Image Resolution Fallback**:
   - Resolves image paths by checking root dataset directory, `images/`, and `images6/` subdirectories.
3. **Precision Benchmark & Timing**:
   - Executes generation with `do_sample=False` (greedy decoding) and `max_new_tokens=512`.
   - Records single-image latency (`duration` in seconds) and overall benchmark execution time.
4. **Markdown Report Generation**:
   - Exports a formatted Markdown file (`--output-report`) containing summary metrics (total time, average time per image, evaluated sample count) and a side-by-side comparative table: `Ground Truth` vs `Model Prediction`.

---

### 4. Shell Launcher Scripts (`run_train.sh`, `run_eval.sh`)

- **`run_train.sh`**:
  - Automatically verifies manifest files; runs `split_dataset.py` if missing.
  - Launches Qwen3.5-0.8B or GLM-OCR training.
- **`run_eval.sh`**:
  - Runs test set evaluation across Raw Base Models or LoRA Fine-Tuned checkpoints.

---

## 🚀 Execution & Usage Guide

### Step 1: Dataset Splitting

Generate `train_manifest.json`, `eval_manifest.json`, and `test_manifest.json`:

```bash
# Run 3-way split (80% train, 10% eval, 10% test)
python3 split_dataset.py \
  --dataset-dir /mnt/d/LLM_OCR/dataset \
  --split-mode 3way \
  --seed 42
```

---

### Step 2: Fine-Tuning Models

#### Option A: Qwen3.5-0.8B Fine-Tuning
```bash
python3 train_qwen35.py \
  --train-file /mnt/d/LLM_OCR/dataset/train_manifest.json \
  --eval-file /mnt/d/LLM_OCR/dataset/eval_manifest.json \
  --image-root /mnt/d/LLM_OCR/dataset \
  --model-name /mnt/d/LLM_OCR/models/Qwen3.5-0.8B \
  --output-dir /mnt/d/LLM_OCR/outputs/qwen3.5-0.8b-cosmetic-ocr-lora \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --precision bf16
```

#### Option B: GLM-OCR Fine-Tuning
```bash
python3 train_glm_ocr.py \
  --train-file /mnt/d/LLM_OCR/dataset/train_manifest.json \
  --eval-file /mnt/d/LLM_OCR/dataset/eval_manifest.json \
  --image-root /mnt/d/LLM_OCR/dataset \
  --model-name /mnt/d/LLM_OCR/models/GLM-OCR \
  --output-dir /mnt/d/LLM_OCR/outputs/glm-ocr-cosmetic-ocr-lora \
  --epochs 10 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --precision bf16
```

#### Option C: Unified Trainer
```bash
python3 train_ocr.py \
  --train-file /mnt/d/LLM_OCR/dataset/train_manifest.json \
  --image-root /mnt/d/LLM_OCR/dataset \
  --model-name /mnt/d/LLM_OCR/models/Qwen3.5-0.8B \
  --model-type auto
```

---

### Step 3: Inference Benchmark & Evaluation

#### Evaluate Fine-Tuned Qwen3.5-0.8B LoRA Model:
```bash
python3 infer_ocr.py \
  --test-file /mnt/d/LLM_OCR/dataset/test_manifest.json \
  --image-root /mnt/d/LLM_OCR/dataset \
  --model-name /mnt/d/LLM_OCR/models/Qwen3.5-0.8B \
  --model-type qwen \
  --lora-dir /mnt/d/LLM_OCR/outputs/qwen3.5-0.8b-cosmetic-ocr-lora/checkpoint-96 \
  --max-pixels 401760 \
  --output-report /mnt/d/LLM_OCR/outputs/evaluation_report_qwen3.5_lora.md
```

#### Evaluate Raw Un-tuned GLM-OCR Base Model:
```bash
python3 infer_ocr.py \
  --test-file /mnt/d/LLM_OCR/dataset/test_manifest.json \
  --image-root /mnt/d/LLM_OCR/dataset \
  --model-name /mnt/d/LLM_OCR/models/GLM-OCR \
  --model-type glm \
  --max-pixels 401760 \
  --output-report /mnt/d/LLM_OCR/outputs/evaluation_report_glm_raw.md
```

---

### Step 4: Automated Shell Execution

Run complete training pipeline for Qwen3.5:
```bash
./run_train.sh qwen
```

Run testset benchmark evaluation for fine-tuned checkpoints:
```bash
./run_eval.sh finetuned
```
