# PDP Claim Extraction

Turning complex product-detail-page (PDP) graphics into consistent,
business-usable claim data.

PDP images mix product photos, marketing copy, numerical results, footnotes,
promo banners and decoration. Reading the text off them is only half the job —
the other half is deciding **where one claim ends and the next begins**, which is
a business rule, not an OCR rule.

Three approaches are implemented and measured against the same human-checked
ground truth:

| Approach | Directory | Capability | Trade-off |
|---|---|---|---|
| PaddleOCR | [PaddleOCR/](PaddleOCR/) | OCR baseline | fast and cheap, no claim-grouping logic |
| Qwen Agentic Workflow | [QwenAgenticWorkflow/](QwenAgenticWorkflow/) | visual + language understanding | good for prompt iteration, chunking can drift |
| LoRA fine-tuned | [Fine-Tunning/](Fine-Tunning/) | learns claim logic from ground truth | needs GPU and training data |

Scoring for all three lives in [Performance_evaluation/](Performance_evaluation/).

---

## The two-stage idea

Every approach here follows the same shape, because the two halves fail
differently and are worth separating:

```
PDP image
   │
   ├─ Stage 1  Literal OCR
   │           transcribe all visible text exactly as shown
   │           → raw regions, still in separate pieces
   │
   └─ Stage 2  Business-aware claim chunking
               group related regions into complete claim items
               → structured claim JSON
```

Stage 1 is a transcription problem. Stage 2 is a judgement problem: `连续三年销量第一`
and `72H锁水保湿` are two claims even if they sit on adjacent lines, while a
sentence broken across three lines by the layout is one claim. That is why the
evaluation reports two independent numbers rather than one blended score.

## Dataset

341 images: 311 core PDP images collected from 20 provided product URLs, plus 30
external images for generalization testing. Curation keeps images with
meaningful, consumer-facing claims that are complete and readable — product
benefits, efficacy results, ingredients, suitability, usage, awards,
certifications, sales claims. It drops pure product photos, blurry or cropped
text, duplicates, and store-promotion-only content (price, gift, coupon,
delivery).

Split 80/10/10 with a fixed seed: 249 train / 31 validation / 31 test, plus the
external set as a second, harder test.

```bash
python Fine-Tunning/split_dataset.py \
    --dataset-dir Fine-Tunning/dataset --split-mode 3way --seed 42
```

Manifests hold relative paths only (`images/Clarins_image18.png`), resolved
against `--image-root`. The images themselves are product photography and are not
committed — supply your own folder:

```
your-images/
  images/
    Clarins_image18.png
```

---

## 1. PaddleOCR baseline

Detection and recognition with no language model: PaddleOCR pulls text boxes,
the script rebuilds visual reading order, then a regex keeps only claim-like
lines. Fast, free, and a useful floor to measure against.

```bash
python PaddleOCR/paddleOCR.py --image poster.png --output result.json

# batch, with timing/memory stats
IMG_DIR=/path/to/images ./PaddleOCR/runPaddle_bench.sh
```

Inside [paddleOCR.py](PaddleOCR/paddleOCR.py) the stages are separate functions,
run in this order:

| Function | Does |
|---|---|
| `load_ocr(language, ocr_version)` | build the PaddleOCR instance |
| `extract_boxes(image, ocr)` | detected text → `TextBox` objects with geometry, handling both v2 and v3 result shapes |
| `filter_boxes(...)` | drop low-confidence and fine-print boxes |
| `group_into_lines(boxes)` | cluster boxes into visual lines by vertical overlap |
| `render_line(line)` | one line's boxes → a single string, left to right |
| `is_pdp_claim(text, brands)` | regex gate: keeps benefit/ingredient/measurement copy, drops prices, barcodes, net weight, spec fields |

`TextBox` carries `text`, `confidence` and the bounding box, with `width`,
`height` and `center_y` derived — that geometry is what the line grouping needs.
`is_pdp_claim` is where the business rules sit: `CLAIM_PATTERN` matches efficacy
verbs, ingredient names and measurements like `4周` or `+386%`, while
`EXCLUDED_PATTERN` vetoes prices, barcodes and net-weight fields first.

## 2. Qwen agentic workflow

> Contributor: [@BennyHz](https://github.com/BennyHz)

Two LLM passes wired together with CrewAI Flow. Pass 1 transcribes lines; pass 2
looks at the same image *plus* pass 1's line list and decides the grouping. The
second pass is where the business logic lives.

```bash
cd QwenAgenticWorkflow && python main_flow.py
```

The flow ([main_flow.py](QwenAgenticWorkflow/main_flow.py)) is two steps:

```python
step1_ocr_scan()                    # OCRAgent → ocr_results.json
step2_preprocess_and_consolidate()  # → consolidated_ocr_results.json
```

**Stage 1 — [`OCRAgent`](QwenAgenticWorkflow/agent/OCR.py)**

```python
from agent.OCR import OCRAgent

agent = OCRAgent(model_name="qwen-vl-max", max_workers=5)
results = agent.run("./images", return_json_str=False)
```

| Method | Does |
|---|---|
| `run(image_inputs, return_json_str=True)` | accepts a folder, single file, URL, or list; runs images concurrently across `max_workers` threads |
| `_collect_image_files(...)` | resolves that input into a file list, walking subdirectories |
| `_process_single_image(path)` | one image; **if more than 3 blank lines come back, re-prompts once** with a re-scan warning |
| `_normalize_parsed_result(...)` | flat string array → `{line_id, text}` records, dropping blanks and renumbering |
| `_call_ocr_api(...)` | the API call, plus JSON recovery: strip markdown fences → regex the JSON body → `json_repair` → strip control chars |

Output per image:

```json
{"file_name": "poster.png", "total_lines": 12,
 "lines": [{"line_id": 1, "text": "..."}]}
```

**Stage 2 — [`LayoutConsolidatorAgent`](QwenAgenticWorkflow/agent/Consolidator.py)**

```python
from agent.Consolidator import LayoutConsolidatorAgent

agent = LayoutConsolidatorAgent()
merged = agent.process_single_image("poster.png", lines)
```

`process_single_image(img_path, lines)` sends image + line list and gets back
merge groups. The prompt's hard rule is **zero omission**: every input `line_id`
must appear in some group's `original_line_ids`. Enforced in three tiers:

1. `_check_missing_line_ids(...)` diffs input IDs against covered IDs
2. anything missing triggers one retry naming the dropped IDs explicitly
3. still missing after that, they are appended as standalone single-line groups
   in Python, so nothing is ever silently lost

```json
{"file_name": "poster.png", "total_merged_lines": 2,
 "merged_lines": [
   {"merged_id": 1, "original_line_ids": [1, 2], "merged_text": "..."},
   {"merged_id": 2, "original_line_ids": [3],    "merged_text": "..."}]}
```

Tier 3 firing means the model needed help, so those images are worth a human
look. `simplify_ocr_results.py` finds them by scanning the run log and flags them:

```bash
python QwenAgenticWorkflow/utils/simplify_ocr_results.py \
    --src output/consolidated_ocr_results.json \
    --dst output/simplified.json --log run.log
```

It also flattens `merged_lines` down to a plain `lines: [str]` array and sorts by
filename, which is the shape the evaluator wants.

**Helpers and tests**

`prepare_consolidation_inputs(ocr_results, image_inputs)` sits between the two
agents, pairing each stage-1 result with its image path.

```bash
cd QwenAgenticWorkflow
pytest tests/test_ocr_agent.py       # structure, blank filtering, contiguous IDs
pytest tests/test_consolidator.py    # zero-omission guarantee
python tests/test_prompt_ab.py       # bare prompt vs constrained prompt
python tests/test_ocr_model_compare.py   # qwen3.5-ocr vs qwen-vl-max
```

Tests look for images in `QwenAgenticWorkflow/images/`.

## 3. LoRA fine-tuning

> Contributor: [@BennyHz](https://github.com/BennyHz)

The prompt-based approaches have to be *told* the chunking rules every call. A
fine-tuned model learns them from the ground truth instead.

Hardware budget was one 12 GB RTX 4070, which rules out full fine-tuning and
rules out large models. LoRA freezes the base weights and trains small adapter
matrices, cutting trainable parameters and VRAM footprint enough to fit.

Model selection ran a trial SFT over four candidates — GLM-OCR (1B),
Qwen3.5-0.8B, Qwen2.5-Omni (3B), DeepSeek-OCR2 (3B) — scored on native Chinese
pretraining, visual modality support, and size. **GLM-OCR (1B)** and
**Qwen3.5-0.8B** made the shortlist and got the full pipeline.

```bash
# model type auto-detected from --model-name
python Fine-Tunning/train_ocr.py \
    --train-file Fine-Tunning/dataset/train_manifest.json \
    --eval-file  Fine-Tunning/dataset/eval_manifest.json \
    --image-root /path/to/your-images \
    --model-name Qwen/Qwen3.5-0.8B \
    --epochs 3 --batch-size 1 --gradient-accumulation-steps 8

./Fine-Tunning/run_train.sh qwen     # or: glm, all
```

[train_ocr.py](Fine-Tunning/train_ocr.py) is a thin dispatcher: it reads
`--model-name`, picks the architecture, fills in per-model defaults, and hands
off to `train_qwen35(config)` or `train_glm_ocr(config)`.

| Flag | Default | Note |
|---|---|---|
| `--model-type` | `auto` | `auto` matches "glm"/"qwen" in the model name |
| `--epochs` | 3.0 qwen / 10.0 glm | picked from the loss curves |
| `--max-pixels` | 1048576 qwen / 401760 glm | image budget, the main VRAM lever |
| `--precision` | `auto` | bf16 where supported, else fp16 |
| `--gradient-accumulation-steps` | 8 | effective batch 8 at batch-size 1 |
| `--resume-from-checkpoint` | — | path to a checkpoint folder |

Both trainers take a config dataclass (`Qwen35TrainConfig`, `GLMOCRTrainConfig`)
with the same fields, so you can call them directly:

```python
from train_qwen35 import train_qwen35, Qwen35TrainConfig, DEFAULT_PROMPT

metrics = train_qwen35(Qwen35TrainConfig(
    train_file="dataset/train_manifest.json",
    image_root="/path/to/your-images",
    output_dir="outputs/qwen-lora",
    model_name="Qwen/Qwen3.5-0.8B",
    prompt=DEFAULT_PROMPT,
    epochs=3.0, lora_rank=16,
))
```

Inference loads the base model and applies the adapter on top:

```bash
python Fine-Tunning/infer_ocr.py \
    --test-file Fine-Tunning/dataset/test_manifest.json \
    --image-root /path/to/your-images \
    --model-name Qwen/Qwen3.5-0.8B --model-type qwen \
    --lora-dir outputs/qwen3.5-0.8b-cosmetic-ocr-lora \
    --output-report report.md

./Fine-Tunning/run_eval.sh all    # raw + fine-tuned, both models
```

`load_model_and_processor(model_name, model_type, lora_dir, max_pixels)` resolves
dtype and device, loads base weights, and wraps them with `PeftModel` when
`lora_dir` is given. Drop the flag to score the raw base model — that comparison
is the point.

Every run is scored on **both** the internal test split and the external set, so
you can see the gap between in-distribution and unseen brands. Eight reports
(2 models × raw/LoRA × internal/external) are in
[Fine-Tunning/results/](Fine-Tunning/results/).

---

## Evaluation

Scoring compares a list of predicted claims against a list of annotated claims.
Neither side has IDs, so before anything can be measured the two lists have to be
**aligned** — and a model that fuses two claims or splits one is exactly the case
where alignment is ambiguous.

Alignment runs cheapest-first, so the expensive judge only sees what the cheap
checks could not settle:

```
1  exact match after normalization        free
2  structural groups (merge / split)      free       ← one↔many
3  similarity ≥ 0.90                      free       ← one↔one
4  LLM judge on the 0.55–0.90 band        paid       ← ambiguous only
```

Steps 1–3 are local string work. Step 4 sends only the leftover ambiguous pairs
to an external LLM (DeepSeek), which keeps cost down and, being a different model
family from the ones under test, avoids grading a model with itself.

### Two independent numbers

Once aligned, two things get measured separately, because they are different
failures with different fixes.

**OCR accuracy — did it read the characters right?**

```
              Σ char_errors
CER  =  ─────────────────────       accumulate integers, divide once
              Σ char_length
```

| Source | → numerator | → denominator |
|---|---|---|
| matched pair / group | `levenshtein(gt, pred)` | `len(gt)` |
| missing annotation | `len(gt)` | `len(gt)` |
| extra prediction | `len(pred)` | — |

Extras enter the numerator only, so **CER can exceed 1** — a model that
hallucinates 300 characters onto a 27-character poster scores far above 1.0, and
that is the intended reading. Reported alongside: WER on the same logic over
tokens, plus precision (does it emit garbage?) and recall (does it find every
claim?).

**Chunking accuracy — did it cut in the right places?**

```
                    whole
Acc  =  ───────────────────────────      counted on the prediction side
          whole + merged + split
```

A **merge** costs 1 bad chunk however many claims it fused — the wording is all
there, one boundary is missing. A **split** costs N — every fragment is half a
claim, unusable downstream. Counting the prediction side rather than the
annotation side is what makes that asymmetry fall out without a weight parameter.

Granularity errors are charged here and **not** to CER. Text that is
character-perfect but cut in two scores `CER 0.00` with the chunking penalty
taking the hit:

```
GT   ['连续三年销量第一', '72H锁水保湿']
PRED ['连续三年销量第一 72H锁水保湿']

CER 0.00   recall 1.00   ·   chunking 0.00
```

Every character right, zero usable chunks. One metric per failure mode, so the
numbers tell you which half to go fix.

### Three axes

Accuracy is not the only thing that matters in production, so each approach is
also scored on **inference speed** and **cost per image**, min-max normalized for
comparison. PaddleOCR wins on speed and cost and loses on chunking; the API
approaches invert that; the fine-tuned models sit in between and are the only ones
that learned the claim logic rather than being told it.

### Running the evaluator

[eval_ocr_with_llm.py](Performance_evaluation/eval_ocr_with_llm.py) scores one
prediction file against the annotations. Both arguments are required; everything
else has a working default.

```bash
# local metrics only, no API calls
python Performance_evaluation/eval_ocr_with_llm.py \
  --ground-truth gt.json --prediction pred.json --no-llm-judge

# full four-pass alignment, judge decides the uncertain band
export DEEPSEEK_API_KEY=...
python Performance_evaluation/eval_ocr_with_llm.py \
  --ground-truth gt.json --prediction pred.json \
  --output outputs/qwen_internal.json
```

`--no-llm-judge` and `--llm-judge` are mutually exclusive: the first forces the
judge off and keeps the run offline and free, the second forces it on. Without
either, the config decides (default: on).

Input JSON is a list of records. Field names are read in order and the first one
present wins, so most formats work unchanged — file from `source_file` /
`file_name` / `image`, text from `claim_text` / `lines` / `text` / `prediction`.
Text may be a list of regions or a newline-separated string; `<EMPTY>` means a
page with no claims. Only files the prediction mentions are scored — annotations
it never speaks to land in `summary.unpredicted_files` instead of counting as
all-missing.

| Flag | Use |
|---|---|
| `--output PATH` | where the report goes (default `outputs/ocr_eval.json`) |
| `--detail` | add per-record missing/extra lists |
| `--debug` | full pairing trace: every pair, its similarity, why unmatched regions failed |
| `--verbose` | print each judged pair as it resolves |
| `--group-similarity` | how completely fragments must reconstruct the whole (default 0.85) |
| `--no-cache` | ignore the verdict cache and re-judge |
| `--prompt-file` | swap in a different judge prompt |
| `--no-merge-credit`<br>`--no-split-credit` | make CER charge a granularity error a second time, as missing plus extra |

The two credit flags are off-by-default escape hatches. Leave them alone unless
CER should mean "delivered as a usable claim" rather than "read the characters
correctly" — neither affects chunking accuracy, which always reports every group
it found. Judge verdicts are cached by content hash, so the same pair resolves
identically across records and reruns.

`--config` points at a TOML file that overrides any default (thresholds, field
aliases, normalization). It is optional: a missing file falls back to the
built-in defaults rather than erroring, so the commands above run as-is.

Stdout is the summary block — `cer`, `wer`, `precision`, `recall`, `f1`,
`chunking_accuracy`, plus the raw counts behind them
(`chunks_whole` / `chunks_merged` / `chunks_split`, `merges`, `splits`). The file
at `--output` carries the same summary plus per-record `items`.

[notebook.ipynb](Performance_evaluation/notebook.ipynb) is the visualization
layer. It reads the JSON reports and draws the comparison charts — per-model
accuracy bars, score distributions, and the speed/cost axes — split into internal
and external test set sections. Analysis only; it computes no metrics of its own.

---

## Setup

```bash
# OCR baseline
pip install paddlepaddle paddleocr pillow numpy

# agentic workflow
pip install crewai openai pydantic python-dotenv json-repair pytest

# fine-tuning (CUDA)
pip install torch transformers peft trl datasets accelerate pillow psutil pynvml

# visualization
pip install pandas matplotlib seaborn notebook
```

The evaluator itself needs no third-party packages — it runs on the standard
library alone.

Keys come from environment variables — copy `.env.example` to
`QwenAgenticWorkflow/.env` and fill in:

```bash
DASHSCOPE_API_KEY=...     # Qwen-VL, DashScope OpenAI-compatible endpoint
DEEPSEEK_API_KEY=...      # evaluation judge
```

The judge also accepts `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`, or the
`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `EVAL_MODEL` aliases, and `--api-key` /
`--base-url` / `--model` override both. With `--no-llm-judge` no key is needed
at all.

`.env` is gitignored. Confirm before your first push:

```bash
git status --porcelain | grep -i env    # expect no output
```

## Layout

```
PaddleOCR/
  paddleOCR.py           detection → reading-order rebuild → claim regex
  runPaddle_bench.sh     batch driver with timing/memory stats
  runtime_env.py         device/precision helpers

QwenAgenticWorkflow/
  main_flow.py           CrewAI Flow: OCR → preprocess → consolidate
  agent/OCR.py           stage 1, concurrent line extraction, blank-line retry
  agent/Consolidator.py  stage 2, layout-aware merge, zero-omission guarantee
  utils/                 payload prep, log parsing, output simplification
  tests/                 zero-omission, prompt A/B, model comparison

Fine-Tunning/
  train_ocr.py           unified entry, architecture auto-detect
  train_qwen35.py        Qwen3.5-0.8B LoRA
  train_glm_ocr.py       GLM-OCR LoRA
  infer_ocr.py           inference + markdown evaluation report
  split_dataset.py       80/10/10, seed 42
  dataset/               249 / 31 / 31 manifests (relative paths)
  results/               8 reports: 2 models × raw/LoRA × internal/external

Performance_evaluation/
  eval_ocr_with_llm.py   the evaluator: four-pass alignment, CER/WER/P/R/F1,
                         chunking accuracy, DeepSeek judge (stdlib only)
  notebook.ipynb         visualization: reads the reports, draws the charts
  README.md              every metric formula in full
```
