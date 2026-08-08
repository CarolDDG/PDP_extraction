#!/usr/bin/env bash
set -euo pipefail

# 批量跑 paddleOCR.py 并收集耗时/内存统计。
# 路径相对脚本位置，可用环境变量覆盖：
#   IMG_DIR=/path/to/images OUT_DIR=/path/to/out PYTHON=python3 ./runPaddle_bench.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

IMG_DIR="${IMG_DIR:-$SCRIPT_DIR/images}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/output}"
PYTHON="${PYTHON:-python3}"

JSON_DIR="$OUT_DIR/json"
STATS_FILE="$OUT_DIR/bench_stats.jsonl"

if [ ! -d "$IMG_DIR" ]; then
    echo "❌ 图片目录不存在: $IMG_DIR" >&2
    echo "   用 IMG_DIR=/your/images $0 指定" >&2
    exit 1
fi

mkdir -p "$JSON_DIR"
: > "$STATS_FILE"

find "$IMG_DIR" -maxdepth 1 -type f -name '*.png' | sort | while IFS= read -r line
do
    name="$(basename "$line" .png)"
    echo "$line"
    "$PYTHON" "$SCRIPT_DIR/paddleOCR.py" \
        --image "$line" \
        --output "$JSON_DIR/$name.json" \
        --stats "$STATS_FILE"
done
