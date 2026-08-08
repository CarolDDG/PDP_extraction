"""Extract PDP claims from a cosmetics poster with PaddleOCR.

PaddleOCR performs text detection and recognition. This script then rebuilds
the visual reading order and applies local rules for PDP claims, so it does not
require an API or language model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable



CLAIM_PATTERN = re.compile(
    r"(?:"
    r"淡化|改善|修护|修复|保湿|补水|紧致|抗皱|抗老|抗氧|抗醣|抗糖|提亮|"
    r"美白|焕亮|舒缓|控油|祛痘|减少|降低|提升|增强|保护|改善|促进|"
    r"细纹|皱纹|毛孔|弹性|屏障|损伤|胶原|饱满|光泽|透亮|柔嫩|平滑|"
    r"净澈|匀净|添加|蕴含|富含|配方|成分|玻尿酸|透明质酸|烟酰胺|"
    r"甘草酸|维生素|视黄醇|胜肽|神经酰胺|水杨酸|果酸|精华|"
    r"测试|实验|实证|验证|认证|临床|真人|消费者|使用前|使用后|"
    r"区域|荧光|基底膜|真表皮|肤质|肌肤|皮肤|清爽|滋润|不黏腻|"
    r"\d+(?:[.,]\d+)?\s*[%％倍小时天周月]"
    r")"
)

EXCLUDED_PATTERN = re.compile(
    r"(?:"
    r"^[¥￥]\s*\d|"
    r"^\d{8,}$|"
    r"\b\d+(?:\.\d+)?\s*(?:ml|mL|ML|g|kg|克|千克|毫升)\b|"
    r"条形码|净含量|规格|生产许可证|执行标准|保质期"
    r")"
)


@dataclass(frozen=True)
class TextBox:
    text: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2


@dataclass
class TextLine:
    boxes: list[TextBox]

    @property
    def y_min(self) -> float:
        return min(box.y_min for box in self.boxes)

    @property
    def y_max(self) -> float:
        return max(box.y_max for box in self.boxes)

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


def load_ocr(language: str, ocr_version: str) -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit("未安装 PaddleOCR，请在已配置好的环境中运行该脚本。") from exc

    try:
        return PaddleOCR(
            lang=language,
            ocr_version=ocr_version,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(lang=language, use_angle_cls=True)
    except ValueError as exc:
        if "Unknown argument" not in str(exc):
            raise
        return PaddleOCR(lang=language)


def _result_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result
    else:
        data = getattr(result, "json", None)
        if callable(data):
            data = data()
        if not isinstance(data, dict):
            try:
                return {
                    "rec_polys": result["rec_polys"],
                    "rec_texts": result["rec_texts"],
                    "rec_scores": result["rec_scores"],
                }
            except (KeyError, TypeError) as exc:
                raise ValueError("无法解析 PaddleOCR 3.x 的输出格式") from exc

    nested = data.get("res")
    return nested if isinstance(nested, dict) else data


def _boxes_from_v3(image: Any, ocr: Any) -> list[TextBox]:
    pages = list(ocr.predict(image))
    boxes: list[TextBox] = []
    for page in pages:
        data = _result_mapping(page)
        polygons = data.get("rec_polys")
        if polygons is None:
            polygons = data.get("dt_polys")
        texts = data.get("rec_texts")
        scores = data.get("rec_scores")
        polygons = [] if polygons is None else polygons
        texts = [] if texts is None else texts
        scores = [] if scores is None else scores
        for polygon, text, confidence in zip(polygons, texts, scores):
            boxes.append(_make_box(polygon, text, confidence))
    return boxes


def _boxes_from_v2(image: Any, ocr: Any) -> list[TextBox]:
    pages = ocr.ocr(image, cls=True) or []
    if pages and _looks_like_v2_line(pages[0]):
        pages = [pages]

    boxes: list[TextBox] = []
    for page in pages:
        for line in page or []:
            polygon, recognition = line
            text, confidence = recognition
            boxes.append(_make_box(polygon, text, confidence))
    return boxes


def _looks_like_v2_line(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) == 2
        and isinstance(value[1][0], str)
    )


def _make_box(polygon: Any, text: Any, confidence: Any) -> TextBox:
    import numpy as np

    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    return TextBox(
        text=str(text).strip(),
        confidence=float(confidence),
        x_min=float(points[:, 0].min()),
        y_min=float(points[:, 1].min()),
        x_max=float(points[:, 0].max()),
        y_max=float(points[:, 1].max()),
    )


def extract_boxes(image: Any, ocr: Any) -> list[TextBox]:
    if hasattr(ocr, "predict"):
        try:
            return _boxes_from_v3(image, ocr)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return _boxes_from_v2(image, ocr)


def filter_boxes(
    boxes: Iterable[TextBox],
    image_height: int,
    min_confidence: float,
    min_height_ratio: float,
) -> list[TextBox]:
    return [
        box
        for box in boxes
        if box.text
        and box.confidence >= min_confidence
        and box.height / image_height >= min_height_ratio
    ]


def _vertical_overlap(first: TextBox, second: TextLine) -> float:
    overlap = max(0.0, min(first.y_max, second.y_max) - max(first.y_min, second.y_min))
    return overlap / max(1.0, min(first.height, second.height))


def group_into_lines(boxes: Iterable[TextBox]) -> list[TextLine]:
    lines: list[TextLine] = []
    for box in sorted(boxes, key=lambda item: (item.center_y, item.x_min)):
        candidates = [
            line
            for line in lines
            if _vertical_overlap(box, line) >= 0.45
            or abs(box.center_y - line.center_y) <= 0.35 * max(box.height, line.height)
        ]
        if candidates:
            line = min(candidates, key=lambda item: abs(box.center_y - item.center_y))
            line.boxes.append(box)
        else:
            lines.append(TextLine([box]))

    for line in lines:
        line.boxes.sort(key=lambda item: item.x_min)
    return sorted(lines, key=lambda item: (item.center_y, item.boxes[0].x_min))


def render_line(line: TextLine) -> str:
    output = ""
    previous: TextBox | None = None
    for box in line.boxes:
        if previous is not None:
            gap = box.x_min - previous.x_max
            if gap > 0.8 * max(previous.height, box.height):
                output += " "
        output += box.text
        previous = box
    return output.strip()


def is_pdp_claim(text: str, brands: Iterable[str]) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact or EXCLUDED_PATTERN.search(compact):
        return False
    if any(compact.casefold() == brand.casefold() for brand in brands):
        return False
    return bool(CLAIM_PATTERN.search(compact))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="input image path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output json file path",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="optional JSONL stats file path",
    )
    parser.add_argument("--lang", default="ch", help="PaddleOCR language, default: ch")
    parser.add_argument(
        "--ocr-version",
        choices=["PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"],
        default="PP-OCRv4",
        help="model family; PP-OCRv4 is compatible with Paddle 3.0 on macOS",
    )
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument(
        "--min-height-ratio",
        type=float,
        default=0.008,
        help="minimum OCR box height divided by image height",
    )
    parser.add_argument(
        "--brand",
        action="append",
        default=["SkinCeuticals", "修丽可"],
        help="exact brand/logo text to exclude; may be repeated",
    )
    parser.add_argument("--all-text", action="store_true", help="output all OCR lines")
    parser.add_argument("--debug", action="store_true", help="print OCR boxes to stderr")
    return parser.parse_args()


def _get_rusage() -> dict[str, float]:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        max_rss_mb = max_rss / (1024.0 * 1024.0)
    else:
        max_rss_mb = max_rss / 1024.0
    return {
        "cpu_user_seconds": float(usage.ru_utime),
        "cpu_sys_seconds": float(usage.ru_stime),
        "max_rss_mb": max_rss_mb,
    }


def _append_stats(stats_path: Path, payload: dict[str, Any]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_output(image_path: Path, lines: list[str]) -> dict[str, Any]:
    return {
        "source_file": image_path.name,
        "claim_text": lines,
    }


def main() -> None:
    args = parse_args()
    started_wall = time.perf_counter()
    started_ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    started_usage = _get_rusage()

    import numpy as np
    from PIL import Image

    if not args.image.is_file():
        raise SystemExit(f"图片不存在：{args.image}")

    image = np.asarray(Image.open(args.image).convert("RGB"))
    ocr = load_ocr(args.lang, args.ocr_version)
    boxes = extract_boxes(image, ocr)
    boxes = filter_boxes(
        boxes,
        image_height=image.shape[0],
        min_confidence=args.min_confidence,
        min_height_ratio=args.min_height_ratio,
    )

    if args.debug:
        for box in sorted(boxes, key=lambda item: (item.y_min, item.x_min)):
            print(
                f"conf={box.confidence:.3f} "
                f"box=({box.x_min:.0f},{box.y_min:.0f},{box.x_max:.0f},{box.y_max:.0f}) "
                f"text={box.text}",
                file=sys.stderr,
            )

    lines = [render_line(line) for line in group_into_lines(boxes)]
    if not args.all_text:
        lines = [line for line in lines if is_pdp_claim(line, args.brand)]

    output_data = _build_output(args.image, lines)
    if args.output is None:
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.stats is not None:
        ended_wall = time.perf_counter()
        ended_usage = _get_rusage()
        cpu_user = max(0.0, ended_usage["cpu_user_seconds"] - started_usage["cpu_user_seconds"])
        cpu_sys = max(0.0, ended_usage["cpu_sys_seconds"] - started_usage["cpu_sys_seconds"])
        cpu_total = cpu_user + cpu_sys
        wall = max(1e-9, ended_wall - started_wall)
        _append_stats(
            args.stats,
            {
                "timestamp": started_ts,
                "image": str(args.image),
                "output": None if args.output is None else str(args.output),
                "ocr_version": args.ocr_version,
                "lang": args.lang,
                "all_text": args.all_text,
                "wall_seconds": round(wall, 6),
                "cpu_user_seconds": round(cpu_user, 6),
                "cpu_sys_seconds": round(cpu_sys, 6),
                "cpu_total_seconds": round(cpu_total, 6),
                "cpu_percent_estimate": round((cpu_total / wall) * 100.0, 2),
                "max_rss_mb": round(ended_usage["max_rss_mb"], 2),
                "line_count": len(lines),
            },
        )


if __name__ == "__main__":
    main()
