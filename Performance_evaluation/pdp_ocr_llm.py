"""PDP copy extraction: PaddleOCR detection + LLM polishing.

Stage 1 - PaddleOCR pulls every text box; a height filter drops fine print
          (label ingredients, certificate body copy, footnote markers).
Stage 2 - Claude repairs what survives: fixes recognition errors, separates
          footnote superscripts that got merged into headlines, rejoins
          phrases split across boxes, and tags each block with a role.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python pdp_ocr_llm.py cetaphil.jpg
    python pdp_ocr_llm.py *.jpg --json out.json
    python pdp_ocr_llm.py cetaphil.jpg --no-llm      # raw OCR only
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

MIN_HEIGHT_RATIO = 0.018   # box height as a fraction of image height
MIN_CONF = 0.60            # kept low on purpose: the LLM repairs shaky reads
MODEL = "claude-sonnet-5"


def load_ocr(lang="ch"):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,                              # "ch" covers Chinese + English
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def extract_boxes(path, ocr):
    """Return prominent text boxes in reading order, with geometry kept."""
    img = np.array(Image.open(path).convert("RGB"))
    H, W = img.shape[:2]

    try:                                        # PaddleOCR 3.x
        res = ocr.predict(img)[0]
        raw = zip(res["rec_polys"], res["rec_texts"], res["rec_scores"])
    except AttributeError:                      # PaddleOCR 2.x
        page = ocr.ocr(img, cls=False)[0] or []
        raw = ((ln[0], ln[1][0], ln[1][1]) for ln in page)

    boxes = []
    for poly, text, conf in raw:
        poly = np.asarray(poly, dtype=float)
        xs, ys = poly[:, 0], poly[:, 1]
        h = ys.max() - ys.min()
        if h / H < MIN_HEIGHT_RATIO or conf < MIN_CONF:
            continue
        boxes.append({
            "text": text.strip(),
            "conf": round(float(conf), 3),
            "height_ratio": round(h / H, 4),
            "y_ratio": round(float(ys.min()) / H, 4),
            "x_ratio": round(float(xs.min()) / W, 4),
        })

    boxes.sort(key=lambda b: (b["y_ratio"], b["x_ratio"]))
    for i, b in enumerate(boxes):
        b["id"] = i
    return boxes


SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Cleaned copy, original language, no footnote digits.",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["headline", "sales_claim", "product_name",
                                 "award", "label", "other"],
                    },
                    "footnote_markers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Superscript markers stripped from this text, e.g. ['5'].",
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "OCR box ids merged into this block.",
                    },
                },
                "required": ["text", "role", "footnote_markers", "source_ids"],
            },
        },
        "dropped": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "reason"],
            },
        },
    },
    "required": ["blocks", "dropped"],
}


PROMPT = """\
These are OCR text boxes from an e-commerce product detail page (PDP), sorted
top to bottom. Geometry is normalised: height_ratio is the text height as a
fraction of image height, y_ratio/x_ratio are the top-left position.

Clean them up:
1. Fix obvious OCR recognition errors.
2. Superscript footnote markers are frequently merged into the text by OCR.
   Move them into footnote_markers. Example: "销量NO.15" -> text "销量NO.1",
   markers ["5"]. Be careful: do not strip digits that belong to the copy
   itself, such as "连续3年" or "40,000,000+".
3. Rejoin a phrase split across several boxes into one block, listing every
   contributing id in source_ids.
4. Tag each block with a role. Use "label" for text printed on the product
   package or packaging itself.
5. Drop pure noise (stray marks, single meaningless characters) via `dropped`.

Keep the original language. Do not translate, and do not invent copy that is
not present in the input.

OCR boxes:
%s
"""


def _extract_json(text):
    """Pull the first JSON object out of a model response."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in response")


def polish_llamacpp(boxes, server="http://127.0.0.1:8080", timeout=600):
    """Polish via a local llama-server. Works on the b4300-era HTTP API."""
    import urllib.error
    import urllib.request

    if not boxes:
        return {"blocks": [], "dropped": []}

    body = {
        "messages": [
            {"role": "system", "content":
             "You clean up OCR output. Reply with a single JSON object and nothing else."},
            {"role": "user", "content":
             PROMPT % json.dumps(boxes, ensure_ascii=False, indent=1)
             + "\nReply with JSON matching this schema:\n"
             + json.dumps(SCHEMA, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 4096,
        # Older builds ignore unknown fields, so asking for JSON is best-effort;
        # _extract_json covers the case where it comes back wrapped in prose.
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach llama-server at {server}. Start it with:\n"
            f"  ./llama-server -m model.gguf -c 8192 -t $(nproc) --host 127.0.0.1 --port 8080"
        ) from exc

    result = _extract_json(payload["choices"][0]["message"]["content"])
    result.setdefault("blocks", [])
    result.setdefault("dropped", [])
    for b in result["blocks"]:                      # local models skip fields
        b.setdefault("footnote_markers", [])
        b.setdefault("source_ids", [])
        b.setdefault("role", "other")
    return result


def polish(boxes, model=MODEL):
    """Send OCR boxes to Claude, get back cleaned + tagged blocks."""
    from anthropic import Anthropic

    if not boxes:
        return {"blocks": [], "dropped": []}

    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        tools=[{
            "name": "emit_pdp_copy",
            "description": "Return the cleaned PDP copy blocks.",
            "input_schema": SCHEMA,
        }],
        tool_choice={"type": "tool", "name": "emit_pdp_copy"},
        messages=[{
            "role": "user",
            "content": PROMPT % json.dumps(boxes, ensure_ascii=False, indent=1),
        }],
    )

    for block in msg.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("model did not call the tool")


def render(result):
    for b in result["blocks"]:
        marks = f'  [fn {",".join(b["footnote_markers"])}]' if b["footnote_markers"] else ""
        print(f'{b["role"]:<13} {b["text"]}{marks}')
    for d in result["dropped"]:
        print(f'  dropped #{d["id"]}: {d["reason"]}', file=sys.stderr)


def main():
    global MIN_HEIGHT_RATIO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--lang", default="ch")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min-height", type=float, default=MIN_HEIGHT_RATIO,
                    help="min box height as fraction of image height (default 0.018)")
    ap.add_argument("--json", metavar="PATH", help="write full results here")
    ap.add_argument("--no-llm", action="store_true", help="skip polishing")
    ap.add_argument("--backend", choices=["anthropic", "llamacpp"], default="anthropic",
                    help="llamacpp uses a local llama-server (offline, CPU)")
    ap.add_argument("--server", default="http://127.0.0.1:8080",
                    help="llama-server address for --backend llamacpp")
    args = ap.parse_args()

    MIN_HEIGHT_RATIO = args.min_height
    if not args.no_llm and args.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (use --backend llamacpp or --no-llm)")

    ocr = load_ocr(args.lang)
    results = []
    for path in args.images:
        print(f"\n=== {path} ===")
        boxes = extract_boxes(path, ocr)

        if args.no_llm:
            for b in boxes:
                print(f'{b["height_ratio"]:.3f}  {b["conf"]:.2f}  {b["text"]}')
            result = {"blocks": [], "dropped": []}
        elif args.backend == "llamacpp":
            result = polish_llamacpp(boxes, args.server)
            render(result)
        else:
            result = polish(boxes, args.model)
            render(result)

        result["image"] = path
        result["ocr_boxes"] = boxes
        results.append(result)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
