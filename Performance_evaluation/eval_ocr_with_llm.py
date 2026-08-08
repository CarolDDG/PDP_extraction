"""Evaluate OCR predictions against ground truth.

Pipeline
--------
1. Load both JSON files and normalise every text region.
2. Align regions, order-independently, in four passes: exact matches, then
   structural groups (see below), then high-similarity matches, then the
   uncertain band.
3. Only genuinely uncertain pairs reach the LLM. Each pair is one
   independent request, and the verdict is cached by content hash so the
   same pair always resolves the same way across records and runs.
4. Report two independent dimensions:
     OCR accuracy      CER, WER, precision, recall, F1
     Chunking accuracy were the rows cut where the business rules say

   They are reported apart because one can be perfect while the other is
   zero: a clean whitespace merge reads every character correctly (CER 0)
   yet delivers one claim where three were expected (chunking 0).

Only files present in the prediction are scored. Ground-truth files the
prediction never mentions are skipped, not counted as all-missing, and
reported as summary.unpredicted_files.

Granularity errors belong to chunking, not to CER
-------------------------------------------------
A MERGE is several annotated rows arriving as one prediction; a SPLIT is one
annotated row arriving as several. Both are boundary mistakes, not reading
mistakes, so CER charges neither of them for the boundary: the group is scored
on its concatenation and pays only the characters it actually got wrong. A
clean whitespace merge therefore costs nothing, and a split that read every
character correctly also costs nothing.

Charging CER for a split would mean reporting "not one character correct" for
a prediction that in fact read all of them, only in two pieces. That is what
chunking accuracy is for, and it is where the asymmetry lives: counted on the
prediction side, a merge is one badly cut chunk however many rows it
swallowed, while a split contributes one per fragment. Fusing 10 rows costs
1; breaking 1 row into 4 costs 4.

align.merge_credit and align.split_credit (both true by default) can be
switched off to make CER charge that error a second time - the rows land in
missing and extra, so the text counts as both deleted and inserted. Use that
only when CER should mean "delivered as a usable claim" rather than "read the
characters correctly". Neither flag affects chunking accuracy, which always
reports every group it found.

Every metric formula is documented in README.md.

Judge credentials come from env vars:
DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
OPENAI_API_KEY / OPENAI_BASE_URL / EVAL_MODEL are accepted as aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import socket
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib import error, request

JUDGE_PROMPT = """你是 OCR 评估中的对齐裁判。给你两段文本：一段来自人工标注，一段来自 OCR 预测。
你只需判断它们是否指向同一处文字区域（即预测是这段标注的识别结果，可能存在识别错误）。

判为 same=true 的情况：
1. 文字主体相同，只是个别字识别错误、缺字、多字。
2. 只差上标、括号编号、标点、空格、大小写、全半角。
3. 预测把这段标注**完整地包含**进去了，只是同时还带上了别的文字（合并）。
标注的内容一个不少，只是没有被切开，算识别到了。

判为 same=false 的情况：
1. 指向画面上两处不同的文字。
2. 只是碰巧共享通用词（如"使用前""受试者""%"），主体内容不同。
3. 数字或指标不同，代表不同的一条数据。
4. 预测只是这段标注的**一个片段**，标注剩下的意思没了（拆分）。
一条标注是一个完整的意思，只认出半句不算认出来。

第 3 条和第 4 条的区别只看方向：预测比标注多 = true，预测比标注少 = false。

只输出 JSON，不要 Markdown，不要解释：
{"same": true 或 false}
"""

DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        # Field names are read in order; the first one present wins.
        "file_keys": ["source_file", "file_name", "image"],
        "text_keys": ["claim_text", "lines", "text", "prediction"],
        "empty_token": "<EMPTY>",
    },
    "normalize": {
        # Applied before any comparison. Metrics are computed on these forms.
        "ignore_whitespace": True,
        "case_sensitive": False,
        "unicode_nfkc": True,      # full-width （5） and half-width (5) compare equal
        "strip_punctuation": False,
    },
    "align": {
        # Similarity is difflib ratio on normalised text, in [0, 1].
        "auto_match": 0.90,        # >= this: matched locally, no API call
        "reject_below": 0.55,      # <  this: never a candidate, no API call
        # The band in between is what the LLM decides.

        # Granularity errors, detected structurally before any similarity pass.
        # Granularity is chunking's business; CER charges only real character
        # errors. Setting either to False makes CER charge the error twice.
        "merge_credit": True,      # several GT rows in one prediction: still a match
        "split_credit": True,      # one GT row in several predictions: still a match
        "group_similarity": 0.85,  # concatenation must reach this to be a group
        "min_group_part": 2,       # a part shorter than this is never absorbed
        "group_tolerance": 0.25,   # per-row slack, so one misread char still groups
    },
    "judge": {
        "enabled": True,
        "cache_file": "benchmark/.judge_cache.json",
        "temperature": 0,
        "seed": 20260805,          # sent when the provider honours it
        "timeout": 120,
        "retries": 3,
        "response_format_json": True,
        "on_failure": "auto",      # auto | reject | error
    },
    "output": {
        "detail": False,           # true adds per-record missing/extra lists
        "debug": False,            # true adds the full pairing trace with scores
    },
}

def _coalesce_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_toml_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_parse_toml_scalar(p.strip()) for p in inner.split(",")] if inner else []
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    for module_name in ("tomllib", "tomli"):
        try:
            module = __import__(module_name)
        except ModuleNotFoundError:
            continue
        return module.loads(text)

    # Minimal fallback parser for Python 3.8-3.10 without tomli installed.
    config: dict[str, Any] = {}
    section: dict[str, Any] = config
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = config
            for part in line[1:-1].split("."):
                section = section.setdefault(part.strip(), {})
        elif "=" in line:
            key, value = line.split("=", 1)
            section[key.strip()] = _parse_toml_scalar(value)
    return config


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"config file does not exist: {config_path}")
    return _deep_merge(DEFAULT_CONFIG, _load_toml(config_path))


def _first_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _regions_from_value(value: Any, empty_token: str) -> list[str]:
    """Split one record's text field into a list of regions."""
    if isinstance(value, list):
        regions = [str(v).strip() for v in value]
    elif isinstance(value, str):
        regions = [] if value.strip() == empty_token else value.splitlines()
        regions = [r.strip() for r in regions]
    elif value is None:
        regions = []
    else:
        regions = [str(value).strip()]
    return [r for r in regions if r and r != empty_token]


def _load_records(path: str) -> list[Any]:
    """Read a JSON array, a JSONL stream, or an object wrapping an items list."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Not one JSON value. Try JSONL: one object per line. Stray brackets and
        # trailing commas are tolerated so a half-array file still loads.
        rows: list[Any] = []
        for number, line in enumerate(raw.splitlines(), start=1):
            line = line.strip().rstrip(",")
            if not line or line in {"[", "]"}:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} is neither JSON nor JSONL; line {number} failed: {exc}"
                ) from exc
        if not rows:
            raise ValueError(f"{path} contains no records")
        return rows

    if isinstance(parsed, dict):
        return parsed["items"] if isinstance(parsed.get("items"), list) else [parsed]
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"{path} must be a JSON array, JSONL, or an object with an items list")


def load_ocr_json(path: str, config: dict[str, Any]) -> dict[str, list[str]]:
    input_config = config["input"]
    rows = _load_records(path)

    data: dict[str, list[str]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"record {index} in {path} must be an object")
        file_name = _first_value(row, input_config["file_keys"])
        if not isinstance(file_name, str) or not file_name:
            keys = ", ".join(input_config["file_keys"])
            raise ValueError(f"record {index} in {path} needs one of: {keys}")
        value = _first_value(row, input_config["text_keys"])
        if value is None:
            keys = ", ".join(input_config["text_keys"])
            raise ValueError(f"record {index} in {path} needs one of: {keys}")
        data[Path(file_name).name] = _regions_from_value(value, input_config["empty_token"])
    return data


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str, config: dict[str, Any]) -> str:
    """Canonical form used for every comparison and every metric."""
    norm_config = config["normalize"]
    if _coerce_bool(norm_config["unicode_nfkc"]):
        # NFKC folds full-width to half-width, so （5） == (5).
        text = unicodedata.normalize("NFKC", text)
    if _coerce_bool(norm_config["strip_punctuation"]):
        text = _PUNCT.sub("", text)
    text = text.strip()
    if _coerce_bool(norm_config["ignore_whitespace"]):
        text = re.sub(r"\s+", "", text)
    else:
        text = re.sub(r"\s+", " ", text)
    if not _coerce_bool(norm_config["case_sensitive"]):
        text = text.casefold()
    return text


_TOKEN = re.compile(r"[a-zA-Z]+|[0-9][0-9.,%]*|[^\sa-zA-Z0-9]")


def tokenize(text: str) -> list[str]:
    """Tokens for WER.

    Chinese has no spaces, so splitting on whitespace would make a whole
    region a single token and WER would collapse to 0 or 1. Instead: runs of
    Latin letters are one token, runs of digits (with . , %) are one token,
    and every other character (each CJK glyph, each symbol) is its own token.
    """
    return _TOKEN.findall(text)


def _similarity(first: str, second: str) -> float:
    import difflib

    if first == second:
        return 1.0
    return difflib.SequenceMatcher(None, first, second).ratio()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"judge did not return JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


def _chat_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if not re.search(r"/v\d+$", base_url):
        base_url += "/v1"
    return base_url + "/chat/completions"


class JudgeClient:
    """One request per pair, cached by content hash.

    Consistency comes from three things: temperature 0, a cache keyed on the
    normalised pair (so the same pair anywhere in the dataset gets the same
    verdict), and a prompt that carries no other record's context.
    """

    def __init__(self, config: dict[str, Any], *, api_key: str, base_url: str,
                 model: str, prompt: str, verbose: bool = False) -> None:
        judge_config = config["judge"]
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt = prompt
        self.verbose = verbose
        self.temperature = judge_config["temperature"]
        self.seed = judge_config["seed"]
        self.timeout = int(judge_config["timeout"])
        self.retries = int(judge_config["retries"])
        self.response_format_json = _coerce_bool(judge_config["response_format_json"])
        self.on_failure = str(judge_config["on_failure"])
        self.cache_path = Path(judge_config["cache_file"]) if judge_config["cache_file"] else None
        self.cache: dict[str, bool] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.cache = {}
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.last_source = ""   # "api" | "cache" | "fallback", for the debug trace

    def _cache_key(self, gt: str, pred: str) -> str:
        payload = json.dumps([self.model, self.prompt, gt, pred], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=0, sort_keys=True),
            encoding="utf-8",
        )

    def is_same(self, gt: str, pred: str, *, fallback: bool) -> bool:
        """Return True when the judge says the two regions are the same text."""
        key = self._cache_key(gt, pred)
        if key in self.cache:
            self.cache_hits += 1
            self.last_source = "cache"
            return self.cache[key]
        try:
            verdict = self._request(gt, pred)
        except RuntimeError as exc:
            self.failures += 1
            if self.on_failure == "error":
                raise
            verdict = False if self.on_failure == "reject" else fallback
            self.last_source = "fallback"
            if self.verbose:
                print(f"  [judge failed, using {verdict}] {exc}", flush=True)
            return verdict
        self.last_source = "api"
        self.cache[key] = verdict
        self.save_cache()
        return verdict

    def _request(self, gt: str, pred: str) -> bool:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"标注文本：\n{gt}\n\n预测文本：\n{pred}"},
            ],
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                req = request.Request(
                    _chat_url(self.base_url), data=body, headers=headers, method="POST"
                )
                with request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                self.calls += 1
                parsed = _extract_json(result["choices"][0]["message"]["content"])
                return _coerce_bool(parsed["same"])
            except error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except OSError:
                    detail = str(exc)
                last_error = RuntimeError(f"HTTP {exc.code}: {detail[:400]}")
            except (
                error.URLError,
                http.client.HTTPException,
                TimeoutError,
                socket.timeout,
                KeyError,
                IndexError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(2 * attempt)
        raise RuntimeError(f"judge failed after {self.retries} attempts: {last_error}")


def _infix_distance(needle: str, haystack: str) -> tuple[int, int]:
    """Cheapest edit distance from `needle` to any substring of `haystack`.

    Returns (distance, end offset of the best window). Ordinary Levenshtein
    with a free start (row 0 is all zeros) and a free end (answer is the row
    minimum), which is the standard approximate-substring alignment.

    Needed because strict containment breaks on a single OCR error: the row
    "72H（2）锁水保湿" does not literally occur inside "72h锁水保湿 深层补水",
    yet it is plainly the same row merged with its neighbour.
    """
    if not needle:
        return 0, 0
    if not haystack:
        return len(needle), 0
    previous = [0] * (len(haystack) + 1)
    for i, nch in enumerate(needle, start=1):
        current = [i]
        for j, hch in enumerate(haystack, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (nch != hch),
                )
            )
        previous = current
    best = min(previous)
    return best, previous.index(best)


def _find_groups(
    container_norm: list[str],
    part_norm: list[str],
    free_container: set[int],
    free_part: set[int],
    *,
    threshold: float,
    min_part: int,
    join: str,
    tolerance: float = 0.0,
    floor: int = 3,
) -> list[tuple[int, list[int], float]]:
    """Find one-container-to-many-parts groups by substring containment.

    Returns (container_index, [part indices in container order], similarity).

    Deterministic and API-free: a part belongs to a container only if its
    normalised text literally occurs inside the container's. Containers are
    tried longest first, so the greediest true grouping wins, and each region
    is claimed at most once.

    With tolerance > 0 containment is approximate: a part counts as present if
    its infix edit distance is within that fraction of its own length, so one
    misread character no longer hides a group.

    The threshold guards against absorbing a short incidental substring: the
    parts joined together must reconstruct nearly the whole container, so a
    lone "SK-II" inside a long sentence is rejected rather than swallowed.
    """
    groups: list[tuple[int, list[int], float]] = []
    claimed_parts: set[int] = set()
    order = sorted(free_container, key=lambda i: (-len(container_norm[i]), i))

    for ci in order:
        whole = container_norm[ci]
        if not whole:
            continue
        hits: list[tuple[int, int]] = []   # (offset in container, part index)
        for pi in free_part:
            if pi in claimed_parts:
                continue
            text = part_norm[pi]
            if len(text) < min_part or not text:
                continue
            offset = whole.find(text)
            if offset >= 0:
                hits.append((offset, pi))
            elif tolerance > 0:
                distance, end = _infix_distance(text, whole)
                # A dropped footnote marker costs a near-constant 3-4 chars, so
                # a pure ratio would let it through on a long row and block it
                # on a short one. The floor makes short rows behave.
                if distance <= max(floor, int(len(text) * tolerance)):
                    # Sort by window start so the concatenation still reads in
                    # container order.
                    hits.append((max(0, end - len(text)), pi))
        if len(hits) < 2:
            continue
        # Container order, not list order, so the concatenation reads correctly.
        hits.sort()
        indices = [pi for _, pi in hits]
        joined = join.join(part_norm[pi] for pi in indices)
        score = _similarity(whole, joined)
        if score < threshold:
            continue
        claimed_parts.update(indices)
        groups.append((ci, indices, score))
    return groups


def align_regions(
    gt_regions: list[str],
    pred_regions: list[str],
    config: dict[str, Any],
    judge: JudgeClient | None = None,
    *,
    verbose: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Pair ground-truth regions with predicted regions, order-independently.

    Four passes, cheapest first:
      1. exact match on normalised text
      2. structural groups: merges, then splits (substring containment, no API)
      3. similarity >= auto_match
      4. reject_below <= similarity < auto_match, decided by the LLM

    Groups run before the similarity passes on purpose. A one-to-one pass would
    otherwise pair a merged prediction with whichever single row scores highest
    and leave the rest of the group's rows counted as missed, which penalises a
    merge for something it did not get wrong.

    Greedy by descending similarity, so the strongest pair always wins and the
    result does not depend on list order.

    With debug=True the returned dict also carries a "trace": every pairing
    decision with its SequenceMatcher score, including the ones that lost.
    """
    align_config = config["align"]
    auto = float(align_config["auto_match"])
    reject = float(align_config["reject_below"])
    merge_credit = _coerce_bool(align_config.get("merge_credit", True))
    split_credit = _coerce_bool(align_config.get("split_credit", False))
    group_threshold = float(align_config.get("group_similarity", 0.85))
    min_group_part = int(align_config.get("min_group_part", 2))
    group_tolerance = float(align_config.get("group_tolerance", 0.25))
    # Normalised text has no spaces when ignore_whitespace is on, so the parts
    # butt straight together; otherwise a single space stands in for the join.
    join = "" if _coerce_bool(config["normalize"]["ignore_whitespace"]) else " "

    gt_norm = [normalize(r, config) for r in gt_regions]
    pred_norm = [normalize(r, config) for r in pred_regions]

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    # Only populated when debug is on; every rejected or skipped candidate.
    rejected: list[dict[str, Any]] = []

    def take(i: int, j: int, score: float, how: str) -> None:
        used_gt.add(i)
        used_pred.add(j)
        pairs.append(
            {
                "gt_index": i,
                "pred_index": j,
                "gt": gt_regions[i],
                "pred": pred_regions[j],
                "gt_norm": gt_norm[i],
                "pred_norm": pred_norm[j],
                "similarity": round(score, 4),
                "matched_by": how,
            }
        )

    # Pass 1: exact. Left-to-right on both sides keeps it deterministic.
    by_text: dict[str, list[int]] = {}
    for j, text in enumerate(pred_norm):
        by_text.setdefault(text, []).append(j)
    for i, text in enumerate(gt_norm):
        for j in by_text.get(text, []):
            if j not in used_pred:
                take(i, j, 1.0, "exact")
                break

    # Pass 2: structural groups. Merges first: a merge is the benign error, and
    # merge/split are opposite directions so claiming one can never mask the
    # other. Both are pure string containment, so no API call is involved.
    def add_group(kind: str, gt_indices: list[int], pred_indices: list[int],
                  score: float, credited: bool) -> None:
        gt_joined = join.join(gt_norm[i] for i in gt_indices)
        pred_joined = join.join(pred_norm[j] for j in pred_indices)
        used_gt.update(gt_indices)
        used_pred.update(pred_indices)
        groups.append(
            {
                "kind": kind,
                "gt_indices": gt_indices,
                "pred_indices": pred_indices,
                "gt": " | ".join(gt_regions[i] for i in gt_indices),
                "pred": " | ".join(pred_regions[j] for j in pred_indices),
                "gt_norm": gt_joined,
                "pred_norm": pred_joined,
                "similarity": round(score, 4),
                "credited": credited,
                # True when the only difference was where the rows were cut:
                # every character survived, so there is nothing to charge.
                "layout_only": gt_joined == pred_joined,
            }
        )

    for pred_index, gt_indices, score in _find_groups(
        pred_norm, gt_norm, set(range(len(pred_norm))) - used_pred,
        set(range(len(gt_norm))) - used_gt,
        threshold=group_threshold, min_part=min_group_part, join=join,
        tolerance=group_tolerance,
    ):
        add_group("merge", gt_indices, [pred_index], score, merge_credit)

    for gt_index, pred_indices, score in _find_groups(
        gt_norm, pred_norm, set(range(len(gt_norm))) - used_gt,
        set(range(len(pred_norm))) - used_pred,
        threshold=group_threshold, min_part=min_group_part, join=join,
        tolerance=group_tolerance,
    ):
        add_group("split", [gt_index], pred_indices, score, split_credit)

    # Score the remaining cross product once. best_for tracks the top score seen
    # for each side even when it lost, which is the useful number when debugging
    # a region that failed to match.
    candidates: list[tuple[float, int, int]] = []
    best_gt: dict[int, tuple[float, int]] = {}
    best_pred: dict[int, tuple[float, int]] = {}
    for i in range(len(gt_norm)):
        if i in used_gt:
            continue
        for j in range(len(pred_norm)):
            if j in used_pred:
                continue
            score = _similarity(gt_norm[i], pred_norm[j])
            if debug:
                if score > best_gt.get(i, (-1.0, -1))[0]:
                    best_gt[i] = (score, j)
                if score > best_pred.get(j, (-1.0, -1))[0]:
                    best_pred[j] = (score, i)
            if score >= reject:
                candidates.append((score, i, j))
            # Pairs below reject_below are not logged individually: that is the
            # whole N x M cross product. Each region's best score is reported in
            # unmatched_gt / unmatched_pred instead.
    # Sort by score desc, then index asc, so ties resolve the same way always.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    # Pass 3: confident local matches.
    deferred: list[tuple[float, int, int]] = []
    for score, i, j in candidates:
        if i in used_gt or j in used_pred:
            if debug:
                rejected.append(
                    {
                        "gt": gt_regions[i],
                        "pred": pred_regions[j],
                        "similarity": round(score, 4),
                        "decision": "lost_to_better_pair",
                    }
                )
            continue
        if score >= auto:
            take(i, j, score, "similarity")
        else:
            deferred.append((score, i, j))

    # Pass 4: the uncertain band, one independent API call per pair.
    uncertain = 0
    for score, i, j in deferred:
        if i in used_gt or j in used_pred:
            if debug:
                rejected.append(
                    {
                        "gt": gt_regions[i],
                        "pred": pred_regions[j],
                        "similarity": round(score, 4),
                        "decision": "lost_to_better_pair",
                    }
                )
            continue
        uncertain += 1
        if judge is None:
            if debug:
                rejected.append(
                    {
                        "gt": gt_regions[i],
                        "pred": pred_regions[j],
                        "similarity": round(score, 4),
                        "decision": "uncertain_no_judge",
                    }
                )
            continue
        if verbose:
            print(f"  [judge {score:.2f}] {gt_regions[i][:34]} || {pred_regions[j][:34]}", flush=True)
        # Without a verdict, fall back to the local similarity decision.
        same = judge.is_same(gt_regions[i], pred_regions[j], fallback=score >= auto)
        if same:
            take(i, j, score, "llm")
            if debug:
                pairs[-1]["judge_source"] = judge.last_source
        elif debug:
            rejected.append(
                {
                    "gt": gt_regions[i],
                    "pred": pred_regions[j],
                    "similarity": round(score, 4),
                    "decision": "llm_said_different",
                    "judge_source": judge.last_source,
                }
            )

    pairs.sort(key=lambda p: (p["gt_index"], p["pred_index"]))
    groups.sort(key=lambda g: (g["gt_indices"][0], g["pred_indices"][0]))

    # An uncredited group is scored exactly like a region nobody detected: its
    # annotations are missed and its predictions are spurious. It stays in
    # "groups" so the report can still say a split was recognised and rejected.
    penalized_gt = [i for g in groups if not g["credited"] for i in g["gt_indices"]]
    penalized_pred = [j for g in groups if not g["credited"] for j in g["pred_indices"]]
    missing_idx = sorted(
        [i for i in range(len(gt_regions)) if i not in used_gt] + penalized_gt
    )
    extra_idx = sorted(
        [j for j in range(len(pred_regions)) if j not in used_pred] + penalized_pred
    )

    scored_groups = [g for g in groups if g["credited"]]
    result = {
        "pairs": pairs,
        "groups": groups,
        "scored_groups": scored_groups,
        "missing": [gt_regions[i] for i in missing_idx],
        "extra": [pred_regions[j] for j in extra_idx],
        "uncertain_pairs": uncertain,
        # matched counts diverge once groups exist: one merged prediction can
        # satisfy three annotations, so each side needs its own tally.
        "matched_gt": len(pairs) + sum(len(g["gt_indices"]) for g in scored_groups),
        "matched_pred": len(pairs) + sum(len(g["pred_indices"]) for g in scored_groups),
        "gt_total": len(gt_regions),
        "pred_total": len(pred_regions),
    }
    if debug:
        result["trace"] = {
            "thresholds": {
                "auto_match": auto,
                "reject_below": reject,
                "group_similarity": group_threshold,
                "merge_credit": merge_credit,
                "split_credit": split_credit,
            },
            "groups": [
                {
                    "kind": g["kind"],
                    "gt": g["gt"],
                    "pred": g["pred"],
                    "similarity": g["similarity"],
                    "credited": g["credited"],
                    "layout_only": g["layout_only"],
                    "gt_rows": len(g["gt_indices"]),
                    "pred_rows": len(g["pred_indices"]),
                    "char_errors": edit_distance(g["gt_norm"], g["pred_norm"]),
                    "gt_chars": len(g["gt_norm"]),
                }
                for g in groups
            ],
            "matched": [
                {
                    "gt": p["gt"],
                    "pred": p["pred"],
                    "similarity": p["similarity"],
                    "matched_by": p["matched_by"],
                    **({"judge_source": p["judge_source"]} if "judge_source" in p else {}),
                    "char_errors": edit_distance(p["gt_norm"], p["pred_norm"]),
                    "gt_chars": len(p["gt_norm"]),
                    "token_errors": edit_distance(tokenize(p["gt_norm"]), tokenize(p["pred_norm"])),
                    "gt_tokens": len(tokenize(p["gt_norm"])),
                }
                for p in pairs
            ],
            # For each unmatched region: its best score, so you can see how close
            # it came and pick a threshold from real numbers.
            "unmatched_gt": [
                {
                    "gt": gt_regions[i],
                    "best_similarity": round(best_gt[i][0], 4) if i in best_gt else None,
                    "best_pred": pred_regions[best_gt[i][1]] if i in best_gt else None,
                }
                for i in range(len(gt_regions))
                if i not in used_gt
            ],
            "unmatched_pred": [
                {
                    "pred": pred_regions[j],
                    "best_similarity": round(best_pred[j][0], 4) if j in best_pred else None,
                    "best_gt": gt_regions[best_pred[j][1]] if j in best_pred else None,
                }
                for j in range(len(pred_regions))
                if j not in used_pred
            ],
            "rejected": sorted(rejected, key=lambda r: -r["similarity"]),
        }
    return result


def edit_distance(first: list[Any] | str, second: list[Any] | str) -> int:
    """Levenshtein distance over characters or tokens."""
    left, right = list(first), list(second)
    if not left:
        return len(right)
    previous = list(range(len(right) + 1))
    for i, lval in enumerate(left, start=1):
        current = [i]
        for j, rval in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,              # deletion
                    current[j - 1] + 1,           # insertion
                    previous[j - 1] + (lval != rval),  # substitution
                )
            )
        previous = current
    return previous[-1]


def error_counts(alignment: dict[str, Any], config: dict[str, Any]) -> dict[str, int]:
    """Edit distance and reference length, summed over the alignment.

    Matched pairs contribute their own edit distance. An unmatched ground-truth
    region is all deletions, an unmatched prediction is all insertions. This is
    what makes list order irrelevant: distance is never computed across the
    whole concatenated record, only within aligned pairs.
    """
    char_errors = char_length = 0
    token_errors = token_length = 0

    # Credited groups are charged like a pair, on their concatenations. Only
    # credited ones: an uncredited group's regions already sit in missing/extra,
    # so counting it here too would charge it twice.
    for pair in alignment["pairs"] + alignment.get("scored_groups", []):
        gt, pred = pair["gt_norm"], pair["pred_norm"]
        char_errors += edit_distance(gt, pred)
        char_length += len(gt)
        gt_tokens, pred_tokens = tokenize(gt), tokenize(pred)
        token_errors += edit_distance(gt_tokens, pred_tokens)
        token_length += len(gt_tokens)

    for region in alignment["missing"]:
        text = normalize(region, config)
        char_errors += len(text)
        char_length += len(text)
        tokens = tokenize(text)
        token_errors += len(tokens)
        token_length += len(tokens)

    for region in alignment["extra"]:
        text = normalize(region, config)
        char_errors += len(text)          # insertions, not part of reference
        token_errors += len(tokenize(text))

    return {
        "char_errors": char_errors,
        "char_length": char_length,
        "token_errors": token_errors,
        "token_length": token_length,
    }


def chunking_counts(alignment: dict[str, Any]) -> dict[str, int]:
    """How many located annotations sit at the right granularity.

    A separate question from CER. CER asks whether the characters are right;
    this asks whether the rows were cut where the business rules say to cut
    them.

    Counted on the prediction side: of the chunks the model emitted, how many
    are correctly cut chunks. That is what makes the penalty asymmetric without
    needing a weight parameter, and it matches how merges and splits are scored
    in CER:

      whole   paired one-to-one                 1 good chunk
      merged  several annotations in one row     1 bad chunk, however many
                                                 rows were fused - the wording
                                                 is all still there, only the
                                                 boundary is missing
      split   one annotation across N rows       N bad chunks - every fragment
                                                 is a half claim, unusable
                                                 downstream

    So fusing 10 rows costs 1 and breaking 1 row into 4 costs 4. Counting the
    annotation side instead would invert that and charge the merge 10.

    Annotations nobody found are excluded. A row that was never read is an OCR
    failure, and charging it here would make the two numbers move together and
    stop telling you which one to fix. Raw group counts are reported alongside,
    so the annotation-side convention stays computable.
    """
    merged = sum(len(g["pred_indices"]) for g in alignment["groups"] if g["kind"] == "merge")
    split = sum(len(g["pred_indices"]) for g in alignment["groups"] if g["kind"] == "split")
    whole = len(alignment["pairs"])
    return {
        "chunk_whole": whole,
        "chunk_merged": merged,
        "chunk_split": split,
        "chunk_located": whole + merged + split,
    }


def _rate(errors: int, length: int) -> float:
    """errors / length, with the empty-reference case pinned to 0 or 1."""
    if length == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / length


def _prf(matched_gt: int, matched_pred: int, gt_total: int, pred_total: int) -> dict[str, float]:
    """Each side gets its own numerator so a merge cannot inflate the other.

    A credited merge of three annotations into one prediction is 3 matched
    annotations and 1 matched prediction. Sharing one numerator would either
    overstate precision or understate recall.
    """
    precision = 1.0 if pred_total == 0 else matched_pred / pred_total
    recall = 1.0 if gt_total == 0 else matched_gt / gt_total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def record_metrics(alignment: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    counts = error_counts(alignment, config)
    groups = alignment.get("groups", [])
    matched_gt = alignment["matched_gt"]
    matched_pred = alignment["matched_pred"]
    gt_total = alignment["gt_total"]
    pred_total = alignment["pred_total"]
    merges = [g for g in groups if g["kind"] == "merge"]
    splits = [g for g in groups if g["kind"] == "split"]
    chunks = chunking_counts(alignment)
    metrics = {
        "cer": round(_rate(counts["char_errors"], counts["char_length"]), 4),
        "wer": round(_rate(counts["token_errors"], counts["token_length"]), 4),
        **_prf(matched_gt, matched_pred, gt_total, pred_total),
        "chunking_accuracy": round(
            chunks["chunk_whole"] / chunks["chunk_located"] if chunks["chunk_located"] else 0.0, 4
        ),
        "merges": len(merges),
        "splits": len(splits),
    }
    metrics["_counts"] = {
        **counts,
        **chunks,
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "gt": gt_total,
        "pred": pred_total,
        "merge_count": len(merges),
        "split_count": len(splits),
        "merge_layout_only": sum(1 for g in merges if g["layout_only"]),
        "credited_groups": len(alignment.get("scored_groups", [])),
        "penalized_groups": sum(1 for g in groups if not g["credited"]),
    }
    return metrics


def evaluate(
    ground_truth_file: str,
    prediction_file: str,
    output_file: str,
    *,
    config: dict[str, Any],
    prompt: str = JUDGE_PROMPT,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    judge_enabled = _coerce_bool(config["judge"]["enabled"])
    debug = _coerce_bool(config["output"].get("debug", False))
    # debug implies detail: the trace is only readable next to missing/extra.
    detail = _coerce_bool(config["output"]["detail"]) or debug

    judge: JudgeClient | None = None
    if judge_enabled:
        api_key = api_key or _coalesce_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
        base_url = base_url or _coalesce_env("DEEPSEEK_BASE_URL", "OPENAI_BASE_URL")
        model = model or _coalesce_env("DEEPSEEK_MODEL", "EVAL_MODEL")
        if not api_key or not base_url or not model:
            raise ValueError(
                "judge is enabled but credentials are missing. Set "
                "DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL, "
                "or pass --no-llm-judge."
            )
        judge = JudgeClient(
            config,
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            verbose=verbose,
        )

    ground_truths = load_ocr_json(ground_truth_file, config)
    predictions = load_ocr_json(prediction_file, config)

    # Score only the files the prediction file actually covers. A ground-truth
    # image with no prediction row is a coverage gap, not a model error: folding
    # it in as an all-missing record would drag recall and CER down for a reason
    # the metrics cannot distinguish from bad OCR.
    scored_files = sorted(set(ground_truths) & set(predictions))
    skipped_files = sorted(set(ground_truths) - set(predictions))
    if skipped_files:
        print(
            f"[coverage] {len(skipped_files)} ground-truth file(s) absent from "
            f"predictions, not scored: "
            f"{', '.join(skipped_files[:5])}{' ...' if len(skipped_files) > 5 else ''}",
            flush=True,
        )
    if not scored_files:
        raise ValueError(
            f"no file names shared between {ground_truth_file} and {prediction_file}"
        )

    items: list[dict[str, Any]] = []
    totals = {"char_errors": 0, "char_length": 0, "token_errors": 0, "token_length": 0,
              "matched_gt": 0, "matched_pred": 0, "gt": 0, "pred": 0,
              "merge_count": 0, "split_count": 0, "merge_layout_only": 0,
              "credited_groups": 0, "penalized_groups": 0,
              "chunk_whole": 0, "chunk_merged": 0, "chunk_split": 0, "chunk_located": 0}
    uncertain_total = 0

    for file_name in scored_files:
        alignment = align_regions(
            ground_truths[file_name],
            predictions[file_name],
            config,
            judge,
            verbose=verbose,
            debug=debug,
        )
        metrics = record_metrics(alignment, config)
        counts = metrics.pop("_counts")
        for key in totals:
            totals[key] += counts[key]
        uncertain_total += alignment["uncertain_pairs"]

        item = {"source_file": file_name, **metrics}
        if detail:
            item["missing"] = alignment["missing"]
            item["extra"] = alignment["extra"]
        if debug:
            item["debug"] = alignment["trace"]
        items.append(item)
        granularity = "".join(
            part
            for part in (
                f" merge={metrics['merges']}" if metrics["merges"] else "",
                f" split={metrics['splits']}" if metrics["splits"] else "",
            )
            if part
        )
        print(
            f"{file_name}: cer={metrics['cer']} wer={metrics['wer']} "
            f"p={metrics['precision']} r={metrics['recall']} "
            f"chunk={metrics['chunking_accuracy']}{granularity}",
            flush=True,
        )

    # Corpus-level rates are computed from summed counts, not by averaging the
    # per-image rates, so long images are not under-weighted.
    summary = {
        "count": len(items),
        "cer": round(_rate(totals["char_errors"], totals["char_length"]), 4),
        "wer": round(_rate(totals["token_errors"], totals["token_length"]), 4),
        **_prf(totals["matched_gt"], totals["matched_pred"], totals["gt"], totals["pred"]),
        # Chunking is the second target dimension, independent of CER: of the
        # annotations that were found, how many were cut where the rules say.
        "chunking_accuracy": round(
            totals["chunk_whole"] / totals["chunk_located"], 4
        ) if totals["chunk_located"] else 0.0,
        "chunks_located": totals["chunk_located"],
        "chunks_whole": totals["chunk_whole"],
        "chunks_merged": totals["chunk_merged"],
        "chunks_split": totals["chunk_split"],
        "gt_regions": totals["gt"],
        "pred_regions": totals["pred"],
        "matched_gt_regions": totals["matched_gt"],
        "matched_pred_regions": totals["matched_pred"],
        # Granularity errors, split by direction because they are scored apart.
        "merges": totals["merge_count"],
        "merges_layout_only": totals["merge_layout_only"],
        "splits": totals["split_count"],
        "credited_groups": totals["credited_groups"],
        "penalized_groups": totals["penalized_groups"],
        "merge_credit": _coerce_bool(config["align"].get("merge_credit", True)),
        "split_credit": _coerce_bool(config["align"].get("split_credit", False)),
        # Coverage is reported separately from quality: these counts say how much
        # of the ground truth the prediction file spoke to at all.
        "gt_files": len(ground_truths),
        "scored_files": len(scored_files),
        "unpredicted_files": len(skipped_files),
    }
    if skipped_files:
        summary["unpredicted_file_names"] = skipped_files
    if judge is not None:
        summary["judge_calls"] = judge.calls
        summary["judge_cache_hits"] = judge.cache_hits
        summary["judge_failures"] = judge.failures
    else:
        summary["uncertain_pairs_skipped"] = uncertain_total

    output = {"summary": summary, "items": items}
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if judge is not None:
        judge.save_cache()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR predictions: CER, WER, precision, recall, F1."
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", default="outputs/ocr_eval.json")
    parser.add_argument("--config", default="benchmark/config.toml")
    parser.add_argument("--prompt-file", help="Override the judge prompt")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--detail", action="store_true", help="Add missing/extra lists per record")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add the full pairing trace: every pair, its SequenceMatcher score, "
        "and why unmatched regions failed",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore and skip the judge cache")
    parser.add_argument("--verbose", action="store_true", help="Show each judged pair")
    parser.add_argument(
        "--no-merge-credit",
        action="store_true",
        help="Make CER charge merges a second time, as missing plus extra "
        "(default: charge only the characters actually wrong)",
    )
    parser.add_argument(
        "--no-split-credit",
        action="store_true",
        help="Make CER charge splits a second time, as missing plus extra "
        "(default: charge only the characters actually wrong)",
    )
    parser.add_argument(
        "--group-similarity",
        type=float,
        help="How completely the parts must reconstruct the whole to be a group (default 0.85)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--llm-judge", action="store_true", help="Force the judge on")
    group.add_argument("--no-llm-judge", action="store_true", help="Local metrics only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config if Path(args.config).is_file() else None)
    if args.llm_judge:
        config["judge"]["enabled"] = True
    if args.no_llm_judge:
        config["judge"]["enabled"] = False
    if args.detail:
        config["output"]["detail"] = True
    if args.debug:
        config["output"]["debug"] = True
    if args.no_cache:
        config["judge"]["cache_file"] = ""
    if args.no_merge_credit:
        config["align"]["merge_credit"] = False
    if args.no_split_credit:
        config["align"]["split_credit"] = False
    if args.group_similarity is not None:
        config["align"]["group_similarity"] = args.group_similarity

    prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8")
        if args.prompt_file
        else JUDGE_PROMPT
    )
    output = evaluate(
        args.ground_truth,
        args.prediction,
        args.output,
        config=config,
        prompt=prompt,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        verbose=args.verbose,
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
