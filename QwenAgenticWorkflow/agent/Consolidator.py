import os
import json
import base64
import re
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel

class MergedLine(BaseModel):
    merged_id: int
    original_line_ids: List[int]
    merged_text: str

class ConsolidationResult(BaseModel):
    file_name: str
    total_merged_lines: int
    merged_lines: List[MergedLine]

class LayoutConsolidatorAgent:
    """
    Agent 2: 结合原图与第一轮的 OCR lines 列表，合并位置相近且语义关联的文本
    """
    DEFAULT_PROMPT = """You are an expert in document layout analysis and OCR text consolidation.

You are provided with:
1. An image.
2. A list of OCR extracted text lines (`lines`) with their `line_id` and `text`.

YOUR TASK:
Analyze the image layout (horizontal/vertical proximity, font size, paragraph structure) along with semantic continuity, and consolidate broken text lines into coherent sentences or logical text blocks.

CRITICAL RULES:
1. NO OMISSION ALLOWED (ZERO MISSING LINES): You MUST process and account for EVERY SINGLE `line_id` present in the input `lines` array. No `line_id` may be omitted or dropped! Every original line from 1 to N MUST appear in `original_line_ids` of the output (either merged with adjacent lines or kept as a standalone line).
2. SELECTIVE MERGING ONLY: Do NOT merge all lines blindly or infinitely. ONLY merge lines/blocks that are physically close in proximity (horizontally or vertically) AND semantically continuous within the same paragraph or text block.
3. UNMERGED LINES: Lines that do NOT need merging (such as standalone headings, isolated labels, table cells, or separate list items) MUST be preserved in the output as independent elements. For unmerged lines, `original_line_ids` will contain exactly one ID (e.g., `[3]`).
4. NATURAL TEXT CONCATENATION: When merging texts into `merged_text`, join them cleanly and naturally. Do NOT insert extra unnecessary blank spaces or repetitive padding between merged words/phrases unless punctuation requires it.
5. RECORD ORIGIN: Record all combined original `line_id`s in `original_line_ids` for each output line.

CRITICAL OUTPUT FORMAT:
Return ONLY a valid JSON object strictly matching this schema:
{
  "file_name": "filename.jpg",
  "total_merged_lines": 2,
  "merged_lines": [
    {
      "merged_id": 1,
      "original_line_ids": [1, 2],
      "merged_text": "Complete combined text of line 1 and 2."
    },
    {
      "merged_id": 2,
      "original_line_ids": [3],
      "merged_text": "Standalone unmerged line text."
    }
  ]
}"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen3.5-omni-flash",
        prompt: str = None,
        timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model_name = model_name
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.timeout = timeout

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _call_llm_api(self, image_content: dict, prompt_text: str, lines: List[Dict[str, Any]]) -> dict:
        """调用多模态大模型 API 并返回解析后的 JSON 对象"""
        lines_input_str = json.dumps(lines, ensure_ascii=False, indent=2)
        user_message_text = f"{prompt_text}\n\n[Input Lines Data]:\n{lines_input_str}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message_text},
                        image_content
                    ]
                }
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content or ""
        content_clean = content.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content_clean)
        if json_match:
            content_clean = json_match.group(0)

        try:
            return json.loads(content_clean, strict=False)
        except json.JSONDecodeError:
            from json_repair import repair_json
            return repair_json(content_clean, return_objects=True)

    def _check_missing_line_ids(self, parsed_json: dict, all_line_ids: set) -> set:
        """校验返回结果中的 original_line_ids 集合，找出丢失的 line_id"""
        covered_ids = set()
        merged_lines = parsed_json.get("merged_lines", [])
        if isinstance(merged_lines, list):
            for item in merged_lines:
                if isinstance(item, dict):
                    orig_ids = item.get("original_line_ids", [])
                    if isinstance(orig_ids, list):
                        for lid in orig_ids:
                            try:
                                covered_ids.add(int(lid))
                            except (ValueError, TypeError):
                                pass
        return all_line_ids - covered_ids

    def process_single_image(self, img_path: str, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理单张图片的合并请求，支持丢行自动识别与重试
        """
        if img_path.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": img_path}}
        else:
            with open(img_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode('utf-8')
            image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}

        # 输入所有的 line_id 集合与 lookup dict
        lines_dict = {int(item["line_id"]): item.get("text", "") for item in lines if "line_id" in item}
        all_line_ids = set(lines_dict.keys())

        try:
            # 1. 第一次调用
            parsed_json = self._call_llm_api(image_content, self.prompt, lines)
            # json_repair 可能返回 list，后面的 .get 会炸
            if not isinstance(parsed_json, dict):
                parsed_json = {"merged_lines": parsed_json if isinstance(parsed_json, list) else []}
            missing_ids = self._check_missing_line_ids(parsed_json, all_line_ids)

            # 2. 如果检测到有缺失的 line_id，发起追加重新处理提示的二次重试
            if missing_ids:
                missing_list = sorted(list(missing_ids))
                print(f"⚠️ [{os.path.basename(img_path)}] 首次输出遗漏了 Line IDs: {missing_list}，正在发起补全重试...")
                # 用真实的 id 上界，不假设 1..N 连续
                max_id = max(all_line_ids) if all_line_ids else 0
                retry_prompt = (
                    f"RE-PROCESS WARNING: In your previous response, you dropped/omitted line IDs: {missing_list}. "
                    f"You MUST include EVERY line from line_id 1 to {max_id} without omitting any! "
                    f"Please re-process this image and ensure NO lines are dropped.\n\n" + self.prompt
                )
                parsed_json = self._call_llm_api(image_content, retry_prompt, lines)
                if not isinstance(parsed_json, dict):
                    parsed_json = {"merged_lines": parsed_json if isinstance(parsed_json, list) else []}
                missing_ids = self._check_missing_line_ids(parsed_json, all_line_ids)

            # 3. 兜底保护 (Fallback protection): 若重试后仍有极大特例遗漏的 line_id，在 Python 端补齐
            merged_lines = parsed_json.get("merged_lines", [])
            if not isinstance(merged_lines, list):
                merged_lines = []

            if missing_ids:
                missing_list = sorted(list(missing_ids))
                print(f"ℹ️ [{os.path.basename(img_path)}] 保底策略生效：补齐剩余缺失的 Line IDs: {missing_list}")
                next_merged_id = len(merged_lines) + 1
                for mid in missing_list:
                    merged_lines.append({
                        "merged_id": next_merged_id,
                        "original_line_ids": [mid],
                        "merged_text": lines_dict.get(mid, "")
                    })
                    next_merged_id += 1

            # 重新规范化 merged_id 与 total_merged_lines
            for idx, item in enumerate(merged_lines):
                if isinstance(item, dict):
                    item["merged_id"] = idx + 1

            parsed_json["file_name"] = os.path.basename(img_path)
            parsed_json["total_merged_lines"] = len(merged_lines)
            parsed_json["merged_lines"] = merged_lines
            return parsed_json
        except Exception as e:
            return {
                "file_name": os.path.basename(img_path),
                "error": f"Consolidation failed: {str(e)}",
                "total_merged_lines": 0,
                "merged_lines": []
            }
