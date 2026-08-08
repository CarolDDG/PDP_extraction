import os
import json
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

class OCRAgent:
    """
    Agent 1: 扫描图片 -> 并发调用 Qwen3.5-OCR -> 输出结构化数据 (使用 OpenAI SDK)
    """
    DEFAULT_PROMPT = """Perform OCR on the provided image, scanning and extracting ALL visible text strictly from top to bottom, left to right. 
Include main headings, sub-headings, body text, annotations, footnotes, labels, and fine print. 

CRITICAL FORMAT REQUIREMENTS:
1. Return ONLY a valid JSON object without markdown formatting.
2. `lines` MUST be a simple flat array of extracted text strings. Do NOT output objects, bounding boxes, or extra keys.
3. DO NOT output any empty strings or blank lines.

JSON Schema Example:
{
  "file_name": "filename",
  "lines": [
    "Extracted text content line 1",
    "Extracted text content line 2",
    "Extracted text content line 3"
  ]
}"""

    def __init__(
        self, 
        api_key: str = None, 
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen-vl-max",
        max_workers: int = 5,
        prompt: str = None,
        timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.timeout = timeout
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf')
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _collect_image_files(self, image_inputs) -> list:
        """1. 智能解析输入：支持文件夹、单文件、或 JSON 数组字符串"""
        if isinstance(image_inputs, str):
            if image_inputs.startswith('[') and image_inputs.endswith(']'):
                try:
                    image_inputs = json.loads(image_inputs)
                except Exception:
                    pass

        image_files = []
        if isinstance(image_inputs, str):
            if os.path.isdir(image_inputs):
                for root, _, files in os.walk(image_inputs):
                    for file in files:
                        if file.lower().endswith(self.image_extensions):
                            image_files.append(os.path.join(root, file))
            elif os.path.isfile(image_inputs) or image_inputs.startswith("http"):
                image_files.append(image_inputs)
        elif isinstance(image_inputs, list):
            for item in image_inputs:
                if os.path.isdir(item):
                    for root, _, files in os.walk(item):
                        for file in files:
                            if file.lower().endswith(self.image_extensions):
                                image_files.append(os.path.join(root, file))
                else:
                    image_files.append(item)

        return image_files

    def _normalize_parsed_result(self, raw_parsed: dict, img_path: str, filter_empty: bool = True) -> dict:
        """
        将 LLM 返回的字符串数组 lines: ["line1", "line2"]
        在 Python 端归一化为标准的:
        {
           "file_name": str,
           "total_lines": int,
           "lines": [ {"line_id": 1, "text": "line1"}, ... ]
        }
        """
        lines = []
        if isinstance(raw_parsed, dict):
            raw_lines = raw_parsed.get("lines", [])
        elif isinstance(raw_parsed, list):
            raw_lines = raw_parsed
        else:
            raw_lines = []

        empty_count = 0
        valid_line_idx = 1

        for item in raw_lines:
            # 兼容纯字符串或对象格式
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
            else:
                text = str(item).strip()

            if not text:
                empty_count += 1
                if filter_empty:
                    continue  # 过滤空白行
                
            lines.append({
                "line_id": valid_line_idx,
                "text": text
            })
            valid_line_idx += 1
            
        return {
            "file_name": os.path.basename(img_path),
            # 保留实际路径：_collect_image_files 会递归子目录，
            # 只靠 file_name 无法还原嵌套路径
            "img_path": img_path,
            "total_lines": len(lines),
            "empty_count": empty_count,
            "lines": lines
        }

    def _call_ocr_api(self, image_content: dict, prompt_text: str) -> dict:
        """底层 API 调用辅助函数"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
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
            try:
                from json_repair import repair_json
                return repair_json(content_clean, return_objects=True)
            except Exception:
                cleaned_str = re.sub(r'[\x00-\x1F\x7F]', ' ', content_clean)
                return json.loads(cleaned_str, strict=False)

    def _process_single_image(self, img_path: str) -> dict:
        """2. 使用 OpenAI SDK 调用 Qwen3.5-OCR，支持自动检测空白行重试"""
        if img_path.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": img_path}}
        else:
            with open(img_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode('utf-8')
            image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
        
        try:
            # 第一次使用默认 Prompt 调用
            parsed_json = self._call_ocr_api(image_content, self.prompt)
            # 未过滤清洗前检查空白行数量
            unfiltered_res = self._normalize_parsed_result(parsed_json, img_path, filter_empty=False)
            
            # 如果空白行多于 3 个（> 3），触发重试提示词
            if unfiltered_res.get("empty_count", 0) > 3:
                retry_prompt = (
                    "RE-SCAN WARNING: The previous OCR scan produced multiple empty text fields. "
                    "Please thoroughly re-scan the entire image, extract all visible text accurately, "
                    "and DO NOT output any empty text fields or blank lines.\n\n" + self.prompt
                )
                parsed_json = self._call_ocr_api(image_content, retry_prompt)

            # 最终清洗，剔除空白行并重编 line_id
            final_res = self._normalize_parsed_result(parsed_json, img_path, filter_empty=True)
            final_res.pop("empty_count", None)
            return final_res
        except Exception as e:
            return {
                "file_name": os.path.basename(img_path),
                "img_path": img_path,
                "error": f"Failed to process: {str(e)}",
                "total_lines": 0,
                "lines": []
            }

    def run(self, image_inputs, return_json_str: bool = True):
        """
        执行 Agent 识别流程
        :param image_inputs: 文件夹路径、单文件路径、URL 或路径列表
        :param return_json_str: 为 True 时返回 JSON 字符串，False 时返回 Python list/dict 结果
        """
        image_files = self._collect_image_files(image_inputs)

        if not image_files:
            error_msg = {"error": "No valid images found in the provided path."}
            return json.dumps(error_msg) if return_json_str else error_msg

        # 多线程并发执行。按输入顺序回填结果：as_completed 的完成顺序
        # 每次都不一样，会让同样输入产生的 JSON 行序随机抖动。
        results_by_index: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_single_image, img): idx
                for idx, img in enumerate(image_files)
            }
            for future in as_completed(future_to_index):
                results_by_index[future_to_index[future]] = future.result()

        raw_results = [results_by_index[i] for i in range(len(image_files)) if i in results_by_index]

        return json.dumps(raw_results, ensure_ascii=False) if return_json_str else raw_results


def agent1_ocr_pipeline(image_inputs, max_workers=5):
    """兼容旧调用的函数入口"""
    agent = OCRAgent(max_workers=max_workers)
    return agent.run(image_inputs, return_json_str=True)


# --- Flowise 节点执行入口（在 Flowise Custom Tool 节点中使用，在纯 Python 环境中需注释） ---
# input_data = $imagePaths 
# agent = OCRAgent(max_workers=5)
# final_output = agent.run(input_data)
# return final_output