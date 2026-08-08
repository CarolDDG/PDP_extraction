import os
from typing import List, Dict, Any

def prepare_consolidation_inputs(ocr_results: Any, image_inputs: str) -> List[Dict[str, Any]]:
    """
    中间预处理工具函数：
    解析第一个 Agent 输出的 JSON 数据结构，比对并提取每张图片的物理路径 dir 以及对应的 lines 数据。
    """
    payloads = []
    
    if isinstance(ocr_results, dict):
        ocr_results = [ocr_results]
    elif not isinstance(ocr_results, list):
        return payloads
    
    for item in ocr_results:
        if not isinstance(item, dict):
            continue
        file_name = item.get("file_name")
        if not file_name:
            continue
        raw_lines = item.get("lines", [])
        clean_lines = []
        for line in raw_lines:
            if isinstance(line, dict):
                clean_lines.append({
                    "line_id": line.get("line_id"),
                    "text": line.get("text", "")
                })

        # 优先用 Agent 1 带回的真实路径。
        # OCRAgent._collect_image_files 会递归子目录，用 file_name 重新拼
        # image_inputs/<basename> 在嵌套目录下会指向不存在的文件。
        img_path = item.get("img_path")
        if not img_path or not os.path.exists(img_path):
            if os.path.isdir(image_inputs):
                img_path = os.path.join(image_inputs, file_name)
            else:
                img_path = image_inputs

        payloads.append({
            "file_name": file_name,
            "img_path": img_path,
            "lines": clean_lines
        })

    return payloads
