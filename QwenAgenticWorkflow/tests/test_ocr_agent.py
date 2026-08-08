import pytest
import os
from agent.OCR import OCRAgent

def get_sample_img(name="Cetaphil_image14.png"):
    path1 = os.path.join("./images", name)
    if os.path.exists(path1):
        return path1
    return os.path.join("./samples", name)

def test_ocr_single_image():
    sample_img = get_sample_img("Cetaphil_image14.png")
    assert os.path.exists(sample_img), f"测试图片不存在: {sample_img}"

    agent = OCRAgent(max_workers=1)
    result = agent._process_single_image(sample_img)

    # 1. 结构完整性校验
    assert "file_name" in result
    assert "total_lines" in result
    assert "lines" in result

    # 2. 纯空白行过滤校验 (不能包含空字符串)
    for item in result["lines"]:
        assert isinstance(item, dict), f"Line item 必须为 dict: {item}"
        assert item["text"].strip() != "", f"检测到未过滤的空白行: {item}"

    # 3. 连续编号校验
    line_ids = [item["line_id"] for item in result["lines"]]
    assert line_ids == list(range(1, len(line_ids) + 1)), "line_id 编号不连续"
    print(f"\n✅ Agent 1 测试通过！共扫描到 {result['total_lines']} 行有效文本。")

if __name__ == "__main__":
    test_ocr_single_image()
