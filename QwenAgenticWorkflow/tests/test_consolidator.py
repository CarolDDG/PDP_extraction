import os
import pytest
from agent.Consolidator import LayoutConsolidatorAgent

def get_sample_img(name="Cetaphil_image18.png"):
    path1 = os.path.join("./images", name)
    if os.path.exists(path1):
        return path1
    return os.path.join("./samples", name)

def test_consolidator_zero_omission():
    sample_img = get_sample_img("Cetaphil_image18.png")
    assert os.path.exists(sample_img), f"测试图片不存在: {sample_img}"

    # 模拟输入 5 行文本数据
    mock_lines = [
        {"line_id": 1, "text": "认准中国官方授权版本"},
        {"line_id": 2, "text": "丝塔芙舒润保湿霜（大白罐）不同版本的包装、成分、"},
        {"line_id": 3, "text": "适用人群不同，请认准后购买。"},
        {"line_id": 4, "text": "中国官方授权版本"},
        {"line_id": 5, "text": "550g"}
    ]

    agent = LayoutConsolidatorAgent()
    result = agent.process_single_image(sample_img, mock_lines)

    # 收集所有的 original_line_ids
    covered_ids = set()
    for mline in result.get("merged_lines", []):
        covered_ids.update(mline.get("original_line_ids", []))

    input_ids = {item["line_id"] for item in mock_lines}
    
    # 断言：行覆盖率必须达到 100% (零遗漏)
    assert input_ids.issubset(covered_ids), f"检测到丢行！缺失的 IDs: {input_ids - covered_ids}"
    print(f"\n✅ Agent 2 零遗漏测试通过！所有 1..{len(input_ids)} 个 Line ID 均被完整覆盖。")

if __name__ == "__main__":
    test_consolidator_zero_omission()
