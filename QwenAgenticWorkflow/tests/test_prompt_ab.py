import os
import json
from dotenv import load_dotenv
load_dotenv()

from agent.Consolidator import LayoutConsolidatorAgent

# 候选 Prompt A (对比参考：简陋 Prompt，缺乏强约束)
PROMPT_A = """Analyze the image and merge broken text lines into coherent sentences."""

# 候选 Prompt B (系统当前配置：带有零遗漏和严格排版约束的优化 Prompt)
PROMPT_B = LayoutConsolidatorAgent.DEFAULT_PROMPT

def run_prompt_ab_test(image_path: str, lines: list):
    print(f"\n🧪 正在对图片 [{os.path.basename(image_path)}] 进行 Prompt A/B 对比测试...")

    agent_a = LayoutConsolidatorAgent(prompt=PROMPT_A)
    agent_b = LayoutConsolidatorAgent(prompt=PROMPT_B)

    res_a = agent_a.process_single_image(image_path, lines)
    res_b = agent_b.process_single_image(image_path, lines)

    # 统计覆盖率与合并组数
    ids_a = {id_ for m in res_a.get("merged_lines", []) for id_ in m.get("original_line_ids", [])}
    ids_b = {id_ for m in res_b.get("merged_lines", []) for id_ in m.get("original_line_ids", [])}
    total_input = len(lines)

    print(f"📊 Prompt A 表现 -> ID 覆盖率: {len(ids_a)}/{total_input} | 合并后项数: {res_a.get('total_merged_lines')}")
    print(f"📊 Prompt B 表现 -> ID 覆盖率: {len(ids_b)}/{total_input} | 合并后项数: {res_b.get('total_merged_lines')}")

if __name__ == "__main__":
    sample_img = os.path.join("./images", "Cetaphil_image18.png")
    if not os.path.exists(sample_img):
        sample_img = os.path.join("./samples", "Cetaphil_image18.png")

    sample_lines = [
        {"line_id": 1, "text": "认准中国官方授权版本"},
        {"line_id": 2, "text": "丝塔芙舒润保湿霜（大白罐）不同版本的包装、成分、"},
        {"line_id": 3, "text": "适用人群不同，请认准后购买。"},
        {"line_id": 4, "text": "中国官方授权版本"},
        {"line_id": 5, "text": "550g"}
    ]
    if os.path.exists(sample_img):
        run_prompt_ab_test(sample_img, sample_lines)
