import os
import time
from dotenv import load_dotenv
load_dotenv()

from agent.OCR import OCRAgent

def compare_ocr_models(image_path: str):
    print(f"\n🔍 正在对图片 [{os.path.basename(image_path)}] 进行 OCRAgent 模型对比测试...")
    print("------------------------------------------------------------")

    # 1. 初始化 qwen3.5-ocr 模型实例
    agent_ocr = OCRAgent(model_name="qwen3.5-ocr", max_workers=1)
    t0 = time.time()
    res_ocr = agent_ocr._process_single_image(image_path)
    time_ocr = time.time() - t0

    # 2. 初始化 qwen-vl-max 模型实例
    agent_vl = OCRAgent(model_name="qwen-vl-max", max_workers=1)
    t1 = time.time()
    res_vl = agent_vl._process_single_image(image_path)
    time_vl = time.time() - t1

    # 结果对比输出
    lines_ocr = res_ocr.get("lines", [])
    lines_vl = res_vl.get("lines", [])

    print(f"\n🤖 [模型 1: qwen3.5-ocr]")
    print(f"   ⏱️ 耗时: {time_ocr:.2f}s | 📄 提取有效行数: {len(lines_ocr)}")
    print("   📝 前 3 行提取示例:")
    for l in lines_ocr[:3]:
        print(f"      - Line {l['line_id']}: {l['text']}")

    print(f"\n🤖 [模型 2: qwen-vl-max]")
    print(f"   ⏱️ 耗时: {time_vl:.2f}s | 📄 提取有效行数: {len(lines_vl)}")
    print("   📝 前 3 行提取示例:")
    for l in lines_vl[:3]:
        print(f"      - Line {l['line_id']}: {l['text']}")

    print("\n------------------------------------------------------------")
    print(f"💡 提取行数对比 -> qwen3.5-ocr: {len(lines_ocr)} 行  vs  qwen-vl-max: {len(lines_vl)} 行")

if __name__ == "__main__":
    test_images = [
        os.path.abspath("./images/Clarins_image06.png"),
        os.path.abspath("./images/Cetaphil_image18.png"),
        os.path.abspath("./images/EsteeLauder_Cream_image13.png")
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            compare_ocr_models(img_path)
        else:
            print(f"未找到测试图片: {img_path}")
