import os
import json
import time
from typing import List, Dict, Any
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start

from agent.OCR import OCRAgent
from agent.Consolidator import LayoutConsolidatorAgent
from utils.preprocessor import prepare_consolidation_inputs

class OCRFlowState(BaseModel):
    image_inputs: str = ""
    output_dir: str = "./output"
    raw_ocr_results: List[Dict[str, Any]] = []
    consolidated_results: List[Dict[str, Any]] = []

class OCRAndConsolidationFlow(Flow[OCRFlowState]):
    """
    CrewAI Flow Runtime:
    串联 1. OCRAgent -> 2. preprocessor -> 3. LayoutConsolidatorAgent
    """

    @start()
    def step1_ocr_scan(self):
        """步骤 1：调用现有的 OCRAgent 进行原始行级识别"""
        print("\n🚀 [Step 1] 启动 第一轮 OCRAgent 扫描提取所有 lines...")
        ocr_agent = OCRAgent(max_workers=5)
        
        # 运行识别 (显式指定 return_json_str=False 返回 Python list/dict)
        raw_results = ocr_agent.run(self.state.image_inputs, return_json_str=False)
        self.state.raw_ocr_results = raw_results

        # 保存中间结果到 ocr_results.json
        os.makedirs(self.state.output_dir, exist_ok=True)
        ocr_json_path = os.path.join(self.state.output_dir, "ocr_results.json")
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)
        print(f"✅ 第一轮识别完成，结构化数据保存至: {ocr_json_path}")

        return raw_results

    @listen(step1_ocr_scan)
    def step2_preprocess_and_consolidate(self, ocr_results):
        """步骤 2 & 3：使用 preprocessor 解耦数据，并串联 LayoutConsolidatorAgent 执行排版及语义合并"""
        if isinstance(ocr_results, str):
            try:
                ocr_results = json.loads(ocr_results)
            except Exception:
                ocr_results = []
                
        print("\n⚙️ [Step 2] 执行中间预处理，提炼单图对应的 dir 与 lines...")
        payloads = prepare_consolidation_inputs(ocr_results, self.state.image_inputs)

        print("\n🧠 [Step 3] 启动 第二轮 LayoutConsolidatorAgent 执行位置与语义合并...")
        consolidator_agent = LayoutConsolidatorAgent()
        final_results = []

        for payload in payloads:
            file_name = payload["file_name"]
            img_path = payload["img_path"]
            lines = payload["lines"]

            print(f"👉 正在对 [{file_name}] 的 {len(lines)} 行原始数据做排版与合并校正...")
            
            # Agent 2 逐图处理
            merged_res = consolidator_agent.process_single_image(img_path, lines)
            final_results.append(merged_res)

        self.state.consolidated_results = final_results

        # 保存最终排版校正后的 JSON 结果
        final_json_path = os.path.join(self.state.output_dir, "consolidated_ocr_results.json")
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"✅ 排版合并完成！最终结果保存至: {final_json_path}")

        return final_results

def main():
    image_path = os.path.abspath("./images22")
    output_dir = os.path.abspath("./output_images22")

    start_time = time.time()
    flow = OCRAndConsolidationFlow()
    flow.kickoff(inputs={
        "image_inputs": image_path,
        "output_dir": output_dir
    })
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ [Total Execution Time] 流程总体运行耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")

if __name__ == "__main__":
    main()
