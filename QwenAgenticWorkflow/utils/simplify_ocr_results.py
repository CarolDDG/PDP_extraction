import argparse
import json
import os
import re

# 仓库内默认路径：utils/ 的上一级就是 QwenAgenticWorkflow/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(PROJECT_ROOT, "output", "consolidated_ocr_results.json")
DEFAULT_DST = os.path.join(PROJECT_ROOT, "output", "consolidated_ocr_results_simplified.json")
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "run.log")


def simplify_and_sort_ocr_results(
    src_json_path: str = DEFAULT_SRC,
    dst_json_path: str = DEFAULT_DST,
    log_path: str = DEFAULT_LOG
):
    """
    1. 解析 run.log，识别触发 [保底策略生效] 的图片文件名集合
    2. 对原始 json 进行按 file_name 字母升序排序，并标注 "(Manual Revision Needed)" 和 "status"
    3. 提取极简字符串数组 `lines: ["text1", "text2", ...]`
    4. 输出保存至 dst_json_path
    """
    # 1. 提取保底策略生效的文件名集合
    fallback_files = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '保底策略生效' in line:
                    m = re.search(r'\[(.*?)\]', line)
                    if m:
                        fallback_files.add(m.group(1))

    print(f"🔍 解析 run.log 发现 {len(fallback_files)} 个触发保底策略的文件")

    # 2. 读取原始 JSON 数据
    if not os.path.exists(src_json_path):
        print(f"❌ 未找到源 JSON 文件: {src_json_path}")
        return

    with open(src_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 3. 构造极简数据并标记 status
    simplified_data = []
    for item in raw_data:
        orig_name = item.get('file_name', '')
        clean_name = orig_name.replace(' (Manual Revision Needed)', '').strip()

        is_fallback = (clean_name in fallback_files) or item.get('status') == 'Manual Revision Needed'

        final_file_name = f"{clean_name} (Manual Revision Needed)" if is_fallback else clean_name

        lines = [
            mline.get('merged_text', '').strip()
            for mline in item.get('merged_lines', [])
            if mline.get('merged_text') and mline.get('merged_text').strip()
        ]

        entry = {
            "file_name": final_file_name,
            "lines": lines
        }

        if is_fallback:
            entry["status"] = "Manual Revision Needed"

        simplified_data.append(entry)

    # 4. 按 file_name 排序
    simplified_data.sort(key=lambda x: x.get('file_name', ''))

    # 5. 保存到目标输出路径
    output_dir = os.path.dirname(dst_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(dst_json_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"✅ 转换与极简提取成功！保存至: {dst_json_path}")
    print(f"📊 总条目数: {len(simplified_data)}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="将 consolidated_ocr_results.json 转为极简 lines 数组并标注需人工复核的文件"
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"源 JSON (默认: {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST, help=f"输出 JSON (默认: {DEFAULT_DST})")
    parser.add_argument("--log", default=DEFAULT_LOG, help=f"run.log 路径 (默认: {DEFAULT_LOG})")
    args = parser.parse_args()

    simplify_and_sort_ocr_results(
        src_json_path=args.src,
        dst_json_path=args.dst,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
