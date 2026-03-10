import json
import re

input_path = "Venus_CVPR2026/evaluate/Benchmark_FLMS/1_inference_on_FLMS.json"
output_path = "Venus_CVPR2026/evaluate/Benchmark_FLMS/2_inference_on_FLMS_turn.json"

# 读取JSON文件
with open(input_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 提取并转换
output_data = {}
for key, value in data.items():
    # 使用正则提取坐标中的数字
    matches = re.findall(r"\((\d+),(\d+)\)", value)
    if matches:
        # 将数字用空格分隔后组合为字符串
        coordinates = " ".join([f"{x} {y}" for x, y in matches])
        output_data[key] = coordinates

# 写入新JSON文件
with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(output_data, outfile, indent=4, ensure_ascii=False)

print(f"处理完成，结果已保存到 {output_path}")
