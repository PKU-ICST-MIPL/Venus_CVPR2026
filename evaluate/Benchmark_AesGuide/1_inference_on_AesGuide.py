import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)


model_path = "/path/to/your/model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()


input_json_path = "Venus_CVPR2026/data/Benchmark_AesGuide/json/Benchmark_AesGuide.json"
img_folder = "Venus_CVPR2026/data/Benchmark_AesGuide/images"
output_json_path = "Venus_CVPR2026/evaluate/Benchmark_AesGuide/1_inference_on_AesGuide.json"


with open(input_json_path, 'r') as f:
    data = json.load(f)
    

output_data = {}
for image_name in tqdm(data.keys(), desc="Processing Images", unit="image"):
    image_path = os.path.join(img_folder, image_name)
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': 'Please critically analyze this image from an aesthetic perspective.'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    output_data[image_name] = response
    
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
print(f"Number of processed images: {len(output_data)}")