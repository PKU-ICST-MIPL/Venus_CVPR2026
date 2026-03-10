import json
import os
from openai import OpenAI
import requests
import time
from tqdm import tqdm

API_SECRET_KEY = "your/API_SECRET_KEY";
BASE_URL = "https://api.zhizengzeng.com/v1/"


def chat(query):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    return resp.choices[0].message.content 
    
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
     
def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


MLLM_DESC_path = "Venus_CVPR2026/evaluate/Benchmark_AesGuide/1_inference_on_AesGuide.json"
GOLDEN_DESC_path = "Venus_CVPR2026/data/Benchmark_AesGuide/json/Benchmark_AesGuide.json"
output_file = "Venus_CVPR2026/evaluate/Benchmark_AesGuide/GPT_rate/Preciseness.json"

MLLM_DESC = load_json(MLLM_DESC_path)
GOLDEN_DESC = load_json(GOLDEN_DESC_path)

Score = {}
total_score = 0
for image_name in tqdm(MLLM_DESC.keys(), desc="Processing Images", unit="image"):
    MLLM_DESC_content = MLLM_DESC[image_name]
    GOLDEN_DESC_content = GOLDEN_DESC[image_name]
    prompt = f"""
Evaluate whether the description [MLLM_DESC] precisely reflects the reference description [GOLDEN_DESC], especially in terms of criticism of the image's shortcomings or suggestions for improving the image.
[MLLM_DESC]: {MLLM_DESC_content}
[GOLDEN_DESC]: {GOLDEN_DESC_content}
If [MLLM_DESC] neither criticizes the image's shortcomings nor provides suggestions for improvement, rate a score of 0.
If [MLLM_DESC] has partial differences or controversies compared to [GOLDEN_DESC], rate a score of 1.
If [MLLM_DESC] closely matches [GOLDEN_DESC] or has a similar description, with no major differences or controversies, rate a score of 2.
Please provide the result only in the following format: Score:
    """
    GPT_score = chat(prompt)
    Score[image_name] = GPT_score
    save_json(output_file, Score)
    
