import numpy as np
import torch, json
import os.path as osp
import os
from tqdm import tqdm

from decord import VideoReader
from PIL import Image
from typing import List
import io
import base64
import re

import argparse
import math
import random
import time
import importlib

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.eval_utils import *

def load_data(path):
    datas = {}
    for demo in ['hrd', 'file_sys', 'cup', 'grid', 'card', 'chip']:
        curr_datas = load_json(osp.join(path, f'{demo}.json'))
        for data in curr_datas.values():
            data['demo'] = demo
        curr_datas = {f'{demo}_{k}': v for k, v in curr_datas.items()}
        datas.update(curr_datas)
    return datas

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    print(len(lst), type(n), type(k))
    chunks = split_list(lst, n)
    return chunks[k]


def build_qwen(model_name):
    use_flash_attn = True if importlib.util.find_spec('flash_attn') else False
    print(f"Loading model from {model_name}... \nflash_attn: {use_flash_attn}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attn else 'sdpa',
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def inference(model, processor, video_path, question, nframes=64, fps=-1, resolution=448, max_tokens=4096, temperature=0.):
    assert (fps > 0) != (nframes > 0), "Specify either fps or nframes, but not both."
    print(fps, nframes, type(fps), type(nframes))

    video_content = {
        "type": "video",
        "video": video_path,
        "max_pixels": resolution * resolution,
    }
    if fps > 0:
        video_content["fps"] = fps
    elif nframes > 0:
        video_content["nframes"] = nframes

    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return response[0]

def evaluate(question, response, ground_truth, judge_model, judge_tokenizer):
    response = extract_final_answer(response)
    print(response)
    if not response:
        return False

    if judge_tokenizer:
        return llm_judge_hf(question, response, ground_truth, judge_model, judge_tokenizer)
    else:
        return llm_judge_api(question, response, ground_truth, judge_model)

def evaluate_operation(response, data, judge_model_path, judge_tokenizer, judge_implement):
    """
        Evaluate whether the operations can lead to the target state (no ground-truth answer)
    """
    demo_type = data['demo']
    response = extract_final_answer(response)
    if not response:
        return False
    print(response)
    if demo_type == "hrd":
        return eval_op_hrd(response, data['question'], \
                            data['states'][0 if data['visible_time']=='end' else -1], judge_model_path, judge_tokenizer, judge_implement)
    elif demo_type == "grid":
        return eval_op_grid(response, data['question'], \
                            data['states'][0 if data['visible_time']=='end' else -1], 
                            data['coords'][0 if data['visible_time']=='end' else -1],
                            judge_model_path, judge_implement)
    elif demo_type == "cup":
        return eval_op_cup(response, data['question'], \
                            data['states'][0 if data['visible_time']=='end' else -1], judge_model_path, judge_tokenizer, judge_implement)
    elif demo_type == "file_sys":
        states = {key: dict([list(data['states'][key].items())[0 if data['visible_time']=="end" else -1]]) for key in [f"path{i}" for i in range(int(data['num_state']))]}
        return eval_op_file(response, data['question'], \
                            states, judge_model_path, judge_tokenizer, judge_implement)
    elif demo_type == "card":
        states = {key: data['states'][key][0 if data['visible_time']=="end" else -1] for key in [f"pile{i}" for i in range(int(data['num_state']))]}
        return eval_op_card(response, data['question'], states, judge_model_path, judge_tokenizer, judge_implement)
    elif demo_type == "chip":
        states = {key: data['states'][key][0 if data['visible_time']=="end" else -1] for key in [f"cup{i}" for i in range(int(data['num_state']))]}
        return eval_op_chip(response, data['question'], states, judge_model_path, judge_tokenizer, judge_implement)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def str2bool(s):
    return s.lower() in ('true', '1', 'yes', 'y')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--judge_model_name', default='Qwen/Qwen2.5-VL-72B-Instruct')
    parser.add_argument('--judge_implement', default='api')
    parser.add_argument('--question_path', default="questions")
    parser.add_argument('--result_path', default="predictions")
    parser.add_argument('--result_prefix', default="")
    parser.add_argument('--mode', default="all", choices=["all", "infer"], help="Mode of operation: 'all' for inference and evaluation, 'infer' for only inference.")
    parser.add_argument('--resolution', default=448, type=int)
    parser.add_argument('--nframes', default=64, type=int)
    parser.add_argument('--fps', default=-1, type=float)
    parser.add_argument('--max_new_token', default=2048, type=int)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--num_chunks', default=1, type=int)
    parser.add_argument('--chunk_idx', default=0, type=int)
    args = parser.parse_args()

    random.seed(42)

    result_path = osp.join(args.result_path, f"{args.result_prefix}_{args.chunk_idx}.json")
    os.makedirs(osp.dirname(result_path), exist_ok=True)

    results = load_json(result_path) if osp.exists(result_path) else {}
    datas = load_data(args.question_path)
    items = list(datas.items())
    random.shuffle(items)
    datas = dict(items)
    datas = dict(get_chunk(list(datas.items()), args.num_chunks, args.chunk_idx))

    model, processor = build_qwen(args.model_name)

    # Inference
    print("Start inference...")
    for id, d in tqdm(datas.items()):
        if id not in results or not results[id]['prediction']:
            results[id] = d
        if "prediction" not in results[id]:
            results[id]["prediction"] = inference(model, processor, d["video"], d["question"], nframes=args.nframes, fps=args.fps, resolution=args.resolution, max_tokens=args.max_new_token, temperature=args.temperature)

        save_json(results, result_path)
    
    # Evaluation
    if args.mode == "all":
        print("Start evaluation...")
        judge_model, judge_tokenizer = build_judge(args.judge_model_name, args.judge_implement)
        for id, r in tqdm(results.items()):
            if "rating" not in results[id]:
                if r["answer"]:
                    results[id]["rating"] = evaluate(r['question'], r["prediction"], r["answer"], judge_model, judge_tokenizer)
                else:
                    results[id]["rating"] = evaluate_operation(r["prediction"], r, judge_model, judge_tokenizer, args.judge_implement)
            for k in ["states", "moves", "coords", "commands", "actions"]:
                if k in results[id]:
                    results[id].pop(k)
            save_json(results, result_path)