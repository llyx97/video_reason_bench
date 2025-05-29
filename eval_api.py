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

import openai
from openai import OpenAI
import requests
import argparse
import math
import random
import time

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


def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode()

def image_to_bytes(image: Image.Image, format_="PNG"):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=format_)
    return image_bytes


def resize_image(image: Image.Image, max_size: int = 640, min_size: int = 28):
    width, height = image.size
    if width < min_size or height < min_size:
        # Double both dimensions while maintaining aspect ratio
        scale = min_size / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif max_size > 0 and (width > max_size or height > max_size):
        # Double both dimensions while maintaining aspect ratio
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height))

    return image

def load_video(video_file, long_edge, max_num_frames=128, target_fps=1.0):
    vr = VideoReader(video_file)
    orig_num_frms = len(vr)
    fps = vr.get_avg_fps()

    # Sample frames at target_fps
    total_duration = orig_num_frms / fps
    required_frames = int(total_duration * target_fps)
    step_size = fps / target_fps
    frame_indices = [int(i * step_size) for i in range(required_frames)]
    frame_timestamps = [i / fps for i in frame_indices]

    # If number of frames > max_num_frames, uniformly sample max_num_frames
    if len(frame_indices) > max_num_frames:
        frame_timestamps = [int(orig_num_frms / max_num_frames * (i+0.5)) / fps for i in range(max_num_frames)]
        frame_indices = [int(orig_num_frms / max_num_frames * (i+0.5)) for i in range(max_num_frames)]

    frames_data = vr.get_batch(frame_indices).asnumpy()

    imgs = []
    for idx in range(len(frame_indices)):
        img = resize_image(Image.fromarray(frames_data[idx]).convert("RGB"), long_edge)
        imgs.append(img)

    return imgs, frame_timestamps

def get_video_contents(video_path, question, max_img_size=448, max_num_frames=128, target_fps=1.0, add_timestamp=False):
    contents = []
    frames, frame_timestamps = load_video(video_path, max_img_size, max_num_frames, target_fps)

    for frame, timestamp in zip(frames, frame_timestamps, strict=True):
        frame = image_to_bytes(frame).getvalue()
        frame_base64 = encode_image_to_base64(frame)
        if add_timestamp:
            contents.append(
                                        {
                                            "type": "text",
                                            "text": f"[00:{int(timestamp)//60:02d}:{int(timestamp)%60:02d}]",
                                        }
            )
        contents.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{frame_base64}",
                                            "detail": "high",
                                        },
                                    }
        )

    contents.append( {
                                        "type": "text",
                                        "text": question
                                    }
                )
    return contents


def inference(model, video_path, question, api_type, max_num_frames=128, target_fps=1.0, resolution=448, max_tokens=2048, temperature=0., thinking_budget=8192):
    contents = get_video_contents(video_path, question, \
                                    max_img_size=resolution, max_num_frames=max_num_frames, target_fps=target_fps)

    if api_type=="openai":
        response, token_count = test_chat_completion_openai(model, contents, max_tokens, temperature)
    elif api_type=="gemini":
        response, token_count = test_chat_completion_gemini(model, question, video_path, max_new_tokens=max_tokens, temperature=temperature, thinking_budget=thinking_budget)
    return response, token_count

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
    parser.add_argument('--model_name', default='gemini-2.0-flash')
    parser.add_argument('--judge_model_name', default='gpt-4o-2024-11-20')
    parser.add_argument('--judge_implement', default='api')
    parser.add_argument('--question_path', default="questions")
    parser.add_argument('--result_path', default="predictions")
    parser.add_argument('--result_prefix', default="")
    parser.add_argument('--api_type', default="gemini", choices=["openai", "gemini"], help="api for the inference model")
    parser.add_argument('--resolution', default=448, type=int)
    parser.add_argument('--max_num_frames', default=128, type=int)
    parser.add_argument('--target_fps', default=1.0, type=float)
    parser.add_argument('--max_new_token', default=2048, type=int)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--thinking_budget', default=8192, type=int)
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

    judge_model, judge_tokenizer = build_judge(args.judge_model_name, args.judge_implement)

    for id, d in tqdm(datas.items()):
        if id not in results or not results[id]['prediction']:
            results[id] = d
        if "prediction" not in results[id]:
            results[id]["prediction"], results[id]["token_count"] = inference(args.model_name, d["video"], d["question"], args.api_type, max_num_frames=args.max_num_frames, target_fps=args.target_fps, resolution=args.resolution, max_tokens=args.max_new_token, temperature=args.temperature, thinking_budget=args.thinking_budget)
        if "rating" not in results[id]:
            if d["answer"]:
                results[id]["rating"] = evaluate(results[id]['question'], results[id]["prediction"], d["answer"], judge_model, judge_tokenizer)
            else:
                results[id]["rating"] = evaluate_operation(results[id]["prediction"], d, judge_model, judge_tokenizer, args.judge_implement)

        for k in ["states", "moves", "coords", "commands", "actions"]:
            if k in results[id]:
                results[id].pop(k)

        save_json(results, result_path)