# VideoReasonBench

## Evaluation using VLMEvalKit

### Step 1: Setup

**Installation**
```
cd vlmevalkit
pip install -e .
```

**Setup Keys**
Place the required keys in `vlmevalkit/.env` or directly set them as the environment variable.


### Step 2: Configuration
Setup the model and dataset configuration in `vlmevalkit/configs/{your_config}.json`. For example:
```python
{
    "model": {
        "Qwen2.5-VL-72B-Instruct": {
            "class": "Qwen2VLChat",
            "model_path": "Qwen/Qwen2.5-VL-72B-Instruct",
            "min_pixels": 50176,
            "max_pixels": 200704,
            "max_new_tokens": 4096
        }
    },
    "data": {
        "VideoReasoningBench_64frame": {
            "class": "VideoReasoningBench",
            "dataset": "VideoReasoningBench",
            "nframe": 64
        }
    }
}
```

### Step 3: Evaluation
```bash
torchrun --nproc-per-node=8 run.py --judge gpt-4o-1120 --config configs/video_reasoning_bench_qwen2.5-vl-7b.json --reuse        # 7B-scale model
AUTO_SPLIT=1 torchrun --nproc-per-node=1 run.py --judge gpt-4o-1120 --config configs/video_reasoning_bench_qwen2.5-vl-72b.json  # 72B-scale model
```