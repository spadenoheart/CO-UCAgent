#!/usr/bin/env python3
"""Convert UCAgent trajectory SFT JSONL into LLaMA-Factory Alpaca format.

The local trajectory dataset stores full chat messages and metadata. For the
first supervised fine-tuning pass we train only on the final assistant action:
all previous messages are flattened into one instruction, and the last
assistant message becomes the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "你是 UCAgent 芯片验证代理。你需要根据当前任务阶段、工具输出、checker 报错、"
    "已有文档和测试上下文，选择下一步最小有效动作。优先保证测试有有效 assert、"
    "覆盖标记与功能点一致、bug 文档符合 schema，并避免无效重复修复循环。"
)


def flatten_context(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, msg in enumerate(messages, 1):
        role = str(msg.get("role") or "unknown")
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"### Message {idx} ({role})\n{content}")
    return "\n\n".join(parts)


def convert_file(src: Path, dst: Path, *, max_instruction_chars: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue
            instruction = flatten_context(messages[:-1])
            if max_instruction_chars > 0 and len(instruction) > max_instruction_chars:
                instruction = (
                    "[TRUNCATED_PREFIX: context tail kept for LLaMA-Factory SFT]\n"
                    + instruction[-max_instruction_chars:]
                )
            output = str(messages[-1].get("content") or "").strip()
            if not instruction or not output:
                continue
            converted = {
                "id": row.get("id", ""),
                "system": SYSTEM_PROMPT,
                "instruction": instruction,
                "input": "",
                "output": output,
                "metadata": row.get("metadata", {}),
            }
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_dataset_info(out_dir: Path) -> None:
    dataset_info = {
        "ucagent_traj_train": {
            "file_name": "ucagent_traj_train.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        "ucagent_traj_val": {
            "file_name": "ucagent_traj_val.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        "ucagent_traj_test": {
            "file_name": "ucagent_traj_test.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_train_yaml(out_dir: Path) -> None:
    yaml_text = """\
### model
# Replace this with the HF/safetensors base model matching your Ollama qwen35moe GGUF.
model_name_or_path: /data/models/Qwen3.5-Coder-122B-HF
trust_remote_code: true
quantization_bit: 4
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### dataset
dataset_dir: ./benchmark/ucagent_finetune_dataset/llamafactory_data
dataset: ucagent_traj_train
eval_dataset: ucagent_traj_val
template: qwen3_nothink
enable_thinking: false
cutoff_len: 8192
preprocessing_num_workers: 8
dataloader_num_workers: 2

### output
output_dir: ./saves/qwen35moe-122b/ucagent-lora-sft
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
"""
    (out_dir / "ucagent_qwen35moe_lora_sft.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="benchmark/ucagent_finetune_dataset",
        help="Directory containing trajectory_sft_{train,val,test}.jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark/ucagent_finetune_dataset/llamafactory_data",
        help="Output directory used as LLaMA-Factory dataset_dir.",
    )
    parser.add_argument(
        "--max-instruction-chars",
        type=int,
        default=120000,
        help="Character cap before LLaMA-Factory tokenization. Use 0 to disable.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": convert_file(
            source_dir / "trajectory_sft_train.jsonl",
            out_dir / "ucagent_traj_train.jsonl",
            max_instruction_chars=args.max_instruction_chars,
        ),
        "val": convert_file(
            source_dir / "trajectory_sft_val.jsonl",
            out_dir / "ucagent_traj_val.jsonl",
            max_instruction_chars=args.max_instruction_chars,
        ),
        "test": convert_file(
            source_dir / "trajectory_sft_test.jsonl",
            out_dir / "ucagent_traj_test.jsonl",
            max_instruction_chars=args.max_instruction_chars,
        ),
    }
    write_dataset_info(out_dir)
    write_train_yaml(out_dir)

    print(json.dumps({"out_dir": str(out_dir), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
