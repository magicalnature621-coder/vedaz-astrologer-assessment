# Fine-Tuning Qwen2.5 / Qwen3 on the Vedaz Astrologer Chat Data

Your dataset is already in the right shape for supervised fine-tuning (SFT): each line is a JSON object with a `messages` array (`system` / `user` / `assistant` turns, some multi-turn). This uses LoRA/QLoRA, which is the practical choice — full fine-tuning a 7B+ model needs far more VRAM for the same result.

## 1. Pick the Base Model

- **Qwen2.5-7B-Instruct** — mature, well-documented, strong Hindi/Hinglish handling, good default choice.
- **Qwen3-8B** (or the smaller Qwen3-4B) — newer, slightly better reasoning, has a "thinking" mode you'll likely want to disable for a chat-style astrologer persona.

For a WhatsApp/chat-style product with mixed Hindi/Hinglish/English like yours, Qwen2.5-7B-Instruct is the safer starting point; try Qwen3-8B once you have a working pipeline if you want to compare quality.

## 2. Environment Setup

```bash
python3.11 -m venv ~/finetune-env
source ~/finetune-env/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft trl bitsandbytes datasets
```

GPU needed for training: a 7B model with QLoRA (4-bit) fits comfortably on a single 24 GB GPU (RTX 3090/4090, A10).

## 3. Clean & Format the Dataset

Your file mixes valid JSONL lines with some stray commas/whitespace (it looks hand-assembled). First pass: load it defensively and normalize to strict JSONL.

```python
# prepare_data.py
import json, re

def load_loose_json_objects(path):
    """Handles a file that's mostly-JSONL but has trailing commas/blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Split on lines that start a new top-level object
    objs = []
    depth = 0
    buf = ""
    for ch in raw:
        buf += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and buf.strip():
                cleaned = buf.strip().rstrip(",")
                try:
                    objs.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    pass
                buf = ""
    return objs

data = load_loose_json_objects("Chat_Data_for_assessment_of_applicants.json")
print(f"Loaded {len(data)} conversations")

with open("train.jsonl", "w", encoding="utf-8") as f:
    for conv in data:
        f.write(json.dumps({"messages": conv["messages"]}, ensure_ascii=False) + "\n")
```

Split off a small validation set (10-15%) so you can catch overfitting:

```python
import random, json

lines = open("train.jsonl", encoding="utf-8").readlines()
random.seed(42)
random.shuffle(lines)
n_val = max(1, int(len(lines) * 0.12))

open("val.jsonl", "w", encoding="utf-8").writelines(lines[:n_val])
open("train_final.jsonl", "w", encoding="utf-8").writelines(lines[n_val:])
```

With ~20 conversations in your sample this split is tiny — for a real training run you'll want this dataset grown into the hundreds/low-thousands of conversations before results generalize well. Treat the current file as a seed set, not the full corpus.

## 4. Training Script (QLoRA via TRL's SFTTrainer)

```python
# train.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

dataset = load_dataset("json", data_files={"train": "train_final.jsonl", "validation": "val.jsonl"})

def format_chat(example):
    # Uses Qwen's built-in chat template so special tokens match the base model exactly
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

dataset = dataset.map(format_chat)

sft_config = SFTConfig(
    output_dir="./vedaz-astro-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="epoch",
    bf16=True,
    max_seq_length=2048,
    dataset_text_field="text",
    packing=False,   # keep False for chat data so turns don't bleed across examples
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./vedaz-astro-lora/final")
```

Run it:
```bash
python train.py
```

## 5. Merge LoRA Weights for Serving

vLLM can load LoRA adapters directly (`--enable-lora`), but for simplicity and best inference speed, merge into a standalone model folder:

```python
# merge.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "./vedaz-astro-lora/final")
merged = model.merge_and_unload()

merged.save_pretrained("./merged_model", safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained("./merged_model")
```

This `./merged_model` folder is what you upload/scp to the VPS in the hosting write-up.

## 6. Evaluate Before Deploying

Before serving to real users, specifically test the safety behaviors your dataset was built around:
- Does it still refuse to predict death/illness/exact dates after fine-tuning? (this is the #1 regression risk — fine-tuning can dilute safety behavior if the safety examples are underrepresented)
- Does it still redirect self-harm / domestic violence / medical emergencies to helplines and professionals instead of doing astrology?
- Does it hold its ground under the "give me an exact date" pushback pattern, like in your sample conversation, rather than caving after 2-3 user turns?
- Does it stay in the language/register the user used (Hindi vs Hinglish vs English), matching one of your system prompts' explicit instruction?

A practical tip given your dataset size: oversample the safety/refusal conversations relative to the "normal" kundli-reading conversations, or your model will learn the astrology persona faster than it learns the guardrails.

## 7. Iterate

Once merged and hosted (see write-up 1), collect real conversations, have someone label problem cases, add them back into `train.jsonl`, and retrain — this is how the dataset grows from ~20 seed conversations into a production-scale set.
