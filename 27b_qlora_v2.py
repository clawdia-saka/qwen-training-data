# 27B QLoRA v2 — Clean Data, Canary Run
# Base: Qwen/Qwen3.5-27B
# Data: rs_sft_train_v2.jsonl (275 clean samples)
# Purpose: smoke test / canary — NOT production run
#
# Changes from v1:
# - Cleaned data: removed 102 diff patches, 58 syntax errors
# - Added 13 DPO chosen + 2 HardPublic gold
# - 1 epoch (was 2) — canary only
# - lr 2e-5 (was 3e-5) — more conservative

import subprocess, os, json, gc
subprocess.run(['pip', 'install', '-q', '--upgrade', 'transformers', 'peft', 'accelerate', 'huggingface_hub', 'bitsandbytes', 'datasets', 'trl'], check=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

assert torch.cuda.is_available(), 'NO GPU'
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu} ({vram:.1f}GB)')

# Config
BASE = 'Qwen/Qwen3.5-27B'
DATA_URL = 'https://raw.githubusercontent.com/clawdia-saka/qwen-training-data/master/rs_sft_train_v2.jsonl'
OUTPUT = 'tetsugan/qwen3.5-27b-lora-v2'

# Download data
subprocess.run(['wget', '-q', '-O', '/content/train.jsonl', DATA_URL], check=True)
raw = [json.loads(l) for l in open('/content/train.jsonl')]
print(f'Training samples: {len(raw)}')

# Stats
sources = {}
for r in raw:
    s = r.get('_source', '?')
    sources[s] = sources.get(s, 0) + 1
print(f'Sources: {sources}')

# Tokenizer
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Format as chat
def format_chat(example):
    text = tok.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
    return {'text': text}

ds = Dataset.from_list(raw)
ds = ds.map(format_chat, remove_columns=ds.column_names)
print(f'Dataset: {len(ds)} samples')

# Check token lengths
sample_lengths = []
for i in range(min(20, len(ds))):
    tokens = tok(ds[i]['text'], return_tensors='pt')
    sample_lengths.append(tokens['input_ids'].shape[1])
print(f'Token lengths (sample 20): min={min(sample_lengths)} max={max(sample_lengths)} avg={sum(sample_lengths)/len(sample_lengths):.0f}')

# Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map='auto',
    attn_implementation='sdpa',
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
print(f'Model loaded: {torch.cuda.memory_allocated(0)/1e9:.2f}GB')

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training — CANARY: 1 epoch only
training_args = SFTConfig(
    output_dir='/content/qwen35-27b-lora-v2',
    num_train_epochs=1,  # canary
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,  # conservative
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    max_seq_length=2048,
    packing=False,
    report_to='none',
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tok,
)

print('Starting training...')
trainer.train()
print('Training complete!')

# Save
trainer.save_model('/content/qwen35-27b-lora-v2-final')
tok.save_pretrained('/content/qwen35-27b-lora-v2-final')

# Push to hub
from huggingface_hub import login
login()
model.push_to_hub(OUTPUT, commit_message='v2: 275 clean samples, 1 epoch canary')
tok.push_to_hub(OUTPUT)
print(f'Pushed to {OUTPUT}')

# Summary
final_loss = trainer.state.log_history[-1].get('train_loss', '?')
print(f'\nFinal train loss: {final_loss}')
print(f'Samples: {len(raw)}')
print(f'Epochs: 1 (canary)')
print(f'LR: 2e-5')
print(f'Output: {OUTPUT}')
