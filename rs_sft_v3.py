"""
RS-SFT v3: Qwen3.5-9B Fine-tuning with Japanese + English data
358 samples: 278 EN code + 30 JP code + 15 JP/EN switch + 15 JP ops + 20 JP analysis
Base adapter: tetsugan/qwen3.5-9b-rs-sft-v2
"""
import os, json, hashlib, subprocess
os.environ["WANDB_DISABLED"] = "true"

# === Install (Qwen3.5 requires latest transformers) ===
os.system("pip install -q --upgrade transformers peft accelerate bitsandbytes datasets huggingface_hub")

# === Download data from GitHub ===
DATA_DIR = "/kaggle/working/data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_FILE = "rs_sft_train_v2.jsonl"
url = f"https://raw.githubusercontent.com/clawdia-saka/qwen-training-data/master/{TRAIN_FILE}"
subprocess.run(["wget", "-q", "-O", f"{DATA_DIR}/{TRAIN_FILE}", url], check=True)
print(f"Downloaded {TRAIN_FILE}")

DATA_PATH = f"{DATA_DIR}/{TRAIN_FILE}"
if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) < 1000:
    import glob
    candidates = glob.glob(f"/kaggle/input/**/{TRAIN_FILE}", recursive=True)
    if candidates:
        DATA_PATH = candidates[0]
    else:
        raise FileNotFoundError(f"Cannot find {TRAIN_FILE}")

with open(DATA_PATH) as f:
    line_count = sum(1 for _ in f)
print(f"Using data: {DATA_PATH}")
print(f"Samples: {line_count}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.utils.data import Dataset

# === Config ===
BASE_MODEL = "Qwen/Qwen3.5-9B"
ADAPTER = "tetsugan/qwen3.5-9b-rs-sft-v2"
OUTPUT_DIR = "/kaggle/working/rs_sft_v3"
MAX_LEN = 1536  # Increased for JP analysis tasks (longer prompts)
EPOCHS = 2
LR = 3e-5  # Slightly lower for continued training

# === Load model with 4bit ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
    trust_remote_code=True,
)

# Load existing v2 adapter and merge for continued training
print(f"Loading adapter: {ADAPTER}")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
print("Adapter merged")

# Apply new LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Dataset ===
class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        skipped = 0
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    msgs = d.get("messages", [])
                    if not msgs:
                        skipped += 1
                        continue
                    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
                    if len(enc["input_ids"]) > 10:
                        self.samples.append(enc)
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
        print(f"Loaded {len(self.samples)} samples (skipped {skipped})")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"input_ids": s["input_ids"], "attention_mask": s["attention_mask"], "labels": s["input_ids"].copy()}

dataset = SFTDataset(DATA_PATH, tokenizer, MAX_LEN)

# Data integrity check
with open(DATA_PATH, 'rb') as f:
    data_hash = hashlib.md5(f.read()).hexdigest()[:12]
print(f"train_hash: {data_hash}")
print(f"dataset_len: {len(dataset)}")

# === Training ===
# With 358 samples, batch_size=1, grad_accum=8 -> ~45 steps/epoch
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=LR,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=4,
    save_steps=16,
    save_total_limit=3,
    fp16=False,
    bf16=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

print(f"\n=== Starting RS-SFT v3 ===")
print(f"Samples: {len(dataset)} | Epochs: {EPOCHS} | LR: {LR}")
print(f"Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
print(f"Max seq len: {MAX_LEN}")

trainer.train()

# === Save ===
final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Save training metadata
meta = {
    "version": "rs-sft-v3",
    "base_model": BASE_MODEL,
    "base_adapter": ADAPTER,
    "data_file": TRAIN_FILE,
    "data_hash": data_hash,
    "samples": len(dataset),
    "epochs": EPOCHS,
    "lr": LR,
    "max_len": MAX_LEN,
    "lora_r": 16,
    "lora_alpha": 32,
}
with open(os.path.join(final_dir, "training_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ RS-SFT v3 DONE — {len(dataset)} samples, {EPOCHS} epochs")
print(f"Output: {final_dir}")
