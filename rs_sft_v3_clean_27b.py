"""
RS-SFT v3-clean: Qwen3.5-27B QLoRA with eval-decontaminated data
340 samples (JP80 + EN260), eval contamination zero.
Base: Qwen/Qwen3.5-27B (no prior adapter — fresh LoRA)
"""
import os, json, hashlib, subprocess, gc
os.environ["WANDB_DISABLED"] = "true"

os.system("pip install -q --upgrade transformers peft accelerate bitsandbytes datasets huggingface_hub")

DATA_DIR = "/content/data" if os.path.exists("/content") else "/kaggle/working/data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_FILE = "rs_sft_train_v3_clean.jsonl"
url = f"https://raw.githubusercontent.com/clawdia-saka/qwen-training-data/master/{TRAIN_FILE}"
subprocess.run(["wget", "-q", "-O", f"{DATA_DIR}/{TRAIN_FILE}", url], check=True)
DATA_PATH = f"{DATA_DIR}/{TRAIN_FILE}"

with open(DATA_PATH) as f:
    line_count = sum(1 for _ in f)
print(f"Data: {DATA_PATH} ({line_count} samples)")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset, random_split

assert torch.cuda.is_available(), "NO GPU"
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB)")

BASE_MODEL = "Qwen/Qwen3.5-27B"
OUTPUT_DIR = "/content/output" if os.path.exists("/content") else "/kaggle/working/rs_sft_v3_clean_27b"
MAX_LEN = 768   # Median=239, only 5 samples >1024. Saves VRAM.
EPOCHS = 2
LR = 2e-5       # Conservative for 27B
BATCH_SIZE = 1
GRAD_ACCUM = 8

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gc.collect(); torch.cuda.empty_cache()
print("Loading 27B in 4bit QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map={"": 0},
    attn_implementation="sdpa", trust_remote_code=True,
)
print(f"Model loaded: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
model.print_trainable_parameters()

# Forward sanity check
b = tokenizer("def fibonacci(n):\n    if n <= 1: return n", return_tensors="pt").to(model.device)
with torch.no_grad():
    loss = model(**b, labels=b["input_ids"]).loss.item()
assert loss < 20 and not (loss != loss), f"Forward FAILED: loss={loss}"
print(f"Forward OK loss={loss:.4f}")

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    text = tokenizer.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=False)
                    enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
                    if len(enc["input_ids"]) > 10:
                        self.samples.append(enc)
                except: pass
        print(f"Loaded {len(self.samples)} samples")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return {"input_ids": s["input_ids"], "attention_mask": s["attention_mask"], "labels": s["input_ids"].copy()}

dataset = SFTDataset(DATA_PATH, tokenizer, MAX_LEN)
eval_size = max(1, len(dataset) // 10)
train_ds, eval_ds = random_split(dataset, [len(dataset) - eval_size, eval_size])
print(f"Train: {len(train_ds)}, Eval: {eval_size}")

with open(DATA_PATH, 'rb') as f:
    print(f"hash: {hashlib.md5(f.read()).hexdigest()[:12]}")

trainer = Trainer(model=model, args=TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    logging_steps=4,
    save_steps=8,
    save_total_limit=3,
    bf16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to="none",
    eval_strategy="steps", eval_steps=20, eval_accumulation_steps=1,
    gradient_checkpointing=True,
), train_dataset=train_ds, eval_dataset=eval_ds,
   data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))

print(f"\n=== Starting 27B QLoRA v3-clean ===")
print(f"Data: {len(dataset)} clean samples (eval-decontaminated)")
trainer.train()
print("\nTraining complete!")

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if HF_TOKEN:
    from huggingface_hub import HfApi
    repo = "tetsugan/qwen3.5-27b-lora-v3-clean"
    api = HfApi()
    try: api.create_repo(repo, repo_type="model", private=True, token=HF_TOKEN)
    except: pass
    api.upload_folder(folder_path=f"{OUTPUT_DIR}/final", repo_id=repo, repo_type="model", token=HF_TOKEN)
    print(f"Uploaded to {repo}")
else:
    print("HF_TOKEN not set — skipping upload")

del model, trainer, dataset
gc.collect(); torch.cuda.empty_cache()
