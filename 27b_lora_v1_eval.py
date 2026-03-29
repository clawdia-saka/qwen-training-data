import subprocess
subprocess.run(['pip', 'install', '-q', '--upgrade', 'transformers', 'peft', 'accelerate', 'huggingface_hub', 'bitsandbytes'], check=True)
import transformers, peft, torch
print(f'transformers={transformers.__version__} peft={peft.__version__} torch={torch.__version__}')
assert torch.cuda.is_available(), 'NO GPU'
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)')
import os, json, re, time, gc
os.environ['WANDB_DISABLED'] = 'true'
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
BASE = 'Qwen/Qwen3.5-27B'
ADAPTER = 'tetsugan/qwen3.5-27b-lora-v1'
subprocess.run(['wget', '-q', '-O', '/content/gate34.jsonl', 'https://raw.githubusercontent.com/clawdia-saka/qwen-training-data/master/gate34.jsonl'], check=True)
tasks = [json.loads(l) for l in open('/content/gate34.jsonl')]
print(f'Tasks: {len(tasks)}')
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
if vram >= 75:
    print('A100 80GB: bf16')
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa', trust_remote_code=True)
elif vram >= 35:
    print('A100 40GB: 4bit')
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True), device_map='auto', attn_implementation='sdpa', trust_remote_code=True)
else:
    raise RuntimeError(f'Need A100, got {vram:.0f}GB')
print(f'Loaded: {torch.cuda.memory_allocated(0)/1e9:.2f}GB')
def generate(prompt):
    msgs = [{'role':'system','content':'Generate Python code.'},{'role':'user','content':prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    resp = re.sub(r'<think>.*?</think>','',resp,flags=re.DOTALL).strip()
    for tag in ['```python', '```']:
        if tag in resp:
            return resp.split(tag)[1].split('```')[0].strip()
    return resp
def run_tests(code, tcs):
    p = 0
    for tc in tcs:
        try:
            ns = {}; exec(code, ns)
            if tc.get('setup'): exec(tc['setup'], ns)
            ns['result'] = eval(tc['call'], ns)
            if all(eval(a, ns) for a in tc.get('asserts',[])): p += 1
        except: pass
    return p
def ev(label):
    scores, bk = [], {}
    for i, t in enumerate(tasks):
        msg = next((m['content'] for m in t['messages'] if m['role']=='user'),'')
        hid = t['ground_truth'].get('hidden_tests', t['ground_truth'].get('test_cases',[]))
        if not msg or not hid: continue
        b = t.get('bucket','?')
        code = generate(msg)
        try: compile(code,'<t>','exec'); ok=True
        except: ok=False
        p = run_tests(code, hid) if ok else 0
        sc = p/len(hid); scores.append(sc)
        bk.setdefault(b,[]).append(sc)
        if (i+1)%10==0: print(f'  [{i+1}/{len(tasks)}] {sum(scores)/len(scores):.3f}')
    avg = sum(scores)/len(scores)
    print(f'\n{label}: {avg:.3f}')
    for b in sorted(bk): print(f'  {b}: {sum(bk[b])/len(bk[b]):.3f}')
    return avg, bk
print('=== BASE ===')
a1, b1 = ev('Base')
model = PeftModel.from_pretrained(model, ADAPTER); model.eval()
print('\n=== LoRA v1 ===')
a2, b2 = ev('LoRA')
print(f'\nDelta: {a2-a1:+.3f}')
