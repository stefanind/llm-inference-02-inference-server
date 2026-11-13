"""Model loading and GPU timing from the notebook."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


MODEL_ID = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

print(f"Loaded {MODEL_ID}")
