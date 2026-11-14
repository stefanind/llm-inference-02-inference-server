import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"


def load_model(model_id=MODEL_ID, device="cuda:0"):
    """Load a BF16 model and its tokenizer on the selected CUDA GPU."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("This server requires a CUDA device, such as cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your GPU and PyTorch installation.")

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device_index} is unavailable")
    device = torch.device("cuda", device_index)

    print(f"Loading {model_id} on {device} with bfloat16", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # An explicit device keeps all layers on this GPU, with no CPU offloading.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=str(device),
        ).eval()
    except Exception as error:
        # Keep the original exception attached so its full cause is still visible.
        raise RuntimeError(
            f"Failed to load {model_id} on {device} with bfloat16: {error}"
        ) from error

    print(f"Loaded {model_id}", flush=True)
    return model, tokenizer


def sync_gpu(device):
    """Wait for work on this GPU before reading a benchmark timestamp."""
    torch.cuda.synchronize(device)
