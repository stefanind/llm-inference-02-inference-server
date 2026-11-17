import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"


def load_model(model_id=MODEL_ID, device="cuda:0", revision=None):
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
        # Resolve a branch/tag once, then use that exact commit for both loads.
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        resolved_revision = getattr(config, "_commit_hash", None) or revision
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved_revision)
        # Left padding keeps every prompt's last real token at the batch's end.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # An explicit device keeps all layers on this GPU, with no CPU offloading.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            revision=resolved_revision,
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
