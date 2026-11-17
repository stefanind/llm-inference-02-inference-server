"""A tiny randomly initialized model for local tests; no downloads or GPU."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


def make_tiny_model():
    torch.manual_seed(42)
    words = ["[PAD]", "[UNK]", "hello", "world", "You", "are", "a", "helpful", "assistant", ".", "Request", "Explain", "KV", "caching"]
    words += [f"word{index}" for index in range(18)]
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}assistant"
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=1024, eos_token_id=None, pad_token_id=0,
    )).eval()
    model.name_or_path = "tiny-local-qwen3"
    return model, tokenizer
