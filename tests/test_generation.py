"""Check the unchanged token loops with a tiny model created locally."""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from inference_server.generation import generate_cached, generate_stream, generate_uncached


class TinyTokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, **options):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, **options):
        return "".join(f"{int(token)} " for token in token_ids)


class GenerationTests(unittest.TestCase):
    def test_cached_uncached_and_streaming_outputs_agree(self):
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=None,
        )
        model = Qwen3ForCausalLM(config).eval()
        tokenizer = TinyTokenizer()

        # The test constructs its own tiny CPU model. Production loading remains
        # GPU-only; only its timing synchronization is skipped for this check.
        with patch("inference_server.generation.sync_gpu"):
            cached = generate_cached("hello", model, tokenizer, max_new_tokens=4)
            uncached = generate_uncached("hello", model, tokenizer, max_new_tokens=4)
            streamed = "".join(generate_stream("hello", model, tokenizer, max_new_tokens=4))

        self.assertEqual(len(cached["token_ids"]), 4)
        self.assertEqual(cached["token_ids"], uncached["token_ids"])
        self.assertEqual(cached["text"], streamed)


if __name__ == "__main__":
    unittest.main()
