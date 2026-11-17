"""Check revision selection without loading weights or using CUDA."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from inference_server.model import load_model


class ModelTests(unittest.TestCase):
    def test_weights_and_tokenizer_use_the_same_resolved_commit(self):
        config = SimpleNamespace(_commit_hash="resolved-commit")
        tokenizer = SimpleNamespace(pad_token_id=0)
        with (
            patch("inference_server.model.torch.cuda.is_available", return_value=True),
            patch("inference_server.model.torch.cuda.device_count", return_value=1),
            patch("inference_server.model.AutoConfig.from_pretrained", return_value=config),
            patch("inference_server.model.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
            patch("inference_server.model.AutoModelForCausalLM.from_pretrained") as load_weights,
        ):
            load_model(revision="requested-tag")
        self.assertEqual(load_tokenizer.call_args.kwargs["revision"], "resolved-commit")
        self.assertEqual(load_weights.call_args.kwargs["revision"], "resolved-commit")
