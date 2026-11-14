"""Check GPU loading choices without a GPU or pretrained weights."""

import unittest
from unittest.mock import Mock, patch

import torch

from inference_server.model import load_model, sync_gpu


class ModelTests(unittest.TestCase):
    def setUp(self):
        # Only the CUDA calls are replaced. No real GPU is initialized by these tests.
        cuda_patch = patch("inference_server.model.torch.cuda")
        self.cuda = cuda_patch.start()
        self.addCleanup(cuda_patch.stop)
        self.cuda.is_available.return_value = True
        self.cuda.device_count.return_value = 2
        self.cuda.current_device.return_value = 0

    def test_invalid_hardware_fails_before_loading_weights(self):
        with patch("inference_server.model.AutoTokenizer.from_pretrained") as load_tokenizer:
            with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
                load_model(device="cpu")
            with self.assertRaisesRegex(ValueError, "device 2 is unavailable"):
                load_model(device="cuda:2")

            self.cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                load_model()
            load_tokenizer.assert_not_called()

    def test_load_uses_bfloat16_on_the_requested_gpu(self):
        model = Mock()
        model.eval.return_value = model
        tokenizer = object()
        with (
            patch("inference_server.model.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("inference_server.model.AutoModelForCausalLM.from_pretrained", return_value=model) as load_weights,
        ):
            loaded_model, loaded_tokenizer = load_model("test-model", device="cuda:1")

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_tokenizer, tokenizer)
        load_weights.assert_called_once_with(
            "test-model", dtype=torch.bfloat16, device_map="cuda:1"
        )
        model.eval.assert_called_once()

    def test_loading_failure_keeps_the_original_cause(self):
        original_error = RuntimeError("CUDA out of memory")
        with (
            patch("inference_server.model.AutoTokenizer.from_pretrained"),
            patch("inference_server.model.AutoModelForCausalLM.from_pretrained", side_effect=original_error),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to load test-model on cuda:0") as raised:
                load_model("test-model")

        self.assertIs(raised.exception.__cause__, original_error)

    def test_synchronization_uses_the_requested_gpu(self):
        sync_gpu("cuda:1")
        self.cuda.synchronize.assert_called_once_with("cuda:1")


if __name__ == "__main__":
    unittest.main()
