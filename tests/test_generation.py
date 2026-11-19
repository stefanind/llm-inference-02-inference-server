"""Check chunked caches with a tiny local model. No GPU or downloads needed."""

import time
import unittest
from types import SimpleNamespace

import torch
from tiny_model import make_tiny_model

from inference_server.cache import SlotCache
from inference_server.engine import ActiveRequest
from inference_server.generation import build_response, finish_step, generate_step
from inference_server.metrics import EngineMetrics


class GenerationTests(unittest.TestCase):
    def test_chunked_mixed_batches_match_individual_generation(self):
        model, tokenizer = make_tiny_model()
        cache = SlotCache(model, slots=3, tokens_per_slot=12)
        prompts = [[1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14]]
        jobs = []
        expected = []
        for prompt in prompts:
            request = SimpleNamespace(max_new_tokens=3, temperature=0, top_k=None)
            jobs.append(ActiveRequest(
                request, 0, prompt_token_ids=prompt,
                generation_started_at=time.perf_counter(),
            ))
            output = model.generate(
                torch.tensor([prompt]), max_new_tokens=3, do_sample=False,
            )
            expected.append(output[0, len(prompt):].tolist())

        # Leave a hole between the first two slots, then reuse slot 0 for a
        # new request while the long prompt in slot 2 is still being processed.
        active = jobs[:2]
        active[0].cache_slot = 0
        active[1].cache_slot = 2
        waiting = jobs[2:]
        while active:
            generate_step(active, model, tokenizer, cache, max_batched_tokens=4)
            for job in active:
                if job.prefill_tokens < len(job.prompt_token_ids):
                    self.assertEqual(job.token_ids, [])
            active = [job for job in active if job.response is None]
            if waiting and len(active) < 2:
                new_job = waiting.pop()
                new_job.cache_slot = 0
                active.append(new_job)

        self.assertEqual([job.token_ids for job in jobs], expected)

    def test_stream_waits_for_a_complete_unicode_character(self):
        request = SimpleNamespace(max_new_tokens=2, temperature=0, top_k=None)
        job = ActiveRequest(request, 0, prompt_token_ids=[1], generation_started_at=time.perf_counter())
        tokenizer = SimpleNamespace(
            eos_token_id=-1, decode=lambda tokens, **kwargs: "\ufffd" if len(tokens) == 1 else "é",
        )
        logits = torch.tensor([[0.0, 1.0]])
        finish_step([job], logits, tokenizer, stop_token_ids={-1})
        self.assertEqual(job.text_delta, "")
        finish_step([job], logits, tokenizer, stop_token_ids={-1})
        self.assertEqual(job.text_delta, "é")
        self.assertEqual(job.response["finish_reason"], "length")

    def test_secondary_model_eos_stops_generation(self):
        model, tokenizer = make_tiny_model()
        # Zero logits make greedy decoding return token 0, the secondary EOS.
        with torch.no_grad():
            model.lm_head.weight.zero_()
        model.generation_config.eos_token_id = [2, 0]
        request = SimpleNamespace(max_new_tokens=3, temperature=0, top_k=None)
        job = ActiveRequest(
            request, 0, prompt_token_ids=[1], generation_started_at=time.perf_counter(), cache_slot=0,
        )
        cache = SlotCache(model, slots=1, tokens_per_slot=4)

        generate_step([job], model, tokenizer, cache)

        self.assertEqual(job.token_ids, [0])
        self.assertEqual(job.response["finish_reason"], "stop")

    def test_generation_tpot_uses_token_ready_times(self):
        job = ActiveRequest(None, 0, prompt_token_ids=[1], token_ids=[1, 2, 3], generation_started_at=1, first_token_at=2)
        self.assertEqual(build_response(job, 4, "length")["generation_tpot_ms"], 1000)
        job.token_ids = [1]
        self.assertIsNone(build_response(job, 2, "stop")["generation_tpot_ms"])

    def test_profile_records_chunked_prefill_and_decode(self):
        model, tokenizer = make_tiny_model()
        request = SimpleNamespace(max_new_tokens=2, temperature=0, top_k=None)
        job = ActiveRequest(
            request, 0, prompt_token_ids=[1, 2, 3], generation_started_at=time.perf_counter(), cache_slot=0,
        )
        cache = SlotCache(model, slots=1, tokens_per_slot=5)
        metrics = EngineMetrics(profile_enabled=True)

        for _ in range(3):
            generate_step([job], model, tokenizer, cache, max_batched_tokens=2, metrics=metrics)

        profile = metrics.snapshot()["profile"]
        for phase, expected_calls in (("prefill", 2), ("decode", 1)):
            self.assertEqual(profile[phase]["model_forward"]["calls"], expected_calls)
            for stage in ("batch_setup", "model_forward", "token_handling"):
                self.assertGreater(profile[phase][stage]["total_ms"], 0)
            self.assertEqual(profile[phase]["cache_save"]["calls"], 0)
