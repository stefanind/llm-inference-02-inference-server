"""Capture and replay the forward for one full, short-context decode batch."""

from copy import copy

import torch


GRAPH_BATCH_SIZE = 4
GRAPH_CONTEXT = 64
GRAPH_WARMUP_STEPS = 4


class FixedDecodeBatch:
    """Keep the tensor addresses and cache view fixed across replays."""

    def __init__(self, model, cache):
        self.model = model
        self.context_tokens = min(GRAPH_CONTEXT, cache.tokens_per_slot)
        device = model.device
        self.input_ids = torch.zeros((GRAPH_BATCH_SIZE, 1), device=device, dtype=torch.long)
        self.position_ids = torch.zeros_like(self.input_ids)
        self.key_positions = torch.arange(self.context_tokens, device=device)
        self.mask = torch.zeros((GRAPH_BATCH_SIZE, 1, 1, self.context_tokens), device=device, dtype=model.dtype)

        # Share the keys and values with eager forwards. The shallow copy gives
        # the graph its own indices and view while eager prepare() changes its own.
        self.cache = copy(cache)
        self.cache.first_slot = 0
        self.cache.last_slot = GRAPH_BATCH_SIZE
        self.cache.key_width = self.context_tokens
        self.cache.write_rows = torch.arange(GRAPH_BATCH_SIZE, device=device)
        self.cache.input_columns = torch.zeros_like(self.cache.write_rows)
        # The model positions and the cache write positions share storage.
        self.cache.write_positions = self.position_ids[:, 0]

    def load_step(self, tokens, positions):
        self.input_ids.copy_(tokens)
        self.position_ids.copy_(positions)
        # Mask unused positions, including history left by a previous request.
        future = self.key_positions > self.position_ids[:, None, :, None]
        self.mask.zero_().masked_fill_(future, torch.finfo(self.mask.dtype).min)

    def forward(self):
        return self.model(
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            attention_mask={"full_attention": self.mask},
            past_key_values=self.cache,
            use_cache=True,
        ).logits[:, -1, :]

    def capture(self, tokens):
        device = self.model.device
        # Use the last position for warmup. Requests start at position zero
        # and each step masks everything beyond its own current position.
        self.load_step(tokens, torch.full_like(tokens, self.context_tokens - 1))
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(GRAPH_WARMUP_STEPS):
                self.forward()
        torch.cuda.current_stream(device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = self.forward()
        return graph, logits


class DecodeGraph:
    """Adapt the fixed forward to the server's current requests."""

    @torch.inference_mode()
    def __init__(self, model, cache, pad_token_id):
        self.batch = FixedDecodeBatch(model, cache)
        tokens = torch.full_like(self.batch.input_ids, pad_token_id)
        self.graph, self.logits = self.batch.capture(tokens)

    def can_replay(self, jobs):
        # On a four-slot server a full decode batch occupies all four slots.
        return len(jobs) == GRAPH_BATCH_SIZE and all(
            job.cache_length < self.batch.context_tokens for job in jobs
        )

    def prepare(self, jobs, sequences):
        tokens = [[0] for _ in range(GRAPH_BATCH_SIZE)]
        positions = [[0] for _ in range(GRAPH_BATCH_SIZE)]
        rows = []
        for job, sequence in zip(jobs, sequences):
            slot = job.cache_slot
            tokens[slot] = sequence
            positions[slot][0] = job.cache_length
            rows.append(slot)
        device = self.batch.model.device
        self.batch.load_step(torch.tensor(tokens, device=device), torch.tensor(positions, device=device))
        return rows

    def replay(self):
        self.graph.replay()
        return self.logits
