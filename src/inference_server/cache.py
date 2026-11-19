"""Persistent, fixed-slot KV storage for full-attention Qwen3."""

import torch


class SlotCache:
    """One row per active slot, with reusable key and value buffers per layer."""

    def __init__(self, model, slots, tokens_per_slot):
        config = model.config
        if config.model_type != "qwen3" or any(kind != "full_attention" for kind in config.layer_types):
            raise ValueError("The fixed-slot cache requires a full-attention Qwen3 model")

        self.device = model.device
        self.dtype = model.dtype
        self.tokens_per_slot = tokens_per_slot
        shape = (slots, config.num_key_value_heads, tokens_per_slot, config.head_dim)
        # Persistent KV cache: allocate [slot, KV head, token, head feature]
        # buffers once. The same slot holds a request's history across steps.
        self.keys = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in config.layer_types]
        self.values = [torch.zeros_like(key) for key in self.keys]

    def prepare(self, jobs, sequences, pad_token_id):
        """Prepare one forward and remember where its new tokens should be written."""
        self.first_slot = min(job.cache_slot for job in jobs)
        self.last_slot = max(job.cache_slot for job in jobs) + 1
        rows = self.last_slot - self.first_slot
        width = max(len(sequence) for sequence in sequences)

        # A forward takes a slice of the slot buffers. Any holes in that slice
        # get dummy inputs and produce no cache writes or returned logits.
        inputs = [[pad_token_id] * width for _ in range(rows)]
        positions = [[0] * width for _ in range(rows)]
        output_rows = []
        write_rows, write_positions, input_columns = [], [], []
        end_positions = []
        for job, sequence in zip(jobs, sequences):
            row = job.cache_slot - self.first_slot
            end = job.cache_length + len(sequence)
            if end > self.tokens_per_slot:
                raise ValueError("Request does not fit in its KV cache slot")
            padding = width - len(sequence)
            inputs[row] = [pad_token_id] * padding + sequence
            # Positions are relative to the request's history, even when its
            # slot used to hold another request or its input has left padding.
            positions[row] = [0] * padding + list(range(job.cache_length, end))
            output_rows.append(row)
            end_positions.append(end)
            write_rows.extend([row] * len(sequence))
            write_positions.extend(range(job.cache_length, end))
            input_columns.extend(range(padding, width))

        self.key_width = max(end_positions)
        self.write_rows = torch.tensor(write_rows, device=self.device)
        self.write_positions = torch.tensor(write_positions, device=self.device)
        self.input_columns = torch.tensor(input_columns, device=self.device)
        input_ids = torch.tensor(inputs, device=self.device)
        position_ids = torch.tensor(positions, device=self.device)

        # Causal attention mask: a query can see its own and earlier positions.
        # This also hides stale data beyond the new history. A single decode
        # token can skip the mask when every real row has the same key width.
        mask = None
        if width != 1 or any(end != self.key_width for end in end_positions):
            key_positions = torch.arange(self.key_width, device=self.device)
            visible = key_positions <= position_ids[:, None, :, None]
            mask = torch.zeros(visible.shape, device=self.device, dtype=self.dtype)
            mask.masked_fill_(~visible, torch.finfo(self.dtype).min)

        # Qwen3 accepts a prepared mask, and emits logits in slot order.
        return input_ids, position_ids, {"full_attention": mask}, output_rows

    def update(self, key_states, value_states, layer_idx):
        """Qwen3 calls this once per layer before attending to the cache."""
        keys = self.keys[layer_idx][self.first_slot:self.last_slot]
        values = self.values[layer_idx][self.first_slot:self.last_slot]
        rows = self.write_rows
        # In-place KV append: copy only the new tokens into this layer's
        # buffers. Attention receives views of the history in the same storage.
        keys[rows, :, self.write_positions] = key_states[rows, :, self.input_columns]
        values[rows, :, self.write_positions] = value_states[rows, :, self.input_columns]
        return keys[:, :, :self.key_width], values[:, :, :self.key_width]
