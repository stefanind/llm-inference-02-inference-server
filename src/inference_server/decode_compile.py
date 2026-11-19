"""Compile Qwen3's hidden-state RMSNorms for decode."""

import torch


class DecodeNorms:
    def __init__(self, model):
        self.active = False
        layers = model.model.layers
        # Both norms see [batch, 1, hidden_size] during decode. The batch size
        # can change. Share one function across layers, using each norm's weights.
        norm = layers[0].input_layernorm
        self.compiled_forward = torch.compile(type(norm).forward, fullgraph=True, dynamic=True)
        for layer in layers:
            self.wrap(layer.input_layernorm)
            self.wrap(layer.post_attention_layernorm)

    def wrap(self, norm):
        eager_forward = norm.forward

        def forward(hidden_states):
            if self.active:
                return self.compiled_forward(norm, hidden_states)
            return eager_forward(hidden_states)

        norm.forward = forward

    def run_decode(self, model, **inputs):
        # The server runs one model forward at a time. Reset the flag so
        # prefill uses eager mode even if its chunk has only one token.
        self.active = True
        try:
            return model(**inputs)
        finally:
            self.active = False
