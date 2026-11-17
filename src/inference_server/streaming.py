"""Small helpers for Server-Sent Events (SSE)."""

import json


def encode_event(event):
    """One JSON event, followed by the blank line that ends an SSE message."""
    return f"data: {json.dumps(event)}\n\n"


async def read_sse(response):
    """Read data messages, even when HTTP chunks split lines or JSON spans lines."""
    data = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line == "" and data:
            yield "\n".join(data)
            data = []
        # Ignore comments, event names, and other SSE fields.
    if data:
        raise ValueError("Stream ended in the middle of an SSE message")
