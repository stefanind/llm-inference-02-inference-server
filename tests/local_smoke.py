"""Run W1-W6 over local HTTP with the tiny test model: python tests/local_smoke.py.

This checks infrastructure, not GPU performance. The real server stays GPU-only.
"""

import asyncio
import json
import socket
from unittest.mock import patch

import httpx
import torch
import uvicorn

from inference_server.runner import run_workload
from inference_server.server import create_app
from inference_server.streaming import read_sse
from inference_server.workloads import make_w1, make_w2, make_w3, make_w4, make_w5, make_w6
from tiny_model import make_tiny_model


async def main():
    torch.set_num_threads(1)
    model, tokenizer = make_tiny_model()
    app = create_app(batch_wait_ms=0, max_batched_tokens=32, max_context_tokens=1024, max_cache_tokens=1024)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(512)
    url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))

    with patch("inference_server.server.load_model", return_value=(model, tokenizer)):
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async def started():
                while not server.started:
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(started(), timeout=10)
            workloads = [
                ("W1", make_w1()), ("W2", make_w2()), ("W3", make_w3()),
                ("W4", make_w4("hello", "hello " * 64)),
                ("W5", make_w5()), ("W6", make_w6(20, 0.25)),
            ]
            for name, requests in workloads:
                report = await run_workload(requests, server_url=url, workload_name=name)
                summary = report["summary"]
                metrics = report["engine_metrics"]
                assert summary["total_requests"] == len(requests)
                assert summary["successful_requests"] > 0
                assert report["client_metrics"]["in_flight_requests"] == 0
                assert report["gpu_metrics"]["average_gpu_utilization_percent"] is None
                assert metrics["peak_allocated_gpu_memory_mib"] is None
                assert summary["other_error_requests"] == summary["timed_out_requests"] == 0
                if name == "W5":
                    assert summary["rejected_requests"] > 0
                else:
                    assert summary["failed_requests"] == 0
                assert metrics["active_requests"] == metrics["reserved_cache_tokens"] == 0
                assert metrics["peak_active_requests"] <= 4 and metrics["peak_reserved_cache_tokens"] <= 1024
                assert metrics["prefill_forward_count"] > 0 and metrics["decode_forward_count"] > 0
                assert summary["average_generation_tpot_ms"] is not None
                assert report["run_info"]["server"]["model_id"] == "tiny-local-qwen3"
                print(f"{name}: {summary['successful_requests']} completed, {summary['rejected_requests']} rejected; peaks: {metrics['peak_queued_requests']} queued, {metrics['peak_active_requests']} active", flush=True)

            # Verify a live stream cannot reset counters and closing it frees the
            # cache before all 512 requested tokens have been generated.
            async with httpx.AsyncClient(base_url=url) as client:
                (await client.post("/metrics/reset")).raise_for_status()
                async with client.stream("POST", "/generate", json={"prompt": "hello", "max_new_tokens": 512}) as response:
                    response.raise_for_status()
                    async for data in read_sse(response):
                        if json.loads(data)["type"] == "text":
                            break
                    reset = await client.post("/metrics/reset")
                    assert reset.status_code == 409
                response = await client.get("/metrics", params={"wait_for_idle": "true"})
                response.raise_for_status()
                metrics = response.json()
                assert metrics["active_requests"] == metrics["reserved_cache_tokens"] == 0
                assert metrics["decode_forward_count"] < 511
                print("Cancellation: released cache early; reset correctly refused while active", flush=True)
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=10)
            listener.close()


if __name__ == "__main__":
    asyncio.run(main())
