"""Run explicitly on a GPU: python tests/gpu_smoke.py"""

import asyncio
import json
from unittest.mock import patch

import httpx

from inference_server.runner import run_workload
from inference_server.server import create_app
from inference_server.workloads import ScheduledRequest


async def main():
    app = create_app(max_batched_tokens=32)
    async with app.router.lifespan_context(app):
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        )
        requests = []
        for index, (prompt, limit) in enumerate([
            ("Say hello", 1), ("Explain KV caching. " * 30, 8),
            ("Explain why batching helps inference", 16), ("Tell a short story", 32),
            ("Name a color", 2),
        ]):
            requests.append(ScheduledRequest(str(index), None, 0, prompt, limit))
        # This in-process transport buffers output. Use the CLI for HTTP timings.
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(requests, server_url="http://test")

    for result in report["results"]:
        assert result["success"], result.get("error")
        assert result["output_tokens"] > 0
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
