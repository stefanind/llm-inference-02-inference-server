"""Run explicitly on a GPU: python tests/gpu_smoke.py"""

import asyncio
import json

import httpx

from inference_server.server import create_app


async def main():
    app = create_app(mode="real", scheduler="continuous")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            calls = []
            for prompt, limit in [
                ("Say hello", 1), ("Explain KV caching", 8),
                ("Explain why batching helps inference", 16), ("Tell a short story", 32),
                ("Name a color", 2),
            ]:
                calls.append(client.post("/generate", json={
                    "prompt": prompt, "max_new_tokens": limit,
                }))
            responses = await asyncio.gather(*calls)

    for response in responses:
        response.raise_for_status()
        assert response.json()["scheduler"] == "continuous"
        assert response.json()["output_tokens"] > 0
    print(json.dumps([response.json() for response in responses], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
