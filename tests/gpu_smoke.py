"""Run explicitly on a GPU: python tests/gpu_smoke.py"""

import asyncio
import json

import httpx

from inference_server.server import create_app


async def main():
    app = create_app(mode="real")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            calls = []
            for limit in [4, 8, 16, 32]:
                calls.append(client.post("/generate", json={
                    "prompt": "Explain KV caching", "max_new_tokens": limit,
                }))
            responses = await asyncio.gather(*calls)

    for response in responses:
        response.raise_for_status()
        assert response.json()["batch_size"] == 4
        assert response.json()["output_tokens"] > 0
    print(json.dumps([response.json() for response in responses], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
