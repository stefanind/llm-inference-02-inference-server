"""Run explicitly on a GPU: python tests/gpu_smoke.py"""

import json

from fastapi.testclient import TestClient

from inference_server.server import create_app


if __name__ == "__main__":
    with TestClient(create_app(mode="real")) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/generate", json={
            "prompt": "Explain KV caching", "max_new_tokens": 32,
        })
        response.raise_for_status()
        assert response.json()["output_tokens"] > 0
        print(json.dumps(response.json(), indent=2))
