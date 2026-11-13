"""Run the notebook smoke test with: python tests/test_server.py"""

import json

from fastapi.testclient import TestClient

from inference_server.server import app


def test_server():
    client = TestClient(app)

    health_response = client.get("/health")
    generation_response = client.post(
        "/generate",
        json={
            "prompt": "Explain KV caching",
            "max_new_tokens": 32,
            "temperature": 0.0,
            "top_k": None,
            "use_cache": True,
        },
    )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert generation_response.status_code == 200
    assert generation_response.json()["output_tokens"] > 0

    print(json.dumps(generation_response.json(), indent=2))


if __name__ == "__main__":
    test_server()
