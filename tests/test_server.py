"""Run the notebook smoke test with: python tests/test_server.py"""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from inference_server.server import app


def test_server():
    # The context starts the server lifespan, including its one model load.
    with TestClient(app) as client:
        health_response = client.get("/health")
        generation_response = client.post(
            "/generate",
            json={
                "prompt": "Explain KV caching",
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_k": None,
                "use_kv_cache": True,
            },
        )

        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}
        assert generation_response.status_code == 200
        assert generation_response.json()["output_tokens"] > 0

        print(json.dumps(generation_response.json(), indent=2))


class ServerLoadingTests(unittest.TestCase):
    def test_startup_loads_once_and_all_endpoints_share_the_objects(self):
        model = object()
        tokenizer = object()
        result = {
            "text": "hello", "token_ids": [1], "input_tokens": 3,
            "ttft_ms": 1, "latency_ms": 2,
        }
        with (
            patch("inference_server.server.load_model", return_value=(model, tokenizer)) as load,
            patch("inference_server.server.generate_cached", return_value=result) as cached,
            patch("inference_server.server.generate_uncached", return_value=result) as uncached,
            patch("inference_server.server.generate_stream", return_value=iter(["hello"])) as stream,
        ):
            load.assert_not_called()
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.post("/generate", json={"prompt": "hi"}).status_code, 200)
                response = client.post("/generate", json={"prompt": "hi", "use_kv_cache": False})
                self.assertFalse(response.json()["used_kv_cache"])
                self.assertEqual(client.post("/stream", json={"prompt": "hi"}).text, "hello")
                load.assert_called_once()

            for generate in [cached, uncached, stream]:
                self.assertIs(generate.call_args.kwargs["model"], model)
                self.assertIs(generate.call_args.kwargs["tokenizer"], tokenizer)
            self.assertFalse(hasattr(app.state, "model"))
            self.assertFalse(hasattr(app.state, "tokenizer"))

    def test_failed_loading_stops_startup(self):
        with patch("inference_server.server.load_model", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                with TestClient(app):
                    self.fail("The server must not start after model loading fails")


if __name__ == "__main__":
    test_server()
