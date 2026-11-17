"""Check early delivery and disconnect cleanup without loading a model."""

import asyncio
import unittest

from starlette.requests import ClientDisconnect, Request

from inference_server.engine import RequestQueue, run_worker, submit_request
from inference_server.server import stream_response
from inference_server.workloads import ScheduledRequest


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_arrives_before_completion_and_disconnect_frees_cache(self):
        async def step(active):
            for job in active:
                if job.request.prompt == "first":
                    job.kv_cache = object()
                    if not job.token_ids:
                        job.token_ids.append(1)
                        job.text_delta = "hello"
                else:
                    job.response = {"finish_reason": "length"}
            await asyncio.sleep(0)

        queue = RequestQueue(max_cache_tokens=3)
        first = submit_request(queue, ScheduledRequest("1", None, 0, "first", 2), [1])
        second = submit_request(queue, ScheduledRequest("2", None, 0, "second", 2), [1])
        worker = asyncio.create_task(run_worker(queue, step, batch_wait_ms=0))
        text_received = asyncio.Event()
        disconnected = asyncio.Event()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if b'"type": "text"' in message.get("body", b""):
                text_received.set()

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}, "method": "POST", "path": "/generate", "headers": []}
        response = stream_response(first, Request(scope, receive))
        sending = asyncio.create_task(response(scope, receive, send))
        try:
            await asyncio.wait_for(text_received.wait(), timeout=2)
            self.assertIsNone(first.response)
            disconnected.set()
            await asyncio.wait_for(sending, timeout=2)
            done = await asyncio.wait_for(second.events.get(), timeout=2)
            self.assertEqual(done["type"], "done")
            self.assertTrue(first.cancelled)
            self.assertIsNone(first.kv_cache)
            await asyncio.wait_for(queue.queue.join(), timeout=2)

            # A socket can fail before the event generator has even started.
            early = submit_request(queue, ScheduledRequest("3", None, 0, "early", 2), [1])

            async def failed_send(message):
                raise OSError("Socket closed")

            with self.assertRaises(ClientDisconnect):
                await stream_response(early, Request(scope, receive))(scope, receive, failed_send)
            self.assertTrue(early.cancelled)
            await asyncio.wait_for(queue.queue.join(), timeout=2)
        finally:
            disconnected.set()
            sending.cancel()
            worker.cancel()
            await asyncio.gather(sending, worker, return_exceptions=True)
