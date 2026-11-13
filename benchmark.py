"""Compare KV caching locally, or use --http to benchmark the running server."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


BENCHMARK_PROMPT = "Explain why the sky looks blue in two sentences."
SERVER_URL = "http://127.0.0.1:8000"
REQUEST_COUNT = 8
MAX_NEW_TOKENS = 32


def run_local_benchmark():
    """Run the original cached/uncached comparison directly on the model."""
    # Importing generation loads the model. Keep this inside the local mode so
    # HTTP mode can use the server without loading a second model.
    from inference_server.generation import (
        generate_cached,
        generate_uncached,
        summarize_result,
    )

    generate_cached("Say hello in one sentence.", max_new_tokens=2)

    uncached_result = generate_uncached(BENCHMARK_PROMPT, max_new_tokens=24)
    cached_result = generate_cached(BENCHMARK_PROMPT, max_new_tokens=24)

    return {
        "outputs_match": uncached_result["token_ids"] == cached_result["token_ids"],
        "without_cache": summarize_result(uncached_result, used_kv_cache=False),
        "with_cache": summarize_result(cached_result, used_kv_cache=True),
    }


def send_request(client, prompt):
    """Send one request and measure time until its full answer arrives."""
    body = {
        "prompt": prompt,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "use_cache": True,
    }

    started_at = time.perf_counter()
    try:
        response = client.post("/generate", json=body)
        response.raise_for_status()
        data = response.json()
        return {
            "success": True,
            "latency_ms": (time.perf_counter() - started_at) * 1000,
            "output_tokens": data["output_tokens"],
            # This comes from the server's generation timer. It does not measure
            # when the client receives the first token: /generate sends a full answer.
            "ttft_ms": data["ttft_ms"],
        }
    except (httpx.HTTPError, ValueError, KeyError) as error:
        # Return failed requests as data so the other requests can still finish.
        return {
            "success": False,
            "latency_ms": (time.perf_counter() - started_at) * 1000,
            "error": str(error),
        }


def summarize_http_results(results, elapsed_seconds):
    """Calculate the totals using only successful responses for token counts."""
    successful_requests = 0
    total_latency_ms = 0.0
    total_ttft_ms = 0.0
    total_output_tokens = 0
    errors = []

    for result in results:
        if result["success"]:
            successful_requests += 1
            total_latency_ms += result["latency_ms"]
            total_ttft_ms += result["ttft_ms"]
            total_output_tokens += result["output_tokens"]
        else:
            errors.append(result["error"])

    # An average is undefined when every request fails. JSON prints None as null.
    average_latency_ms = None
    average_ttft_ms = None
    if successful_requests > 0:
        average_latency_ms = total_latency_ms / successful_requests
        average_ttft_ms = total_ttft_ms / successful_requests

    output_tokens_per_second = 0.0
    if elapsed_seconds > 0:
        # Use the whole experiment's duration, not the sum of request durations.
        # Concurrent requests overlap, so adding their durations would double-count time.
        output_tokens_per_second = total_output_tokens / elapsed_seconds

    return {
        "total_requests": len(results),
        "successful_requests": successful_requests,
        "failed_requests": len(errors),
        "average_latency_ms": average_latency_ms,
        "average_generation_ttft_ms": average_ttft_ms,
        "elapsed_seconds": elapsed_seconds,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": output_tokens_per_second,
        "errors": errors,
    }


def run_http_benchmark(concurrency):
    """Run the same workload with at most `concurrency` requests in flight."""
    # Reuse one client across the threads so connections can be reused too.
    with httpx.Client(base_url=SERVER_URL, timeout=120.0) as client:
        warmup = send_request(client, BENCHMARK_PROMPT)
        if not warmup["success"]:
            raise RuntimeError(f"Warmup failed. Check the server at {SERVER_URL}: {warmup['error']}")

        # The executor runs at most this many send_request() calls at once.
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            started_at = time.perf_counter()
            tasks = []
            for _ in range(REQUEST_COUNT):
                task = executor.submit(send_request, client, BENCHMARK_PROMPT)
                tasks.append(task)

            # All jobs have been submitted already. Waiting for their results
            # here does not stop the other threads from sending their requests.
            results = []
            for task in tasks:
                results.append(task.result())
            elapsed_seconds = time.perf_counter() - started_at

    summary = summarize_http_results(results, elapsed_seconds)
    summary["concurrency"] = concurrency
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Send requests to the running server at concurrency levels 1, 2, and 4",
    )
    args = parser.parse_args()

    if args.http:
        for concurrency in [1, 2, 4]:
            print(f"Running HTTP benchmark with concurrency {concurrency}", flush=True)
            try:
                summary = run_http_benchmark(concurrency)
            except RuntimeError as error:
                raise SystemExit(str(error))
            print(json.dumps(summary, indent=2), flush=True)
    else:
        print("Running local cached/uncached comparison", flush=True)
        comparison = run_local_benchmark()
        print("__TT_COMPARISON__=" + json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
