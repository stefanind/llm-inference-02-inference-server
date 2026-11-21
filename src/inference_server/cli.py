"""Start the GPU inference server on Runpod, or submit a workload to it."""

import argparse
import asyncio
import json
from pathlib import Path

import uvicorn
import httpx

from inference_server.runner import run_workload
from inference_server.workloads import make_w1, make_w2, make_w3, make_w4, make_w5, make_w6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Address to listen on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--revision", help="Model/tokenizer commit or tag; reports record the resolved commit")
    parser.add_argument("--profile", action="store_true", help="Time prefill and decode stages with CUDA synchronization")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compile-decode", action="store_true", help="Compile Qwen3's decoder-layer RMSNorms during decode")
    modes.add_argument("--cuda-graph-decode", action="store_true", help="Replay full four-request decode batches with short histories")
    parser.add_argument(
        "--queue-capacity", type=int, default=20,
        help="Maximum queued /generate requests (default: 20)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum active requests (default: 4)")
    parser.add_argument("--batch-wait-ms", type=float, default=10, help="Initial collection window in ms (default: 10)")
    parser.add_argument(
        "--max-batched-tokens", type=int, default=512,
        help="Input tokens per scheduling iteration; at least --batch-size (default: 512)",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=4096,
        help="Prompt plus output limit per request, capped by the model's context (default: 4096)",
    )
    parser.add_argument(
        "--max-cache-tokens", type=int, default=8192,
        help="Shared cache-token reservation budget for active requests (default: 8192)",
    )

    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run", help="Submit a workload and print its JSON report")
    run_parser.add_argument("workload", choices=["W1", "W2", "W3", "W4", "W5", "W6"])
    run_parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL, without /v1")
    run_parser.add_argument("--backend", choices=["repo", "vllm"], default="repo", help="Server API (default: repo)")
    run_parser.add_argument("--model", help="Served model name; required with --backend vllm")
    run_parser.add_argument("--server-info", help="vLLM server configuration JSON file to record as supplied")
    run_parser.add_argument("--gpu-uuid", help="Server GPU UUID to sample via NVML; run this client on the GPU host")
    run_parser.add_argument(
        "--timeout", type=float, default=60,
        help="HTTP operation timeout in seconds (default: 60)",
    )
    run_parser.add_argument("--seed", type=int, default=42, help="W3 random seed")
    run_parser.add_argument("--short-prompt", help="W4 prepared short prompt text")
    run_parser.add_argument("--long-prompt", help="W4 prepared long prompt text")
    run_parser.add_argument("--rps", type=float, default=1, help="W6 requests per second")
    run_parser.add_argument("--duration", type=float, default=30, help="W6 arrival-window seconds")
    args = parser.parse_args()

    # Benchmark client entry point. Without "run", this process starts the server.
    if args.command == "run":
        workload_settings = {}
        try:
            server_info = None
            if args.server_info:
                server_info = json.loads(Path(args.server_info).read_text(encoding="utf-8"))
            if args.workload == "W1":
                requests = make_w1()
            elif args.workload == "W2":
                requests = make_w2()
            elif args.workload == "W3":
                requests = make_w3(seed=args.seed)
                workload_settings = {"seed": args.seed}
            elif args.workload == "W4":
                if not args.short_prompt or not args.long_prompt:
                    run_parser.error("W4 requires --short-prompt and --long-prompt")
                requests = make_w4(args.short_prompt, args.long_prompt)
            elif args.workload == "W5":
                requests = make_w5()
            else:
                requests = make_w6(requests_per_second=args.rps, duration_seconds=args.duration)
                workload_settings = {"rps": args.rps, "duration_seconds": args.duration}

            report = asyncio.run(run_workload(
                requests, server_url=args.url, timeout_seconds=args.timeout,
                backend=args.backend, model=args.model,
                workload_name=args.workload, workload_settings=workload_settings,
                server_info=server_info,
                gpu_uuid=args.gpu_uuid,
            ))
        except (ValueError, OSError, httpx.HTTPError) as error:
            run_parser.error(str(error))
        # Redirect stdout to a file to keep both individual results and the summary.
        print(json.dumps(report, indent=2))
        return

    # Workload clients and --help do not need to import the model libraries.
    from inference_server.server import create_app

    try:
        app = create_app(
            queue_capacity=args.queue_capacity,
            batch_size=args.batch_size, batch_wait_ms=args.batch_wait_ms,
            max_batched_tokens=args.max_batched_tokens,
            max_context_tokens=args.max_context_tokens, max_cache_tokens=args.max_cache_tokens,
            revision=args.revision, profile=args.profile, compile_decode=args.compile_decode,
            cuda_graph_decode=args.cuda_graph_decode,
        )
    except ValueError as error:
        parser.error(str(error))

    print("Starting GPU inference server", flush=True)
    # Keep one server process so the model is loaded only once.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
