"""Start an inference server, or run a workload against a running server."""

import argparse
import asyncio
import json

import uvicorn

from inference_server.runner import run_workload
from inference_server.server import create_app
from inference_server.workloads import make_w1, make_w2, make_w3, make_w4, make_w5, make_w6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["simulated", "real"], default="simulated",
        help="Inference mode (default: simulated)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds per simulated fixed batch (default: 2)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--queue-capacity", type=int, default=20,
        help="Maximum queued /generate requests (default: 20; one batch can also be active)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum requests per batch (default: 4)")
    parser.add_argument("--batch-wait-ms", type=float, default=10, help="Batch collection window in ms (default: 10)")
    parser.add_argument(
        "--scheduler", choices=["fixed", "continuous"], default="fixed",
        help="Scheduling mode (default: fixed)",
    )
    parser.add_argument(
        "--step-delay", type=float, default=0.01,
        help="Seconds per simulated continuous decode step (default: 0.01)",
    )

    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run", help="Submit a workload and print its JSON report")
    run_parser.add_argument("workload", choices=["W1", "W2", "W3", "W4", "W5", "W6"])
    run_parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server address")
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

    if args.command == "run":
        try:
            if args.workload == "W1":
                requests = make_w1()
            elif args.workload == "W2":
                requests = make_w2()
            elif args.workload == "W3":
                requests = make_w3(seed=args.seed)
            elif args.workload == "W4":
                if not args.short_prompt or not args.long_prompt:
                    run_parser.error("W4 requires --short-prompt and --long-prompt")
                requests = make_w4(args.short_prompt, args.long_prompt)
            elif args.workload == "W5":
                requests = make_w5()
            else:
                requests = make_w6(requests_per_second=args.rps, duration_seconds=args.duration)

            report = asyncio.run(run_workload(
                requests, server_url=args.url, timeout_seconds=args.timeout,
            ))
        except ValueError as error:
            run_parser.error(str(error))
        # Redirect stdout to a file to keep both individual results and the summary.
        print(json.dumps(report, indent=2))
        return

    try:
        app = create_app(
            mode=args.mode, delay_seconds=args.delay, queue_capacity=args.queue_capacity,
            batch_size=args.batch_size, batch_wait_ms=args.batch_wait_ms,
            scheduler=args.scheduler, step_delay_seconds=args.step_delay,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"Starting {args.mode} inference server", flush=True)
    # Keep one server process. Real mode loads one model during its startup.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
