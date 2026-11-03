#!/usr/bin/env python3
"""Run one SGLang comparison arm in an isolated Linux GPU container."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


HOST = "127.0.0.1"
PORT = 31080
URL = f"http://{HOST}:{PORT}"
ROOT = Path(__file__).resolve().parents[1]


def leader_exit(server):
    # Keep the leader unreaped so its process group ID cannot be reused before
    # cleanup, even if the server exits before its workers.
    return os.waitid(
        os.P_PID, server.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
    )


def wait_for_health(server):
    deadline = time.monotonic() + 300
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while (remaining := deadline - time.monotonic()) > 0:
        status = leader_exit(server)
        if status is not None:
            raise RuntimeError(
                f"SGLang server exited before becoming healthy (status {status.si_status})"
            )
        try:
            with opener.open(f"{URL}/health", timeout=min(remaining, 2)) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError("SGLang /health did not return HTTP 200 within 300 seconds")


def stop_server(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + 15
    while leader_exit(server) is None and time.monotonic() < deadline:
        time.sleep(0.1)

    # A worker may outlive the leader. Signal the still-owned group before
    # reaping the leader and releasing the PID.
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server.wait()


def on_sigterm(signum, _frame):
    raise SystemExit(128 + signum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--label", required=True, help="public model label")
    parser.add_argument("--mode", choices=("baseline", "v0", "v1", "v2", "v3", "v4"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workloads", nargs="+")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.model_path.is_dir():
        parser.error("--model-path must be an existing local model directory")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "server.log"
    stats_path = output / "adapter_stats.json"
    result_path = output / "results.json"
    for path in (log_path, stats_path, result_path, output / "results.correctness.json"):
        if path.exists():
            parser.error(f"output already exists: {path}; choose a new --output-dir")

    sources = ("elastic_kv/adapter.py", "elastic_kv/backing.py", "elastic_kv/launch.py",
               "native/elastic_vmm.cpp", "benchmarks/client.py", "scripts/run_case.py")
    manifest = {"mode": args.mode, "model_label": args.label,
                "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in sources}}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = {
        **os.environ,
        "EKV_MODE": args.mode,
        "EKV_STATS_PATH": str(stats_path),
        "PYTHONUNBUFFERED": "1",
    }
    server_command = [
        sys.executable, "-m", "elastic_kv.launch",
        "--model-path", str(args.model_path.resolve()),
        "--host", HOST, "--port", str(PORT),
        "--page-size", "1", "--attention-backend", "triton",
        "--disable-cuda-graph", "--disable-overlap-schedule", "--disable-radix-cache",
        "--max-total-tokens", "32768", "--context-length", "8192",
        "--chunked-prefill-size", "2048", "--mem-fraction-static", "0.5",
    ]
    client_command = [
        sys.executable, str(ROOT / "benchmarks" / "client.py"),
        "--url", URL, "--output", str(result_path),
        "--model-label", args.label, "--repeats", str(args.repeats),
    ]
    if args.workloads is not None:
        client_command.extend(("--workloads", *args.workloads))

    server = None
    previous_sigterm = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
            address.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            address.bind((HOST, PORT))
        with log_path.open("xb") as log:
            server = subprocess.Popen(
                server_command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            wait_for_health(server)
            return subprocess.run(client_command, cwd=ROOT, env=env).returncode
    except (OSError, RuntimeError) as exc:
        print(f"{exc}\nServer log: {log_path}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if server is not None:
                stop_server(server)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    sys.exit(main())
