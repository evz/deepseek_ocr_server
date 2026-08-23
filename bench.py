#!/usr/bin/env python3
"""
Throughput/latency benchmark for the DeepSeek-OCR server.

Two things this is for:

1. Picking --workers. The right number is whatever stops improving
   throughput on your GPU; guessing from VRAM alone gets you close, this
   tells you the answer. Sweep it:

       python bench.py --images testdata/bevreg-5007-298-sample \\
              --concurrency 1,2,4,8 --repeat 3 --warmup

2. Proving an optimisation did not change the output. Dump results from two
   server configurations and diff them:

       python server.py --streaming-path &     # original inference path
       python bench.py --images IMAGES --concurrency 1 --save-dir out-old
       # restart the server without --streaming-path
       python bench.py --images IMAGES --concurrency 1 --save-dir out-new
       diff -r out-old out-new

Concurrency here means "how many client sockets are in flight at once". A
single REQ socket is lockstep by design, so parallelism comes from opening
several - which is also how a real client should drive this server.
"""
import argparse
import base64
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import zmq

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def collect_images(paths):
    images = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            images.extend(sorted(
                p for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            images.append(path)
        else:
            raise SystemExit(f"No such file or directory: {path}")
    if not images:
        raise SystemExit("No images found")
    return images


def worker(endpoint, timeout_ms, queue, results, errors, save_dir, request_params, lock):
    """One client socket, pulling jobs until the queue is empty."""
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)

    try:
        while True:
            try:
                path, encoded = queue.get_nowait()
            except Empty:
                return

            request = dict(request_params, image=encoded)
            started = time.perf_counter()
            try:
                socket.send_string(json.dumps(request))
                response = json.loads(socket.recv_string())
            except zmq.ZMQError as exc:
                with lock:
                    errors.append((path.name, f'transport: {exc}'))
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000

            if response.get('error'):
                with lock:
                    errors.append((path.name, response['error']))
                continue

            server_ms = response.get('metadata', {}).get('processing_time_ms')
            with lock:
                results.append((elapsed_ms, server_ms))
            if save_dir is not None:
                (save_dir / f'{path.stem}.txt').write_text(response['text'])
    finally:
        socket.close()


def percentile(values, fraction):
    if not values:
        return float('nan')
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def run_round(endpoint, jobs, concurrency, timeout_ms, save_dir, request_params):
    queue = Queue()
    for job in jobs:
        queue.put(job)

    results, errors, lock = [], [], threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(endpoint, timeout_ms, queue, results, errors, save_dir,
                  request_params, lock),
            daemon=True)
        for _ in range(concurrency)
    ]

    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    return wall, results, errors


def main():
    parser = argparse.ArgumentParser(description='Benchmark the DeepSeek-OCR server')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--images', nargs='+', required=True,
                        help='Image files and/or directories to send')
    parser.add_argument('--mode', default='base',
                        choices=['tiny', 'small', 'base', 'large', 'gundam'])
    parser.add_argument('--preserve-layout', action='store_true', default=True)
    parser.add_argument('--no-preserve-layout', dest='preserve_layout',
                        action='store_false')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                        help='Per-request generation cap')
    parser.add_argument('--concurrency', default='1',
                        help='In-flight requests; comma-separate to sweep, e.g. 1,2,4')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Passes over the image set per concurrency level')
    parser.add_argument('--timeout', type=int, default=600000,
                        help='Per-request timeout in ms (default: 10 minutes)')
    parser.add_argument('--save-dir', default=None,
                        help='Write each result to <dir>/<image stem>.txt for diffing')
    parser.add_argument('--warmup', action='store_true',
                        help='Send one throwaway request first so cudnn autotuning and '
                             'allocator warmup are not charged to the measurement')
    args = parser.parse_args()

    endpoint = f'tcp://{args.host}:{args.port}'
    images = collect_images(args.images)
    levels = [int(x) for x in args.concurrency.split(',') if x.strip()]

    request_params = {'mode': args.mode, 'preserve_layout': args.preserve_layout}
    if args.max_new_tokens is not None:
        request_params['max_new_tokens'] = args.max_new_tokens

    print(f"endpoint={endpoint} mode={args.mode} images={len(images)} "
          f"repeat={args.repeat}")

    # Encode once and reuse: base64-ing a 600 DPI page takes long enough to
    # show up in the numbers, and it is client-side work we are not measuring.
    encoded = [(path, base64.b64encode(path.read_bytes()).decode('ascii'))
               for path in images]

    if args.warmup:
        print("warmup...", end='', flush=True)
        run_round(endpoint, encoded[:1], 1, args.timeout, None, request_params)
        print(" done")

    print()
    header = f"{'conc':>5} {'ok':>5} {'err':>4} {'wall s':>8} {'req/s':>7} " \
             f"{'p50 ms':>9} {'p90 ms':>9} {'max ms':>9} {'server p50':>11}"
    print(header)
    print('-' * len(header))

    for concurrency in levels:
        jobs = encoded * args.repeat
        save_dir = None
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        wall, results, errors = run_round(
            endpoint, jobs, concurrency, args.timeout, save_dir, request_params)

        latencies = [r[0] for r in results]
        server_times = [r[1] for r in results if r[1] is not None]
        throughput = len(results) / wall if wall > 0 else float('nan')
        print(f"{concurrency:>5} {len(results):>5} {len(errors):>4} {wall:>8.1f} "
              f"{throughput:>7.2f} {percentile(latencies, 0.50):>9.0f} "
              f"{percentile(latencies, 0.90):>9.0f} "
              f"{max(latencies) if latencies else float('nan'):>9.0f} "
              f"{percentile(server_times, 0.50):>11.0f}")

        for name, message in errors[:5]:
            print(f"      ! {name}: {message}", file=sys.stderr)
        if len(errors) > 5:
            print(f"      ! ... and {len(errors) - 5} more errors", file=sys.stderr)

    print("\nThroughput that stops rising with concurrency means the GPU is "
          "saturated;\nif it is still rising at your highest level, try more "
          "--workers on the server.")


if __name__ == '__main__':
    main()
