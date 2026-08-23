#!/usr/bin/env python3
"""
DeepSeek-OCR Remote Inference Server

Runs on a desktop GPU, accepts image processing requests via ZMQ.

Architecture
------------
Requests arrive on a ROUTER socket and are handed to a pool of worker
processes, each holding its own model replica:

    clients (REQ) --> tcp://*:5555 [ROUTER]
                            |
                     load-balancing broker
                            |
                       ipc:// [ROUTER]
                    /       |       \
                worker-0 worker-1 worker-2      (separate processes)

Why processes and not threads: a decode step is only a few milliseconds of
GPU work but a comparable amount of Python (kernel launches, logits
processors, stopping criteria), so a single-replica server spends much of
its time with the GPU idle and one core pinned. Threads cannot fix that -
they share a GIL. Separate interpreters can: while worker 0 is in Python,
worker 1's kernels have the GPU, and each worker occupies a different core.

The broker dispatches strictly to idle workers rather than round-robin,
which matters here because request durations vary by more than an order of
magnitude (a `tiny` page against a `gundam` one). Round-robin would park a
2-second request behind a 40-second one while another worker sat idle.

The wire protocol is unchanged: ROUTER accepts plain REQ clients, so
existing clients work against this server with no edits.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from typing import Dict, Optional

import zmq

from ocr_engine import READY, worker_main

# Rough per-replica VRAM: ~9.5GB of weights plus activations and KV cache at
# the 8192-token cap, plus a few hundred MB of CUDA context. Used only to
# pick a default worker count; --workers overrides it.
VRAM_PER_WORKER_MB = 9600
# Leave the GPU some room for the display server and allocator slack.
VRAM_HEADROOM_FRACTION = 0.92
MAX_AUTO_WORKERS = 4
# Don't respawn a crash-looping worker faster than this.
RESPAWN_BACKOFF_SECONDS = 10.0


def query_free_vram_mb() -> Optional[int]:
    """Free VRAM according to nvidia-smi.

    Deliberately not torch.cuda.mem_get_info: that would initialise CUDA in
    the parent process, which then holds a context (a few hundred MB) it has
    no use for, since all inference happens in the children.
    """
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    values = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    return max(values) if values else None


def resolve_worker_count(requested: str, device: str) -> int:
    if requested != 'auto':
        return max(1, int(requested))
    if device != 'cuda':
        # On CPU the replicas would contend for the same cores rather than
        # for an idle GPU, so more of them buys nothing.
        return 1
    free_mb = query_free_vram_mb()
    if free_mb is None:
        logging.warning("Could not read free VRAM from nvidia-smi; defaulting to 1 worker")
        return 1
    count = int(free_mb * VRAM_HEADROOM_FRACTION) // VRAM_PER_WORKER_MB
    count = max(1, min(MAX_AUTO_WORKERS, count))
    logging.info(
        f"Auto-sizing pool: {free_mb}MB free VRAM / ~{VRAM_PER_WORKER_MB}MB per replica "
        f"-> {count} worker(s)")
    return count


class WorkerPool:
    """Spawns worker processes and restarts them if they die."""

    def __init__(self, count: int, backend_addr: str, args, torch_threads: int,
                 memory_fraction: Optional[float]):
        self.count = count
        self.backend_addr = backend_addr
        self.args = args
        self.torch_threads = torch_threads
        self.memory_fraction = memory_fraction
        self.procs: Dict[int, mp.Process] = {}
        self.last_spawn: Dict[int, float] = {}

    @staticmethod
    def identity(index: int) -> bytes:
        return f'worker-{index}'.encode()

    def start(self, index: int):
        proc = mp.Process(
            target=worker_main,
            args=(index, self.backend_addr, self.args.model_path, self.args.device,
                  self.args.log_level, self.torch_threads, self.memory_fraction,
                  self.args.no_repeat_ngram_size, self.args.streaming_path),
            name=f'ocr-worker-{index}',
            daemon=True,
        )
        proc.start()
        self.procs[index] = proc
        self.last_spawn[index] = time.monotonic()

    def start_all(self):
        for index in range(self.count):
            self.start(index)

    def dead_indices(self):
        """Indices whose process has exited."""
        for index, proc in list(self.procs.items()):
            if not proc.is_alive():
                yield index

    def may_restart(self, index: int) -> bool:
        """Whether enough time has passed to retry a worker that died.

        Kept separate from detecting the death: a client waiting on a dead
        worker should be told immediately, even when the replica itself is
        being held back because it is crash-looping.
        """
        return time.monotonic() - self.last_spawn.get(index, 0.0) >= RESPAWN_BACKOFF_SECONDS

    def shutdown(self):
        for proc in self.procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self.procs.values():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()


def run_broker(frontend_addr: str, backend_addr: str, pool: WorkerPool):
    """Least-recently-used broker: dispatch only to workers that are idle.

    Frame layouts, all of which the ROUTER sockets prepend/strip for us:

        client REQ  -> frontend:  [client_id, b'', request]
        broker      -> backend:   [worker_id, client_id, b'', request]
        worker      -> backend:   [worker_id, READY]
                              or  [worker_id, client_id, b'', reply]
        broker      -> frontend:  [client_id, b'', reply]

    Everything between the client id and the final payload frame is an
    opaque envelope that is echoed back untouched.

    A worker announces itself once with READY and thereafter its reply
    doubles as the signal that it is free again.
    """
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.ROUTER)
    frontend.bind(frontend_addr)
    backend.bind(backend_addr)

    pool.start_all()

    idle = deque()
    # worker identity -> the client waiting on it, so a crash mid-request can
    # be answered with an error instead of hanging the client until timeout.
    inflight: Dict[bytes, bytes] = {}
    # Identities seen at least once, so a restarted worker reporting READY
    # again is logged as a recovery rather than counted as a new arrival.
    seen = set()
    # Deaths already reported, so a worker held back by the respawn backoff
    # is not re-mourned on every poll tick.
    mourned = set()

    poller = zmq.Poller()
    poller.register(backend, zmq.POLLIN)
    frontend_registered = False

    logging.info(f"DeepSeek-OCR server listening on {frontend_addr}")
    logging.info(f"Starting {pool.count} worker(s); requests are queued until one is ready")

    try:
        while True:
            # Only take work off the wire when someone can do it. Requests
            # that arrive meanwhile sit in the ROUTER's queue, which is
            # exactly the backlog behaviour we want.
            if idle and not frontend_registered:
                poller.register(frontend, zmq.POLLIN)
                frontend_registered = True
            elif not idle and frontend_registered:
                poller.unregister(frontend)
                frontend_registered = False

            events = dict(poller.poll(timeout=1000))

            if backend in events:
                frames = backend.recv_multipart()
                worker_id = frames[0]
                idle.append(worker_id)
                inflight.pop(worker_id, None)
                if frames[1] == READY:
                    if worker_id in seen:
                        logging.info(f"{worker_id.decode()} back online")
                    else:
                        seen.add(worker_id)
                        logging.info(
                            f"{worker_id.decode()} ready ({len(seen)}/{pool.count} online)")
                        if len(seen) == pool.count:
                            logging.info("All workers online. Ready to process images...")
                else:
                    frontend.send_multipart(frames[1:])

            if frontend in events:
                # A plain REQ client sends [client_id, b'', request], but the
                # envelope is not ours to interpret - REQ_CORRELATE adds a
                # frame, and a DEALER client may send none. Keep the client id
                # for routing, forward the rest verbatim, and let the worker
                # reply with the same envelope. Unpacking a fixed frame count
                # here would let one oddly-framed client kill the broker and
                # with it every in-flight request.
                frames = frontend.recv_multipart()
                client_id, envelope = frames[0], frames[1:]
                worker_id = idle.popleft()
                inflight[worker_id] = client_id
                backend.send_multipart([worker_id, client_id] + envelope)

            for index in pool.dead_indices():
                worker_id = pool.identity(index)
                if index not in mourned:
                    mourned.add(index)
                    logging.error(f"{worker_id.decode()} died")
                    # Answer whoever was waiting on it rather than leaving
                    # them to discover it via their own timeout.
                    waiting = inflight.pop(worker_id, None)
                    if waiting is not None:
                        frontend.send_multipart([waiting, b'', json_error(
                            'worker process died while handling this request')])
                    try:
                        idle.remove(worker_id)
                    except ValueError:
                        pass
                if pool.may_restart(index):
                    logging.info(f"restarting {worker_id.decode()}")
                    mourned.discard(index)
                    pool.start(index)

    except KeyboardInterrupt:
        logging.info("Server shutting down...")
    finally:
        frontend.close(linger=0)
        backend.close(linger=0)
        context.term()


def json_error(message: str) -> bytes:
    return json.dumps({'error': message, 'text': '', 'layout': None}).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Remote Inference Server')
    parser.add_argument('--host', default='*', help='Host to bind to (* for all interfaces)')
    parser.add_argument('--port', type=int, default=5555, help='Port to bind to')
    parser.add_argument('--model-path', help='Path to DeepSeek-OCR model')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--workers', default='auto',
                       help='Number of model replicas, or "auto" to size from free VRAM '
                            '(default: auto)')
    parser.add_argument('--gpu-memory-fraction', type=float, default=None,
                       help='Cap each worker at this fraction of total VRAM. Off by default; '
                            'set it to stop one replica from starving the others, but leave '
                            'room for gundam-mode peaks')
    parser.add_argument('--torch-threads', type=int, default=None,
                       help='Intra-op CPU threads per worker (default: cores / workers)')
    parser.add_argument('--no-repeat-ngram-size', type=int, default=20,
                       help='N-gram repeat guard. 20 matches the upstream default '
                            '(default: 20)')
    parser.add_argument('--streaming-path', action='store_true',
                       help='Use the original stdout-scraping inference path instead of '
                            'the faster eval path. For A/B comparison only')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - broker - %(levelname)s - %(message)s'
    )

    # Must be set before any child touches CUDA. This workload allocates
    # wildly different shapes from request to request (crop counts and
    # generation lengths both vary), which is the classic way to fragment the
    # caching allocator into reserved-but-unusable blocks. Expandable
    # segments let the allocator grow a segment in place instead of stranding
    # it, which matters far more now that several replicas compete for one
    # card's memory.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # fork() copies a parent that may already hold a CUDA context and leaves
    # the child with unusable handles. Spawn gives each worker a clean
    # interpreter.
    mp.set_start_method('spawn', force=True)

    workers = resolve_worker_count(args.workers, args.device)
    torch_threads = args.torch_threads or max(1, (os.cpu_count() or 1) // workers)

    frontend_addr = f"tcp://{args.host}:{args.port}"
    # A unix socket for the internal hop: no port to collide with, no TCP
    # stack in the path for what are multi-megabyte base64 payloads.
    backend_dir = tempfile.mkdtemp(prefix='deepseek-ocr-')
    backend_addr = f"ipc://{backend_dir}/backend.ipc"

    pool = WorkerPool(workers, backend_addr, args, torch_threads, args.gpu_memory_fraction)
    try:
        run_broker(frontend_addr, backend_addr, pool)
    finally:
        pool.shutdown()
        shutil.rmtree(backend_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
