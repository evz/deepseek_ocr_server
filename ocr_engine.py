#!/usr/bin/env python3
"""
DeepSeek-OCR inference engine and worker process.

One OCREngine == one model replica == one worker process. The server
(server.py) runs several of these behind a load-balancing broker; nothing in
here knows about that, so this module stays usable standalone.
"""
import base64
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import zmq
from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
)

# Frame the broker uses to mean "this worker is idle". Not a valid ZMQ
# identity for a client, so it can never be confused with a routing frame.
READY = b'\x01READY'


class IncrementalNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """A no-repeat-ngram ban that costs O(1) per decode step instead of O(n).

    This replaces transformers' NoRepeatNGramLogitsProcessor, which is the
    single largest consumer of CPU time in this workload. The stock version
    calls `_get_ngrams()` on every step, and that function rebuilds the
    *entire* ngram dictionary from scratch: it pulls the whole sequence to
    the host with `.tolist()` (a device sync, every token), slices it
    `ngram_size` times, then builds one tuple, one list and one dict entry
    per position. At ngram_size=20 that measured 0.44ms per step at a
    500-token sequence, 2.7ms at 2000, and 12.9ms at 8000 - and since the
    sequence grows every step, the cost over a whole request is quadratic.
    Integrated, that is seconds of pure Python per request (~3s for a base
    -mode page, ~8s for gundam, ~54s for a full 8192-token runaway), all of
    it serial, on the critical path, with the GPU parked waiting for it.

    The observation that makes it cheap: between one step and the next, the
    sequence grows by exactly one token, so the dictionary gains exactly one
    ngram. There is no reason to rebuild it. Keeping the bank across steps
    and folding in that single new entry produces a dictionary identical to
    the rebuilt one - verified step-by-step against the stock implementation
    across degenerate, counting-loop and realistic-text sequences.

    Two smaller savings ride along: the newly generated token is fetched as
    a single scalar rather than by copying the whole sequence, and `scores`
    is only cloned on steps that actually ban something (the stock version
    clones a 129k-wide vocab tensor every step regardless).

    Only the batch-1, non-beam case is handled incrementally, which is what
    infer() uses; anything else falls back to the stock processor.
    """

    def __init__(self, ngram_size: int):
        self.ngram_size = ngram_size
        self._fallback: Optional[NoRepeatNGramLogitsProcessor] = None
        self.reset()

    def reset(self):
        """Drop state between generate() calls."""
        self._seq = []
        self._bank = {}

    def _seed(self, input_ids):
        """Build the bank from a full sequence (first step, or after a reset)."""
        n = self.ngram_size
        self._seq = input_ids[0].tolist()
        self._bank = {}
        for i in range(n - 1, len(self._seq)):
            self._bank.setdefault(tuple(self._seq[i - n + 1:i]), []).append(self._seq[i])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        n = self.ngram_size
        cur_len = input_ids.shape[-1]

        if input_ids.shape[0] != 1:
            if self._fallback is None:
                self._fallback = NoRepeatNGramLogitsProcessor(n)
            return self._fallback(input_ids, scores)

        # The incremental bank is only valid if this call continues the
        # sequence the previous call saw. A shorter sequence, or a different
        # token where we last appended one, means a new generation - rebuild.
        stale = (
            not self._seq
            or cur_len < len(self._seq)
            or int(input_ids[0, len(self._seq) - 1]) != self._seq[-1]
        )
        if stale:
            self._seed(input_ids)
        else:
            while len(self._seq) < cur_len:
                token = int(input_ids[0, len(self._seq)])
                self._seq.append(token)
                if len(self._seq) >= n:
                    self._bank.setdefault(tuple(self._seq[-n:-1]), []).append(token)

        if cur_len + 1 < n:
            return scores

        banned = self._bank.get(tuple(self._seq[cur_len + 1 - n:cur_len]))
        if not banned:
            return scores

        scores = scores.clone()
        scores[0, banned] = -float('inf')
        return scores


class OCREngine:
    """Loads DeepSeek-OCR and runs inference requests against it."""

    def __init__(self, model_path: str = None, device: str = "cuda",
                 no_repeat_ngram_size: int = 20, use_streaming_path: bool = False):
        self.model_path = model_path
        self.device = device
        self.model = None
        # infer() hardcodes 20 on the streaming path and 35 on the eval path.
        # Pinning it here keeps output identical no matter which path runs.
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.use_streaming_path = use_streaming_path
        # Reclaim cached blocks once unused-but-reserved VRAM exceeds this.
        self.cache_release_threshold_bytes = 4 * 1024 ** 3

        logging.info(f"Initializing DeepSeek-OCR engine on {device}")
        self._configure_backend()
        self._load_model()

    def _configure_backend(self):
        """Backend switches that cost nothing and are always right for serving."""
        if self.device != 'cuda' or not torch.cuda.is_available():
            return
        # The vision tower (SAM ViT-B + CLIP-L) is convolution-heavy and sees
        # the same input shapes over and over, since each mode pins its own
        # base_size/image_size. That is precisely the case cudnn's autotuner
        # is for: it pays a one-off search on the first request of each shape
        # and reuses the winning algorithm forever after.
        torch.backends.cudnn.benchmark = True
        # Matmuls run under bf16 autocast, so TF32 only affects the fp32 ops
        # left over around them. Free accuracy-for-speed on those.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _vram_stats(self) -> Dict[str, Any]:
        """Live VRAM figures, so a caller can tell a real leak from the
        allocator merely caching.

        These are different problems with different fixes, and nvidia-smi
        cannot distinguish them - it reports reserved memory, which grows
        to a high-water mark by design and never shrinks on its own.
        `allocated` is memory held by live tensors: if that climbs across
        requests, something is genuinely being retained. If `allocated`
        stays flat while `reserved` grows, it is fragmentation from this
        workload's constantly-varying crop sizes and generation lengths -
        expected, bounded, and reclaimable with empty_cache().
        """
        if self.device != 'cuda' or not torch.cuda.is_available():
            return {}
        return {
            'vram_allocated_mb': round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
            'vram_reserved_mb': round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
            'vram_max_allocated_mb': round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
        }

    def _release_cache_if_needed(self):
        """Hands cached-but-unused blocks back when fragmentation overhead
        grows large.

        Only the gap between reserved and allocated is reclaimable, so this
        triggers on that gap rather than on a request count - a run of
        similarly-sized crops needs no releasing at all, while a run of
        wildly differing ones does. empty_cache() forces a synchronise, so
        it is not something to do on every request.

        With several replicas sharing one GPU this matters more than it did
        with one: cached-but-unused blocks in worker 0 are blocks worker 2
        cannot allocate. Note that PYTORCH_CUDA_ALLOC_CONF=expandable_segments
        (set by server.py) makes the underlying fragmentation much rarer, so
        this should now fire seldom.
        """
        if self.device != 'cuda' or not torch.cuda.is_available():
            return
        overhead = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if overhead > self.cache_release_threshold_bytes:
            torch.cuda.empty_cache()
            logging.info(f"Released {overhead / 1024 ** 3:.1f} GiB of cached VRAM")

    def _filter_debug_output(self, raw_output: str) -> str:
        """
        Filter out debug lines from model output

        Removes lines like:
        - "directly resize"
        - "====================="
        - "BASE:  torch.Size([1, 100, 1280])"
        - "PATCHES:  torch.Size([6, 100, 1280])"
        - "NO PATCHES"

        On the eval path the text arrives as infer()'s return value rather
        than scraped from stdout, so this finds nothing - it is kept as a
        safety net and for the streaming path.
        """
        lines = raw_output.split('\n')
        filtered_lines = []

        for line in lines:
            line_stripped = line.strip()
            # Skip debug lines
            if (line_stripped in ('directly resize', 'NO PATCHES', '=====================') or
                line_stripped.startswith('BASE:') or
                line_stripped.startswith('PATCHES:') or
                line_stripped.startswith('torch.Size')):
                continue
            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _load_model(self):
        """Load DeepSeek-OCR model"""
        model_name = self.model_path or 'deepseek-ai/DeepSeek-OCR'

        logging.info(f"Loading model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True  # Don't download, use cached only
            )

            # DeepSeek-OCR's tokenizer_config.json declares these two only
            # in `added_tokens_decoder` and never sets the top-level
            # `eos_token`/`pad_token` fields, so `tokenizer.eos_token_id`
            # comes back None and infer() forwards that None into
            # `generate(eos_token_id=...)`, which is what emits the
            # "pad token id ... :None" warning at inference time.
            #
            # Setting them is correct hygiene, but measured against real
            # runaway crops it changed nothing: byte-identical output
            # before and after. The reason is that a stopping criterion
            # only helps if the model actually emits EOS, and in the
            # runaway case it never does - it hits the hardcoded
            # max_new_tokens=8192 instead (a crop that degenerated into
            # counting stopped at exactly ~8192 tokens' worth of numbers).
            # See the generate() wrapper below for what does bound it.
            if self.tokenizer.eos_token_id is None:
                self.tokenizer.eos_token = '<｜end▁of▁sentence｜>'
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = '<｜▁pad▁｜>'
            logging.info(
                f"Tokenizer special tokens: eos_token_id={self.tokenizer.eos_token_id}, "
                f"pad_token_id={self.tokenizer.pad_token_id}"
            )

            # device_map='auto' routes the load through accelerate, which
            # wraps every submodule in dispatch hooks that re-check device
            # placement on each forward. That is the price of admission for
            # a model too big for one GPU; this one is ~6.7GB of bf16 weights
            # and fits with room to spare, so the hooks buy nothing and are
            # pure per-forward Python overhead. Load to CPU and move it.
            self.model = AutoModel.from_pretrained(
                model_name,
                _attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_safetensors=True,
                local_files_only=True  # Don't download, use cached only
            )
            if self.device == 'cuda':
                self.model = self.model.to('cuda')

        except OSError as e:
            logging.error(f"Model not found in cache: {model_name}")
            logging.error("Please download the model first using:")
            logging.error("  python download_model.py")
            raise SystemExit(1)

        self.model = self.model.eval()

        # infer() never passes pad_token_id, so generate() falls back to
        # the generation config; seeding it here keeps that fallback from
        # resolving to None as well.
        if getattr(self.model, 'generation_config', None) is not None:
            if self.model.generation_config.eos_token_id is None:
                self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            if self.model.generation_config.pad_token_id is None:
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self._install_generate_bounds()

        logging.info("Model loaded successfully")

    def _install_generate_bounds(self):
        """Let a request bound its own generation length, and swap in the
        cheap ngram processor.

        infer() hardcodes `max_new_tokens=8192` and accepts no override, so
        the only place to intervene is generate() itself. This matters
        because 8192 is not a safety net here - it is the *only* thing that
        stops a degenerate run. On a dense-table crop the model can enter a
        loop that emits novel-but-meaningless tokens (an incrementing
        number run was measured going 1445 -> 4166), and because infer()
        also sets `no_repeat_ngram_size=20`, counting upward never trips
        the repeat check, so the loop sustains itself until the cap. The
        model never emits EOS in that state, so setting eos_token_id does
        not help - the run simply burns the full budget, taking ~150s and
        provoking transformers' "will exceed the model's predefined
        maximum length (8192)" warning.

        A caller that knows roughly how much text its crop contains (say a
        dozen table rows) can pass a much smaller `max_new_tokens`, which
        converts a 150-second runaway into a fast, cheap, obviously-
        truncated response the caller can detect and retry. Requests that
        say nothing keep the original 8192 behaviour.

        The same wrapper is where the ngram processor gets replaced. Setting
        `no_repeat_ngram_size=0` in kwargs stops generate() from building the
        stock O(n)-per-step processor, and the incremental one is appended to
        `logits_processor` in its place. Because the size is taken from this
        engine rather than from the caller, both of infer()'s paths (20 on
        the streaming branch, 35 on the eval branch) end up applying the same
        constraint - so switching paths cannot change what the model emits.
        """
        self._generation_overrides = {}
        self._ngram_processor = IncrementalNoRepeatNGramLogitsProcessor(
            self.no_repeat_ngram_size)
        original_generate = self.model.generate

        def bounded_generate(*args, **kwargs):
            overrides = self._generation_overrides
            cap = overrides.get('max_new_tokens')
            if cap is not None:
                kwargs['max_new_tokens'] = min(int(cap), kwargs.get('max_new_tokens', 8192))
            penalty = overrides.get('repetition_penalty')
            if penalty is not None:
                kwargs['repetition_penalty'] = float(penalty)

            kwargs['no_repeat_ngram_size'] = 0
            self._ngram_processor.ngram_size = self.no_repeat_ngram_size
            self._ngram_processor.reset()
            processors = kwargs.get('logits_processor') or LogitsProcessorList()
            processors.append(self._ngram_processor)
            kwargs['logits_processor'] = processors

            try:
                return original_generate(*args, **kwargs)
            finally:
                # The bank pins one Python int per token of the whole
                # sequence plus a tuple per ngram; on an 8192-token runaway
                # that is tens of megabytes with no reason to outlive the
                # request.
                self._ngram_processor.reset()

        self.model.generate = bounded_generate

    def process_image(self, image_data: bytes, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image with DeepSeek-OCR

        Args:
            image_data: Raw image bytes
            params: Processing parameters (mode, language, etc.)

        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()

        # infer() takes a path, so the bytes have to land on disk. Read the
        # dimensions straight from the buffer while it is already in memory -
        # PIL parses only the header for .size, so this costs a few
        # microseconds and saves reopening the file after inference.
        image_size = Image.open(io.BytesIO(image_data)).size

        # Save image to temp file (model expects file path)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        try:
            # Extract parameters
            mode = params.get('mode', 'base')
            preserve_layout = params.get('preserve_layout', True)

            # Per-request generation bounds, applied by the generate()
            # wrapper installed at load time. Absent keys leave the
            # model's own hardcoded defaults alone.
            self._generation_overrides = {
                key: params[key]
                for key in ('max_new_tokens', 'repetition_penalty')
                if params.get(key) is not None
            }

            # Map mode to DeepSeek-OCR parameters
            # Tiny: base_size = 512, image_size = 512, crop_mode = False
            # Small: base_size = 640, image_size = 640, crop_mode = False
            # Base: base_size = 1024, image_size = 1024, crop_mode = False
            # Large: base_size = 1280, image_size = 1280, crop_mode = False
            # Gundam: base_size = 1024, image_size = 640, crop_mode = True
            mode_config = {
                'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False},
                'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False},
                'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False},
                'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False},
                'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True}
            }
            config = mode_config.get(mode, mode_config['base'])

            # Choose prompt based on preserve_layout
            if preserve_layout:
                prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            else:
                prompt = "<image>\nFree OCR. "

            # Run inference
            # Create temp output directory (model requires it even if save_results=False)
            output_dir = tempfile.mkdtemp()

            try:
                # model.eval() does not disable gradient tracking - it
                # only switches dropout/batchnorm to eval behaviour. HF's
                # generate() carries its own no_grad, but infer() also
                # runs the vision encoder before generating, and nothing
                # here guaranteed that forward pass wasn't building an
                # autograd graph and holding activations alive. This is
                # a serving process that never calls backward, so
                # inference_mode is correct unconditionally and costs
                # nothing.
                with torch.inference_mode():
                    text = self._infer(prompt, tmp_path, output_dir, config)
            finally:
                # Clean up temp output directory
                shutil.rmtree(output_dir, ignore_errors=True)

            # Build result
            result = {
                'text': text,
                'layout': None,  # DeepSeek-OCR doesn't return separate layout
                'metadata': {
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'image_size': image_size,
                    'mode': mode,
                    'device': self.device,
                    'base_size': config['base_size'],
                    'image_size': config['image_size'],
                    'crop_mode': config['crop_mode'],
                    **self._vram_stats(),
                    # Echoed so a caller can tell an applied bound from an
                    # ignored one - the previous protocol silently dropped
                    # unknown keys, which made that indistinguishable.
                    'generation_overrides': dict(self._generation_overrides)
                }
            }

            return result

        finally:
            self._generation_overrides = {}
            self._release_cache_if_needed()
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def _infer(self, prompt: str, image_path: str, output_dir: str,
               config: Dict[str, Any]) -> str:
        """Call infer() and get the text back.

        infer() has two branches. The default one attaches a
        NoEOSTextStreamer, which calls tokenizer.decode() and print() once
        per generated token and returns nothing - the text has to be
        recovered by swapping out the process-global sys.stdout. That is
        both per-token CPU work on the critical path and a global mutation,
        the second of which is what makes two concurrent inferences in one
        process impossible.

        `eval_mode=True` runs the same generation with no streamer and
        returns the decoded string directly. It reads `no_repeat_ngram_size
        =35` instead of 20, but the generate() wrapper overrides that on both
        branches, so the two paths produce the same tokens. Preprocessing
        still prints a stray 'directly resize' and the encoder prints its
        tensor shapes, so stdout is still swallowed - but only to discard it,
        not to parse it.
        """
        sink = io.StringIO()
        with redirect_stdout(sink):
            outputs = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=config['base_size'],
                image_size=config['image_size'],
                crop_mode=config['crop_mode'],
                save_results=False,
                test_compress=False,
                eval_mode=not self.use_streaming_path,
            )

        if self.use_streaming_path:
            return self._filter_debug_output(sink.getvalue())
        return self._filter_debug_output(outputs or '')


def worker_main(index: int, backend_addr: str, model_path: Optional[str], device: str,
                log_level: str, torch_threads: int, memory_fraction: Optional[float],
                no_repeat_ngram_size: int, use_streaming_path: bool):
    """Entry point for one worker process.

    Runs a model replica and serves one request at a time. Parallelism comes
    from running several of these: each is its own interpreter, so each gets
    its own GIL and its own CPU core, and one worker's decode loop can drive
    the GPU while another's is stuck in Python.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=f'%(asctime)s - worker{index} - %(levelname)s - %(message)s'
    )

    # Without this, every worker sizes its intra-op pool to the full core
    # count, and N workers on a 16-core box fight over N*16 threads. The
    # heavy lifting is on the GPU; these threads only serve preprocessing.
    torch.set_num_threads(max(1, torch_threads))

    if memory_fraction is not None and device == 'cuda' and torch.cuda.is_available():
        # Caps this worker's slice of the caching allocator so one replica's
        # runaway request cannot starve its siblings into OOM.
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

    engine = OCREngine(
        model_path=model_path,
        device=device,
        no_repeat_ngram_size=no_repeat_ngram_size,
        use_streaming_path=use_streaming_path,
    )

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, f'worker-{index}'.encode())
    socket.connect(backend_addr)
    socket.send_multipart([READY])
    logging.info("ready")

    request_count = 0
    try:
        while True:
            # [client_id, *envelope, payload]. The envelope belongs to the
            # client's socket type, so it is echoed back rather than parsed.
            frames = socket.recv_multipart()
            client_id, envelope, payload = frames[0], frames[1:-1], frames[-1]
            request_count += 1

            try:
                request = json.loads(payload.decode('utf-8'))
                image_data = base64.b64decode(request['image'])
                params = {k: v for k, v in request.items() if k != 'image'}

                logging.info(
                    f"Processing request #{request_count}, mode={params.get('mode', 'base')}")

                result = engine.process_image(image_data, params)
                response = json.dumps(result)

                # VRAM goes in the log line, not just the response metadata:
                # the thing being watched for is drift across many requests,
                # which is only visible as a series. `allocated` climbing
                # means tensors are being retained; `allocated` flat while
                # `reserved` climbs is the caching allocator reaching its
                # high-water mark, which is expected and plateaus.
                meta = result['metadata']
                vram = ""
                if 'vram_allocated_mb' in meta:
                    vram = (f", vram alloc={meta['vram_allocated_mb']:.0f}MB "
                            f"reserved={meta['vram_reserved_mb']:.0f}MB "
                            f"peak={meta['vram_max_allocated_mb']:.0f}MB")
                logging.info(
                    f"Request #{request_count} completed in {meta['processing_time_ms']}ms{vram}")

            except Exception as e:
                logging.error(f"Error processing request: {e}", exc_info=True)
                response = json.dumps({
                    'error': str(e),
                    'text': '',
                    'layout': None
                })

            # Sending the reply is also what tells the broker this worker is
            # free again, so it must happen even for a failed request.
            socket.send_multipart([client_id] + envelope + [response.encode('utf-8')])

    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)
        context.term()
