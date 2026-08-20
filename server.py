#!/usr/bin/env python3
"""
DeepSeek-OCR Remote Inference Server

Runs on desktop with RTX 5090, accepts image processing requests via ZMQ.
"""
import argparse
import base64
import io
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import torch
import zmq
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class DeepSeekOCRServer:
    """Server that loads DeepSeek-OCR model and handles inference requests"""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        # Reclaim cached blocks once unused-but-reserved VRAM exceeds this.
        self.cache_release_threshold_bytes = 4 * 1024 ** 3

        logging.info(f"Initializing DeepSeek-OCR server on {device}")
        self._load_model()

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

            self.model = AutoModel.from_pretrained(
                model_name,
                _attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                device_map='auto' if self.device == 'cuda' else None,
                trust_remote_code=True,
                use_safetensors=True,
                local_files_only=True  # Don't download, use cached only
            )

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
        """Let a request bound its own generation length.

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
        """
        self._generation_overrides = {}
        original_generate = self.model.generate

        def bounded_generate(*args, **kwargs):
            overrides = self._generation_overrides
            cap = overrides.get('max_new_tokens')
            if cap is not None:
                kwargs['max_new_tokens'] = min(int(cap), kwargs.get('max_new_tokens', 8192))
            penalty = overrides.get('repetition_penalty')
            if penalty is not None:
                kwargs['repetition_penalty'] = float(penalty)
            return original_generate(*args, **kwargs)

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
                # Capture stdout since model.infer() prints to stdout instead of returning
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

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
                        self.model.infer(
                            self.tokenizer,
                            prompt=prompt,
                            image_file=tmp_path,
                            output_path=output_dir,
                            base_size=config['base_size'],
                            image_size=config['image_size'],
                            crop_mode=config['crop_mode'],
                            save_results=False,
                            test_compress=False
                        )

                    # Get the captured output and filter debug lines
                    raw_output = sys.stdout.getvalue()
                    text = self._filter_debug_output(raw_output)
                finally:
                    sys.stdout = old_stdout

            finally:
                # Clean up temp output directory
                shutil.rmtree(output_dir, ignore_errors=True)

            # Build result
            result = {
                'text': text,
                'layout': None,  # DeepSeek-OCR doesn't return separate layout
                'metadata': {
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'image_size': Image.open(tmp_path).size,
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

def main():
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Remote Inference Server')
    parser.add_argument('--host', default='*', help='Host to bind to (* for all interfaces)')
    parser.add_argument('--port', type=int, default=5555, help='Port to bind to')
    parser.add_argument('--model-path', help='Path to DeepSeek-OCR model')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize server
    server = DeepSeekOCRServer(model_path=args.model_path, device=args.device)

    # Setup ZMQ socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    bind_address = f"tcp://{args.host}:{args.port}"
    socket.bind(bind_address)

    logging.info(f"DeepSeek-OCR server listening on {bind_address}")
    logging.info("Ready to process images...")

    # Main processing loop
    request_count = 0
    try:
        while True:
            # Wait for request
            message = socket.recv()
            request_count += 1

            try:
                # Parse request
                request = json.loads(message.decode('utf-8'))

                # Decode image from base64
                image_data = base64.b64decode(request['image'])
                params = {k: v for k, v in request.items() if k != 'image'}

                logging.info(f"Processing request #{request_count}, mode={params.get('mode', 'base')}")

                # Process image
                result = server.process_image(image_data, params)

                # Send response
                response = json.dumps(result)
                socket.send_string(response)

                logging.info(f"Request #{request_count} completed in {result['metadata']['processing_time_ms']}ms")

            except Exception as e:
                logging.error(f"Error processing request: {e}", exc_info=True)
                error_response = json.dumps({
                    'error': str(e),
                    'text': '',
                    'layout': None
                })
                socket.send_string(error_response)

    except KeyboardInterrupt:
        logging.info("Server shutting down...")
    finally:
        socket.close()
        context.term()


if __name__ == '__main__':
    main()
