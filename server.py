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

        logging.info(f"Initializing DeepSeek-OCR server on {device}")
        self._load_model()

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
            # comes back None. That matters because the model's own
            # infer() passes it straight through as
            # `generate(eos_token_id=tokenizer.eos_token_id, ...)` - a None
            # there leaves generation with no stopping criterion, so every
            # request runs to the hardcoded max_new_tokens=8192 instead of
            # stopping when the page is transcribed. Combined with infer()'s
            # `no_repeat_ngram_size=20` (which forbids repeating any
            # 20-token span, so the model cannot idle by repeating itself)
            # the tail past the real content comes back as *novel*
            # hallucinated filler: incrementing number runs, endless LaTeX
            # `\begin{array}{cccc...}` padding, or restated garbage lines.
            # Setting the tokens restores early stopping.
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

        logging.info("Model loaded successfully")

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
                    'crop_mode': config['crop_mode']
                }
            }

            return result

        finally:
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
