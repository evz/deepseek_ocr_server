#!/usr/bin/env python3
"""
Download DeepSeek-OCR model from Hugging Face

Run this script to pre-download the model before starting the server.
This avoids the download happening during the first inference request.
"""
import argparse
import logging

from transformers import AutoModel, AutoTokenizer


def download_model(model_name: str = "deepseek-ai/DeepSeek-OCR"):
    """
    Download DeepSeek-OCR model and tokenizer from Hugging Face

    Args:
        model_name: Model name or path on Hugging Face Hub
    """
    logging.info(f"Downloading model: {model_name}")
    logging.info("This may take a while (~5GB download)...")

    # Download tokenizer
    logging.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    logging.info(f"Tokenizer downloaded: {len(tokenizer)} tokens")

    # Download model
    logging.info("Downloading model weights...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True
    )
    logging.info("Model downloaded successfully!")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    logging.info("\n✓ Download complete!")
    logging.info(f"Model cached at: ~/.cache/huggingface/hub/")
    logging.info("\nYou can now start the server with:")
    logging.info("  python server.py --host 0.0.0.0 --port 5555 --device cuda")


def main():
    parser = argparse.ArgumentParser(description='Download DeepSeek-OCR model')
    parser.add_argument(
        '--model-name',
        default='deepseek-ai/DeepSeek-OCR',
        help='Model name on Hugging Face Hub'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        download_model(args.model_name)
    except KeyboardInterrupt:
        logging.warning("\nDownload interrupted by user")
    except Exception as e:
        logging.error(f"Error downloading model: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
