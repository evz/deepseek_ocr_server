# DeepSeek-OCR Server

A lightweight ZeroMQ-based server for running [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) inference on a GPU machine and accessing it remotely.

## Why This Exists

DeepSeek-OCR is a powerful vision-language model for OCR that produces excellent results, including:
- Markdown formatting (bold, italics, headers)
- Table structure recognition
- Layout-aware text extraction
- Grounding tokens with bounding boxes

However, it requires a GPU to run efficiently. This server allows you to:
- Run the model on a desktop/server with a GPU (e.g., RTX 5090)
- Access it from laptops, CI/CD pipelines, or other machines via network
- Use a simple request/response protocol (ZeroMQ + JSON)

## Quick Start

### 1. Install Dependencies

```bash
# On the GPU server
cd deepseek_ocr
./install.sh

# Or manually:
pip install -r requirements.txt
```

### 2. Download the Model

```bash
python download_model.py
```

This downloads the DeepSeek-OCR model from HuggingFace (~7GB).

### 3. Start the Server

```bash
python server.py --host 0.0.0.0 --port 5555
```

The server will:
- Load the model into GPU memory (~9.5GB VRAM)
- Start listening on port 5555
- Process incoming OCR requests

## Client Usage

See [CLIENT_EXAMPLE.md](CLIENT_EXAMPLE.md) for detailed examples of how to send requests to the server.

### Quick Example

```python
import zmq
import json
import base64
from PIL import Image
import io

# Connect to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Load and encode image
image = Image.open("document.jpg")
buffer = io.BytesIO()
image.save(buffer, format='PNG')
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Send request
request = {
    'image': image_b64,
    'mode': 'base',  # or 'tiny', 'small', 'large', 'gundam'
    'preserve_layout': True
}
socket.send_string(json.dumps(request))

# Get response
response = json.loads(socket.recv_string())
print(response['text'])
```

## Resolution Modes

The server supports multiple resolution modes to balance speed vs accuracy:

| Mode | Resolution | Vision Tokens | Use Case |
|------|-----------|---------------|----------|
| `tiny` | 512×512 | 64 | Fast previews, low-quality scans |
| `small` | 640×640 | 100 | General purpose, good balance |
| `base` | 1024×1024 | 256 | High quality documents (default) |
| `large` | 1280×1280 | 400 | Very detailed documents |
| `gundam` | 1024×1024 tiles + 1280×1280 global | ~1800 | Maximum quality, dynamic tiling |

**Recommendation**: Start with `base` mode. Use `gundam` for high-DPI scans (600+ DPI) with small text.

## API Reference

### Request Format

```json
{
  "image": "<base64-encoded PNG/JPEG>",
  "mode": "base",
  "preserve_layout": true
}
```

**Parameters:**
- `image` (required): Base64-encoded image data
- `mode` (optional): Resolution mode (default: `"base"`)
- `preserve_layout` (optional): Enable layout detection with grounding tokens (default: `true`)

### Response Format

```json
{
  "text": "Extracted text with markdown formatting...",
  "layout": null,
  "metadata": {
    "processing_time_ms": 1234,
    "image_size": [2480, 3508],
    "mode": "base",
    "device": "cuda",
    "base_size": 1024,
    "image_size": 1024,
    "crop_mode": false
  }
}
```

**Response Fields:**
- `text`: Extracted text with markdown formatting and grounding tokens
- `layout`: Reserved for future use (currently `null`)
- `metadata`: Processing details and configuration

### Grounding Tokens

When `preserve_layout=True`, the output includes special tokens:

```
<|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
This is a paragraph of text.

<|ref|>table<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
<table>
  <tr><td>Cell 1</td><td>Cell 2</td></tr>
</table>

<|ref|>sub_title<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
## Section Header
```

These tokens provide:
- Element type (`text`, `table`, `sub_title`, `title`, `image`, `image_caption`)
- Bounding box coordinates `[[x1, y1, x2, y2]]`
- Clean HTML markup for tables

## Server Options

```bash
python server.py [OPTIONS]

Options:
  --host HOST        Host to bind to (default: '*' = all interfaces)
  --port PORT        Port to bind to (default: 5555)
  --model-path PATH  Path to model (default: deepseek-ai/DeepSeek-OCR from HF cache)
  --device DEVICE    Device to use: 'cuda' or 'cpu' (default: 'cuda')
  --log-level LEVEL  Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Performance

**GPU**: RTX 5090 (32GB VRAM)
**Model Size**: ~8GB VRAM

| Mode | Avg Time (600 DPI A4) | Tokens |
|------|---------------------|--------|
| `small` | ~2 seconds | 100 |
| `base` | ~18 seconds | 256 |
| `gundam` | ~38 seconds | ~1800 |

**Network overhead**: ~1-5ms for local network requests

## Troubleshooting

### Server won't start

**Issue**: `Model not found in cache`
**Solution**: Run `python download_model.py` first

**Issue**: `CUDA out of memory`
**Solution**: Close other GPU applications or use `--device cpu` (slow)

### Client timeout

**Issue**: `Server did not respond within 30000ms`
**Solution**: Increase timeout in client code:
```python
socket.setsockopt(zmq.RCVTIMEO, 300000)  # 5 minutes
```

Gundam mode can take 1-2 minutes per page on high-resolution scans.

### Poor OCR quality

**Issue**: Text is garbled or incomplete
**Solution**:
1. Try a higher resolution mode (`base` or `gundam`)
2. Ensure images are high quality (600+ DPI for archival documents)
3. Check that images aren't too compressed (use PNG or high-quality JPEG)

## Architecture

```
Client Machine                Server Machine (GPU)
┌──────────────┐             ┌───────────────────┐
│              │   ZeroMQ    │                   │
│  Your Code   │──REQ/REP───▶│  DeepSeek-OCR     │
│              │   TCP:5555  │  Server           │
└──────────────┘             └───────────────────┘
```

- **Protocol**: ZeroMQ REQ/REP (synchronous request/response)
- **Encoding**: JSON for metadata, Base64 for images
- **Port**: 5555 (configurable)

## Credits

- DeepSeek-OCR Model: [DeepSeek AI](https://github.com/deepseek-ai/DeepSeek-OCR)
- Server implementation: Lightweight ZMQ wrapper

## License

MIT License - see model repository for model licensing
