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
- Size a pool of model replicas from free VRAM (~9.5GB each)
- Start listening on port 5555
- Process incoming OCR requests, several at a time

Pin the pool size yourself with `--workers N` if you would rather not let it
guess.

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

## Concurrency

The server runs several model replicas, each in its own process, behind a
load-balancing broker. Two reasons it is built that way:

- **A single replica leaves the GPU idle.** A decode step is a few
  milliseconds of GPU work bracketed by a comparable amount of Python
  (kernel launches, logits processors, stopping criteria). One replica
  spends much of its time with the GPU waiting on one pinned CPU core.
- **Threads cannot fix that; processes can.** Python threads share a GIL, so
  a second thread would not get a second core. Separate interpreters do:
  while one worker is in Python, another's kernels have the GPU.

The broker hands each request to an *idle* worker rather than round-robin,
which matters because request durations here vary by more than an order of
magnitude — round-robin would park a `tiny` page behind a `gundam` one while
another worker sat doing nothing.

### Getting parallelism from the client

A single ZMQ `REQ` socket is lockstep by design: it will not send a second
request until the first has been answered. Pointing one `REQ` socket at the
server therefore still gives you one request at a time, no matter how many
workers are running. To actually use the pool, open several sockets — see
[CLIENT_EXAMPLE.md](CLIENT_EXAMPLE.md) for a worker-pool client.

### Choosing `--workers`

`--workers auto` (the default) divides free VRAM by ~9.6GB per replica and
caps the result at 4. That is an estimate, not a measurement, so confirm it
against real pages. A sample set ships with the repo:

```bash
unzip testdata/bevreg-5007-298-sample.zip -d testdata/
python bench.py --images testdata/bevreg-5007-298-sample \
    --concurrency 1,2,4,8 --repeat 3 --warmup
```

Throughput that stops rising with concurrency means the pool is saturated.
If it is still climbing at your highest concurrency, raise `--workers`; if
you see CUDA OOM under `gundam` load, lower it.

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
  --host HOST                  Host to bind to (default: '*' = all interfaces)
  --port PORT                  Port to bind to (default: 5555)
  --model-path PATH            Path to model (default: deepseek-ai/DeepSeek-OCR from HF cache)
  --device DEVICE              Device to use: 'cuda' or 'cpu' (default: 'cuda')
  --workers N                  Model replicas, or 'auto' to size from free VRAM (default: auto)
  --gpu-memory-fraction F      Cap each worker at this fraction of total VRAM. Off by
                               default; stops one replica starving the others, but leave
                               room for gundam-mode peaks
  --torch-threads N            Intra-op CPU threads per worker (default: cores / workers)
  --no-repeat-ngram-size N     N-gram repeat guard (default: 20, the upstream value)
  --streaming-path             Use the original stdout-scraping inference path.
                               For A/B comparison only
  --log-level LEVEL            Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Performance

**GPU**: RTX 5090 (32GB VRAM)
**Model Size**: ~9.5GB VRAM per replica

Single-request baseline, before the optimisations below:

| Mode | Avg Time (600 DPI A4) | Tokens |
|------|---------------------|--------|
| `small` | ~2 seconds | 100 |
| `base` | ~18 seconds | 256 |
| `gundam` | ~38 seconds | ~1800 |

**Network overhead**: ~1-5ms for local network requests

### Where the time was going

The server used to peg one CPU core while the GPU idled. Most of that was a
single function. `infer()` sets `no_repeat_ngram_size=20`, and transformers'
`NoRepeatNGramLogitsProcessor` rebuilds its **entire** n-gram dictionary from
scratch on every decode step: it copies the whole sequence to the host with
`.tolist()` (a device sync, every token), slices it 20 times, then builds a
tuple, a list and a dict entry per position.

The dictionary only ever gains **one** n-gram per step, so it is now
maintained incrementally instead of being rebuilt. Cost per decode step, both
processors measured on real tensors at the model's 129,280-token vocabulary:

| Sequence length | Stock | Incremental | Speedup |
|---|---|---|---|
| 500 | 0.54 ms | 0.19 ms | 3× |
| 1000 | 1.38 ms | 0.19 ms | 7× |
| 2000 | 2.71 ms | 0.27 ms | 10× |
| 4000 | 7.25 ms | 0.46 ms | 16× |
| 8000 | 18.01 ms | 1.00 ms | 18× |

Because the sequence grows every step, the stock cost over a whole request is
quadratic. Integrated across a full generation:

| Scenario | Stock | Incremental | Saved |
|---|---|---|---|
| `base` page (~2000 tokens) | 4.9 s | 0.1 s | 4.9 s |
| `gundam` (~1800 vision tokens) | 15.0 s | 0.1 s | 14.9 s |
| 8192-token runaway | 97.6 s | 0.3 s | 97.4 s |

(Measured on a laptop CPU, so absolute figures will be lower on the server —
the ratios are what carry over. All of this time was serial, on the critical
path, with the GPU idle waiting for it.)

This is an optimisation, not a behaviour change, and that is checked rather
than assumed — `test_ngram.py` steps whole generations forward token by
token and asserts the two processors ban identical tokens at every step,
across degenerate, counting-loop and realistic-text sequences:

```bash
python test_ngram.py
```

Alongside that:

- **No per-token streaming.** The default `infer()` path attaches a streamer
  that calls `tokenizer.decode()` and `print()` once per token and returns
  nothing, so the text had to be recovered by swapping the process-global
  `sys.stdout`. The server now uses `infer(eval_mode=True)`, which returns
  the string directly. That removes per-token CPU work *and* the global
  mutation that made two concurrent inferences in one process impossible.
- **No accelerate dispatch hooks.** `device_map='auto'` wraps every submodule
  in placement checks meant for models too big for one GPU. This one fits, so
  it loads to CPU and moves in one step.
- **cudnn autotuning.** Each mode pins its own input shapes, so the vision
  tower sees the same shapes repeatedly — the case `cudnn.benchmark` exists for.
- **`expandable_segments` allocator.** Crop counts and generation lengths vary
  wildly per request, which is the classic way to fragment the caching
  allocator. This matters more now that replicas compete for one card.
- **Bounded CPU threads per worker.** Otherwise every replica sizes its thread
  pool to the full core count and they fight over cores.

Measure the result on your own documents with `bench.py`; the numbers above
are the single-request baseline it should be compared against.

### Verifying output is unchanged

The inference path changed, so if you want proof the text did not:

```bash
python server.py --streaming-path &        # original path
python bench.py --images testdata/bevreg-5007-298-sample \
    --concurrency 1 --save-dir out-old
# restart the server without --streaming-path
python bench.py --images testdata/bevreg-5007-298-sample \
    --concurrency 1 --save-dir out-new
diff -r out-old out-new
```

## Test Data

`testdata/bevreg-5007-298-sample.zip` holds 30 scanned pages (22MB) for
benchmarking, sampled evenly across the five parts of Amsterdam
bevolkingsregister inventory 5007, access 298. They are the embedded JPEGs
copied out of the source PDFs verbatim — no re-encoding or resampling — so
the server sees exactly what the archive scanned.

The spread is deliberate. Decode time tracks how much *text* is on a page,
not how large the image is, so a benchmark set has to span the range: these
run from printed prose through to dense numeric register tables, the latter
being the pages that provoke the runaway generations `max_new_tokens` exists
to bound. Benchmarking on sparse pages alone will tell you the worker pool is
saturated when it is not.

`MANIFEST.txt` inside the zip records which PDF and page each image came
from, and `extract_sample.py` regenerates the set (or a larger one) from the
source PDFs:

```bash
python extract_sample.py /path/to/pdfs ./out --pages 60
```

## Troubleshooting

### Server won't start

**Issue**: `Model not found in cache`
**Solution**: Run `python download_model.py` first

**Issue**: `CUDA out of memory`
**Solution**: Lower `--workers` (each replica needs ~9.5GB, and `gundam` mode
peaks highest), close other GPU applications, or use `--device cpu` (slow).
`--gpu-memory-fraction` caps each worker so one cannot starve the rest.

### Only one request runs at a time

**Issue**: Multiple workers are online but throughput matches a single one
**Solution**: A single `REQ` socket is lockstep and will not send a second
request before the first is answered. Open one socket per in-flight request —
see the worker-pool client in [CLIENT_EXAMPLE.md](CLIENT_EXAMPLE.md).

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
┌──────────────┐             ┌────────────────────────────────┐
│              │   ZeroMQ    │  ROUTER :5555                  │
│  Your Code   │──REQ───────▶│      │                         │
│  (N sockets) │   TCP:5555  │  load-balancing broker         │
└──────────────┘             │      │  (dispatches to idle)   │
                             │  ipc:// ROUTER                 │
                             │   ╱   │   ╲                    │
                             │ w-0  w-1  w-2  ← model replica │
                             │                  per process   │
                             └────────────────────────────────┘
```

- **Protocol**: ZeroMQ ROUTER, which accepts plain REQ clients unchanged
- **Encoding**: JSON for metadata, Base64 for images
- **Port**: 5555 (configurable)
- **Backend**: a unix socket, so the internal hop skips the TCP stack

If a worker process dies mid-request, the broker answers that client with an
error rather than letting it hang, then restarts the replica. An exception
*inside* a worker (a CUDA OOM, say) returns an error for that one request
and leaves the worker in rotation.

### Files

| File | Role |
|---|---|
| `server.py` | CLI, broker, worker supervision |
| `ocr_engine.py` | Model loading, inference, the worker loop |
| `bench.py` | Throughput/latency benchmark and output-diffing tool |
| `test_ngram.py` | Proves the incremental n-gram processor matches the stock one |

## Credits

- DeepSeek-OCR Model: [DeepSeek AI](https://github.com/deepseek-ai/DeepSeek-OCR)
- Server implementation: Lightweight ZMQ wrapper

## License

MIT License - see model repository for model licensing
