# Client Examples

This document shows how to write client code to communicate with the DeepSeek-OCR server.

## Basic Client (Minimal Dependencies)

Only requires `pyzmq` and `Pillow`:

```python
import zmq
import json
import base64
from PIL import Image
import io

class DeepSeekOCRClient:
    def __init__(self, host="localhost", port=5555, timeout=30000):
        """
        Initialize connection to DeepSeek-OCR server

        Args:
            host: Server hostname or IP
            port: Server port
            timeout: Request timeout in milliseconds
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.connect(f"tcp://{host}:{port}")

    def process_image(self, image_path, mode="base", preserve_layout=True):
        """
        Process an image with DeepSeek-OCR

        Args:
            image_path: Path to image file
            mode: Resolution mode ('tiny', 'small', 'base', 'large', 'gundam')
            preserve_layout: Whether to include grounding tokens

        Returns:
            Dictionary with 'text', 'layout', 'confidence', and 'metadata'
        """
        # Load image
        image = Image.open(image_path)

        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        # Encode as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Build request
        request = {
            'image': image_b64,
            'mode': mode,
            'preserve_layout': preserve_layout
        }

        # Send and receive
        self.socket.send_string(json.dumps(request))
        response_str = self.socket.recv_string()
        return json.loads(response_str)

    def close(self):
        """Close connection"""
        self.socket.close()
        self.context.term()


# Usage
if __name__ == "__main__":
    client = DeepSeekOCRClient(host="localhost", port=5555)

    result = client.process_image("document.jpg", mode="base")
    print(result['text'])

    client.close()
```

## Processing Multiple Images

```python
import glob
from pathlib import Path

client = DeepSeekOCRClient(host="localhost")

# Process all JPGs in a directory
for image_path in glob.glob("documents/*.jpg"):
    print(f"Processing {image_path}...")

    result = client.process_image(image_path, mode="base")

    # Save result
    output_path = Path(image_path).with_suffix('.txt')
    output_path.write_text(result['text'])

    print(f"  → {len(result['text'])} chars, {result['metadata']['processing_time_ms']}ms")

client.close()
```

## Parsing Grounding Tokens

Extract structure from the grounding tokens:

```python
import re

def parse_grounding_tokens(text):
    """
    Parse grounding tokens from DeepSeek-OCR output

    Returns:
        List of {'type': str, 'bbox': [x1,y1,x2,y2], 'content': str}
    """
    # Pattern: <|ref|>TYPE<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>CONTENT
    pattern = r'<\|ref\|>([^<]+)<\|/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|/det\|>\s*([^<]*(?:<[^|][^>]*>[^<]*)*)'

    elements = []
    for match in re.finditer(pattern, text, re.DOTALL):
        element_type = match.group(1)
        bbox_str = match.group(2)
        content = match.group(3).strip()

        # Parse bbox coordinates
        bbox = [int(x.strip()) for x in bbox_str.split(',')]

        elements.append({
            'type': element_type,
            'bbox': bbox,
            'content': content
        })

    return elements


# Usage
result = client.process_image("document.jpg", preserve_layout=True)
elements = parse_grounding_tokens(result['text'])

for elem in elements:
    print(f"{elem['type']:15} @ {elem['bbox']}")
    print(f"  {elem['content'][:50]}...")
```

## Extracting Tables

Parse HTML tables from the output:

```python
from html.parser import HTMLParser

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.current_table = []
        self.current_row = []
        self.current_cell = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
            self.current_table = []
        elif tag == 'tr':
            self.in_row = True
            self.current_row = []
        elif tag == 'td':
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
            self.tables.append(self.current_table)
        elif tag == 'tr':
            self.in_row = False
            self.current_table.append(self.current_row)
        elif tag == 'td':
            self.in_cell = False
            self.current_row.append(''.join(self.current_cell))

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


# Usage
result = client.process_image("index_page.jpg")
parser = TableParser()
parser.feed(result['text'])

for i, table in enumerate(parser.tables):
    print(f"\nTable {i+1}: {len(table)} rows")
    for row in table[:3]:  # First 3 rows
        print("  | ".join(cell[:20] for cell in row))
```

## Batch Processing with Progress

```python
from tqdm import tqdm
import time

def process_batch(image_paths, output_dir, mode="base"):
    """Process multiple images with progress bar"""
    client = DeepSeekOCRClient(host="192.168.1.234", timeout=300000)  # 5 min

    results = []
    for image_path in tqdm(image_paths, desc="Processing"):
        try:
            result = client.process_image(image_path, mode=mode)

            # Save to file
            output_file = Path(output_dir) / f"{Path(image_path).stem}.txt"
            output_file.write_text(result['text'])

            results.append({
                'file': image_path,
                'chars': len(result['text']),
                'time_ms': result['metadata']['processing_time_ms']
            })

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({'file': image_path, 'error': str(e)})

    client.close()
    return results


# Usage
images = list(Path("documents").glob("*.jpg"))
results = process_batch(images, output_dir="output", mode="gundam")

# Print stats
total_time = sum(r.get('time_ms', 0) for r in results) / 1000
total_chars = sum(r.get('chars', 0) for r in results)
print(f"\nProcessed {len(images)} images in {total_time:.1f}s")
print(f"Total: {total_chars} characters")
```

## Error Handling

```python
import zmq

def robust_process_image(client, image_path, retries=3):
    """Process image with retries on timeout"""
    for attempt in range(retries):
        try:
            result = client.process_image(image_path)
            return result
        except zmq.Again:
            print(f"Timeout on attempt {attempt+1}/{retries}, retrying...")
            time.sleep(1)
        except zmq.ZMQError as e:
            print(f"ZMQ error: {e}")
            # Reconnect
            client.close()
            client = DeepSeekOCRClient(host="localhost")

    raise TimeoutError(f"Failed after {retries} attempts")


# Usage
try:
    result = robust_process_image(client, "large_document.jpg")
except TimeoutError:
    print("Document too complex, try splitting into regions")
```

## Async Processing (Optional)

For high-throughput applications:

```python
import asyncio
import zmq.asyncio

class AsyncDeepSeekOCRClient:
    def __init__(self, host="localhost", port=5555):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    async def process_image(self, image_path, mode="base"):
        # Same encoding logic as before
        image = Image.open(image_path)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        request = {
            'image': image_b64,
            'mode': mode,
            'preserve_layout': True
        }

        await self.socket.send_string(json.dumps(request))
        response_str = await self.socket.recv_string()
        return json.loads(response_str)


# Usage
async def main():
    client = AsyncDeepSeekOCRClient(host="localhost")

    tasks = [
        client.process_image(f"page_{i}.jpg")
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} images")


asyncio.run(main())
```

## Integration with Existing OCR Pipeline

Replace Tesseract calls:

```python
# Before (with pytesseract)
import pytesseract
text = pytesseract.image_to_string(image_path, lang='nld')

# After (with DeepSeek-OCR)
client = DeepSeekOCRClient(host="localhost")
result = client.process_image(image_path)
text = result['text']

# Bonus: You also get layout information!
elements = parse_grounding_tokens(result['text'])
tables = [e for e in elements if e['type'] == 'table']
```

## Performance Tips

1. **Reuse the client connection**:
   ```python
   # Good: One connection for many requests
   client = DeepSeekOCRClient(...)
   for image in images:
       result = client.process_image(image)
   client.close()

   # Bad: New connection per request
   for image in images:
       client = DeepSeekOCRClient(...)  # Slow!
       result = client.process_image(image)
       client.close()
   ```

2. **Choose the right mode**:
   - `small` for speed (2s/page)
   - `base` for quality (18s/page)
   - `gundam` for high-DPI archival documents (38s/page)

3. **Increase timeout for large images**:
   ```python
   client = DeepSeekOCRClient(timeout=600000)  # 10 minutes
   ```

4. **Pre-process images**:
   - Convert to PNG or high-quality JPEG
   - Don't downscale before sending (let the server handle it)
   - For very large images (>10MB), consider tiling on client side
