"""Wait for the llama.cpp model to finish loading and accept an image request."""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request

from PIL import Image

pixel_buffer = io.BytesIO()
Image.new("RGB", (2, 2), (255, 0, 0)).save(pixel_buffer, format="PNG")
PIXEL = base64.b64encode(pixel_buffer.getvalue()).decode("ascii")


def wait(url: str, timeout: int = 240) -> None:
    until = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < until:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=4) as reply:
                if reply.status == 200:
                    break
        except (OSError, urllib.error.HTTPError) as error:
            last_error = error
        time.sleep(2)
    else:
        raise TimeoutError(f"Vision server did not become ready: {last_error}")
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({
            "model": "qwen3-vl-4b-instruct",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe the color in this image briefly."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PIXEL}},
            ]}], "max_tokens": 32,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as reply:
        answer = json.load(reply)["choices"][0]["message"]["content"]
    if not answer:
        raise RuntimeError("The vision smoke test returned no answer")
    print("Qwen3-VL image inference: OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    wait(args.url)
