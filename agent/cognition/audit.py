"""Small, JSON-safe tool audit records; no images or private model reasoning."""
from __future__ import annotations

import hashlib
import json


def compact(value):
    """Keep evidence metadata and errors without copying image/skill payloads."""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()
                if key not in {"_image_png_base64", "image_png_base64", "_visual_frames"}}
    if isinstance(value, (list, tuple)):
        return [compact(item) for item in value]
    if isinstance(value, str) and len(value) > 1000:
        return {"preview": value[:300], "characters": len(value),
                "sha256": hashlib.sha256(value.encode()).hexdigest()}
    return value


def tool_status(response):
    if isinstance(response, dict):
        if response.get("error") or response.get("error_code") or response.get("accepted") is False:
            return "error"
    return "success"


def append_record(directory, run_id, kind, record):
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{run_id}.{kind}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
