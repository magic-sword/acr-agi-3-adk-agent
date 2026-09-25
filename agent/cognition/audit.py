"""Small, JSON-safe tool audit records; no images or private model reasoning."""
from __future__ import annotations

import hashlib
import json
import base64
from copy import deepcopy


def request_snapshot(payload, directory):
    """Keep the actual request text/schema; content-address image bytes separately."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    snapshot = deepcopy(payload)
    for message in snapshot['messages']:
        if not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if part.get('type') != 'image_url':
                continue
            url = part['image_url']['url']
            if not url.startswith('data:'):
                continue
            header, encoded = url.split(',', 1)
            data = base64.b64decode(encoded)
            image_hash = hashlib.sha256(data).hexdigest()
            relative = f'request-images/{image_hash}.bin'
            if directory is not None:
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
            part['image_url'] = {'sha256': image_hash, 'path': relative,
                                 'mime_type': header[5:].split(';')[0],
                                 'detail': part['image_url'].get('detail')}
    return {'request_sha256': digest, 'request': snapshot}


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
