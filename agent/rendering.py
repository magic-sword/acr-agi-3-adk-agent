"""Deterministic game-only rendering; never steps the environment.

CLI input: JSON with frames and event_id.
current writes only the final PNG; replay writes a single-play historical GIF.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

# Support both packaged imports and direct CLI execution from any directory.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
from arc_agi.rendering import COLOR_MAP, hex_to_rgb

SCALE = 6
ORIGIN = (32, 44)


def frame_image(frame):
    pixels = np.asarray(frame)
    if pixels.size == 0 or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError('frames must contain integer pixels')
    if pixels.ndim == 2:
        if pixels.min() < 0 or pixels.max() > 15:
            raise ValueError('invalid color IDs')
        palette = np.asarray([hex_to_rgb(COLOR_MAP[i]) for i in range(16)], dtype=np.uint8)
        return Image.fromarray(palette[pixels])
    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4) and pixels.min() >= 0 and pixels.max() <= 255:
        return Image.fromarray(pixels.astype(np.uint8)).convert('RGB')
    raise ValueError(f'Unsupported frame shape: {pixels.shape}')


def render_current(frame, *, label='CURRENT: game pixels'):
    board = frame_image(frame)
    ox, oy = ORIGIN
    bw, bh = board.width * SCALE, board.height * SCALE
    image = Image.new('RGB', (max(260, ox+bw+12), oy+bh+24), '#18202b')
    image.paste(board.resize((bw, bh), Image.Resampling.NEAREST), (ox, oy))
    d = ImageDraw.Draw(image)
    d.text((12, 10), label, fill='white')
    for x in sorted(set(range(0, board.width, 8)) | {board.width-1}):
        d.text((ox+x*SCALE, oy-16), str(x), fill='white')
    for y in sorted(set(range(0, board.height, 8)) | {board.height-1}):
        d.text((3, oy+y*SCALE), str(y), fill='white')
    return image


def png_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def render_animation_page(frames, event_id, start_frame=0):
    if type(start_frame) is not int or not 0 <= start_frame < len(frames):
        raise ValueError('start_frame outside recorded animation')
    end = min(start_frame + 4, len(frames))
    images = [render_current(frames[i],
              label=f'HISTORY REPLAY {event_id} | frame {i+1}/{len(frames)} | NOT LIVE')
              for i in range(start_frame, end)]
    sheet = Image.new('RGB', (images[0].width * min(2, len(images)), images[0].height * ((len(images)+1)//2)))
    for i, image in enumerate(images):
        sheet.paste(image, ((i % 2) * image.width, (i // 2) * image.height))
    return sheet, end if end < len(frames) else None


def save_replay(frames, event_id, path):
    images = [render_current(frame,
              label=f'HISTORY REPLAY {event_id} | {i+1}/{len(frames)} | NOT LIVE')
              for i, frame in enumerate(frames)]
    # No loop extension: play once, hold final frame. Duration is presentation-only.
    images[0].save(path, format='GIF', save_all=True, append_images=images[1:], duration=150, disposal=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['current', 'replay'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    frames = data['frames']
    if args.mode == 'current':
        render_current(frames[-1]).save(args.output, format='PNG')
    else:
        save_replay(frames, data['event_id'], args.output)


if __name__ == '__main__':
    main()
