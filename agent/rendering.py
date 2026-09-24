"""Deterministic game/cursor/controller rendering; never steps the environment.

CLI input: JSON with frames, available_actions, cursor and event_id.
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
from agent.controls import ACTION_TO_BUTTON

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


def validate_cursor(cursor, width, height):
    if cursor is None:
        return {'x': width // 2, 'y': height // 2}
    if any(type(cursor.get(k)) is not int for k in ('x', 'y')):
        raise ValueError('cursor requires integer x,y')
    if not (0 <= cursor['x'] < width and 0 <= cursor['y'] < height):
        raise ValueError('cursor outside original frame')
    return {'x': cursor['x'], 'y': cursor['y']}


def render_current(frame, available_actions, cursor=None, *, label='CURRENT: final received frame'):
    board = frame_image(frame)
    cursor = validate_cursor(cursor, *board.size)
    ox, oy = ORIGIN
    bw, bh = board.width * SCALE, board.height * SCALE
    panel = max(ox + bw + 24, 430)
    image = Image.new('RGB', (panel + 310, max(oy + bh + 62, 370)), '#18202b')
    image.paste(board.resize((bw, bh), Image.Resampling.NEAREST), (ox, oy))
    d = ImageDraw.Draw(image)
    d.text((12, 10), label, fill='white')
    d.text((ox, oy + bh + 16), f"Cursor x={cursor['x']} y={cursor['y']} (original pixels)", fill='white')
    d.text((ox, oy + bh + 32), 'Host cursor overlay; center is the target. No click sent.', fill='white')
    for x in sorted(set(range(0, board.width, 8)) | {board.width - 1}):
        d.text((ox + x * SCALE, oy - 16), str(x), fill='white')
    for y in sorted(set(range(0, board.height, 8)) | {board.height - 1}):
        d.text((3, oy + y * SCALE), str(y), fill='white')
    cx, cy = ox + cursor['x'] * SCALE + SCALE // 2, oy + cursor['y'] * SCALE + SCALE // 2
    # Open reticle preserves the target cell's center color, including at edges.
    for color, radius in [('black', 9), ('white', 7)]:
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color, width=2)
    d.text((panel, 32), 'CONTROLLER (host UI)', fill='white')
    buttons = [('RESET', 196, 65), ('ACTION1', 82, 65), ('ACTION3', 0, 125),
               ('ACTION2', 82, 185), ('ACTION4', 164, 125),
               ('ACTION5', 0, 250), ('ACTION6', 98, 250),
               ('ACTION7', 196, 250)]
    for action, x, y in buttons:
        x += panel
        enabled = action in available_actions
        d.rounded_rectangle((x, y, x+88, y+50), radius=9,
                            fill='#356384' if enabled else '#30353c', outline='white' if enabled else '#666666')
        d.text((x+6, y+7), ACTION_TO_BUTTON[action], fill='white' if enabled else '#888888')
        d.text((x+6, y+29), 'button', fill='white' if enabled else '#888888')
        if action in ('ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'):
            # Directional silhouettes supplement textual labels.
            dx, dy = {'ACTION1': (0,-1), 'ACTION2': (0,1), 'ACTION3': (-1,0), 'ACTION4': (1,0)}[action]
            ax, ay = x+70, y+15
            d.polygon([(ax+dx*9, ay+dy*9), (ax-dy*6-dx*4, ay+dx*6-dy*4),
                       (ax+dy*6-dx*4, ay-dx*6-dy*4)], fill='white' if enabled else '#888888')
    d.text((panel, 316), 'Dim = unavailable. Labels describe inputs.', fill='white')
    d.text((panel, 332), 'Game effects require observed evidence.', fill='white')
    return image


def png_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def render_animation_page(frames, available_actions, cursor, event_id, start_frame=0):
    if type(start_frame) is not int or not 0 <= start_frame < len(frames):
        raise ValueError('start_frame outside recorded animation')
    end = min(start_frame + 4, len(frames))
    images = [render_current(frames[i], available_actions, cursor,
              label=f'HISTORY REPLAY {event_id} | frame {i+1}/{len(frames)} | NOT LIVE')
              for i in range(start_frame, end)]
    sheet = Image.new('RGB', (images[0].width * min(2, len(images)), images[0].height * ((len(images)+1)//2)))
    for i, image in enumerate(images):
        sheet.paste(image, ((i % 2) * image.width, (i // 2) * image.height))
    return sheet, end if end < len(frames) else None


def save_replay(frames, available_actions, cursor, event_id, path):
    images = [render_current(frame, available_actions, cursor,
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
        render_current(frames[-1], data['available_actions'], data.get('cursor')).save(args.output, format='PNG')
    else:
        save_replay(frames, data['available_actions'], data.get('cursor'), data['event_id'], args.output)


if __name__ == '__main__':
    main()
