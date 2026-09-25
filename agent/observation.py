"""Visual evidence assembly using the deterministic host renderer."""
import hashlib
from . import rendering as _renderer
import json

frame_image = _renderer.frame_image
render_current = _renderer.render_current
render_animation_page = _renderer.render_animation_page
png_base64 = _renderer.png_base64

# Raw frames and media are ephemeral, never textual model context or cognitive memory.
VISUAL_PAYLOAD_KEYS = {'image_png_base64', '_visual_frames'}


def attach_visuals(observation, frames, source_action=None):
    images = [frame_image(frame) for frame in frames]
    if any(image.size != images[-1].size for image in images):
        raise ValueError('animation frame dimensions changed within one action')
    width, height = images[-1].size
    digest = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()[:16]
    event_id = f"{observation['game_id']}:{observation['step']}:{digest}"
    observation.update(
        _visual_frames=frames,
        viewport={'origin': list(_renderer.ORIGIN), 'scale': _renderer.SCALE,
                  'width': width, 'height': height, 'coordinates': 'original pixels; x=column, y=row'},
        animation={'event_id': event_id, 'frame_count': len(frames), 'source_step': observation['step'],
                   'source_action': source_action, 'available': len(frames) > 1,
                   'status': 'completed recorded transition; current image is final frame',
                   'timing': 'frame order only; real durations unavailable'},
        image_png_base64=png_base64(render_current(frames[-1])))
