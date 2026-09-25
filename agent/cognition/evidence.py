"""Immutable, bounded, per-run observation archive independent of diagnostic logging."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.observation import VISUAL_PAYLOAD_KEYS, render_current, render_animation_page, png_base64


class EvidenceStore:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self._directory = TemporaryDirectory(prefix='arc-evidence-')
        self.index = {}
        self.segment = 0
        self.pinned = set()

    def add(self, observation, *, boundary=False, source_action=None):
        oid = observation['observation_id']
        if oid in self.index:
            return
        if boundary:
            self.segment += 1
        record = deepcopy(observation)
        record['segment'] = self.segment
        record['source_action'] = deepcopy(source_action)
        path = Path(self._directory.name) / f'{observation["step"]}.json'
        path.write_text(json.dumps(record))
        self.index[oid] = {'path': path, 'observation_id': oid, 'step': record['step'],
                           'segment': self.segment, 'levels_completed': record.get('levels_completed', 0),
                           'source_action': {k: v for k, v in (source_action or {}).items()
                                             if k in ('decision_id', 'observation_id', 'step', 'action')},
                           'event_id': record.get('animation', {}).get('event_id')}
        self._evict(protect={oid})

    def pin(self, observation_ids):
        ids = set(observation_ids)
        if len(ids) > 2 or not ids <= self.index.keys():
            raise ValueError('pin at most two retained experiment endpoints')
        self.pinned = ids
        self._evict()

    def _evict(self, protect=frozenset()):
        while len(self.index) > self.capacity:
            key = next((key for key in self.index if key not in self.pinned | protect), None)
            if key is None:
                break
            self.index.pop(key)['path'].unlink()

    def list(self):
        return [{k: v for k, v in item.items() if k != 'path'} for item in self.index.values()]

    def get(self, observation_id):
        if observation_id not in self.index:
            raise ValueError('Observation unavailable or evicted; use list_observations')
        return json.loads(self.index[observation_id]['path'].read_text())

    def view(self, observation_id):
        record = self.get(observation_id)
        result = {k: v for k, v in record.items() if k not in VISUAL_PAYLOAD_KEYS | {'changed_cells'}}
        result.update(view='recorded_observation', time_advanced=False)
        if record.get('grid'):
            result['_image_png_base64'] = png_base64(render_current(
                record['grid'],
                label=f'RECORDED observation: step {record["step"]} (not live)'))
        elif record.get('image_png_base64'):
            result['_image_png_base64'] = record['image_png_base64']
        else:
            result['grid'] = record.get('grid')
        return result

    def compare(self, before_id, after_id, offset=0):
        before, after = self.get(before_id), self.get(after_id)
        if before['segment'] != after['segment']:
            raise ValueError('Cannot compare across a level or RESET boundary')
        if before['step'] >= after['step']:
            raise ValueError('Comparison requires ordered distinct observations')
        a, b = before.get('grid'), after.get('grid')
        if not a or not b or len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError('Comparable color-ID grids unavailable')
        if type(offset) is not int or offset < 0:
            raise ValueError('offset must be a nonnegative integer')
        changes = [{'x': x, 'y': y, 'before': a[y][x], 'after': v}
                   for y, row in enumerate(b) for x, v in enumerate(row) if a[y][x] != v]
        intervening = [r for r in self.list() if before['step'] < r['step'] <= after['step']]
        # Model gets both labeled endpoints via the normal image transport.
        from PIL import Image, ImageDraw
        images = [render_current(r['grid'],
                                 label=f'RECORDED step {r["step"]} (not live)')
                  for r in (before, after)]
        sheet = Image.new('RGB', (sum(i.width for i in images), max(i.height for i in images)+40), 'white')
        draw = ImageDraw.Draw(sheet)
        x = 0
        for label, record, img in zip(('BEFORE', 'AFTER'), (before, after), images):
            draw.text((x+8, 10), f'{label} recorded step {record["step"]}', fill='black')
            sheet.paste(img, (x, 40)); x += img.width
        return {'before_id': before_id, 'after_id': after_id, 'view': 'recorded_comparison',
                'time_advanced': False, 'changed_cell_count': len(changes),
                'changed_cells': changes[offset:offset+96],
                'next_offset': offset+96 if offset+96 < len(changes) else None,
                'transitions': intervening, 'causal_attribution': 'not inferred from pixel changes',
                '_image_png_base64': png_base64(sheet)}

    def animation(self, event_id, start_frame=0):
        entry = next((r for r in self.index.values() if r['event_id'] == event_id), None)
        if entry is None:
            raise ValueError('Recorded event unavailable or evicted')
        record = self.get(entry['observation_id'])
        if not record.get('_visual_frames'):
            raise ValueError('Intermediate frames unavailable')
        sheet, next_frame = render_animation_page(record['_visual_frames'], event_id, start_frame)
        return {'observation_id': entry['observation_id'], 'event': record['animation'],
                'view': 'historical_replay_not_live', 'start_frame': start_frame,
                'next_start_frame': next_frame,
                'time_advanced': False, '_image_png_base64': png_base64(sheet)}

    def close(self):
        self._directory.cleanup()
