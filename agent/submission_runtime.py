"""Strict offline Kaggle setup and one-process SAM warmup before the Swarm."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def configure_sam(expected, input_root=Path('/kaggle/input'),
                  source_root=Path('/tmp/arc_sam')):
    """Verify pinned assets before starting Qwen; never silently use program-only."""
    matches = sorted(input_root.rglob('sam-bundle.json'))
    matches = [p for p in matches if json.loads(p.read_text()) == expected]
    if not matches:
        raise FileNotFoundError('Attach the prepared arc-agi-3-sam-vit-b dataset (matching sam-bundle.json)')
    bundle = matches[0].parent
    for name, digest in expected['files'].items():
        with (bundle / name).open('rb') as stream:
            actual = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                actual.update(chunk)
            if actual.hexdigest() != digest:
                raise ValueError(f'SAM asset checksum mismatch: {name}')
    sources = json.loads((bundle / 'sam-source.json').read_text())
    if set(sources) != set(expected['source_files']):
        raise ValueError('Unexpected SAM source contents')
    for name, digest in expected['source_files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe SAM source path')
        data = sources[name].encode()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'SAM source checksum mismatch: {name}')
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    os.environ.update(COGNITION_PROPOSALS='sam_initial',
                      COGNITION_SAM_CHECKPOINT=str((bundle / 'sam_vit_b_01ec64.pth').resolve()),
                      COGNITION_SAM_SOURCE=str(source_root.resolve()),
                      COGNITION_SAM_DEVICE='cuda', MPLBACKEND='agg')
    # Compose can run a numeric UID absent from /etc/passwd. PyTorch's default
    # cache discovery otherwise fails in getpass.getuser() before SAM loads.
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/arc_torch_cache')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/arc_matplotlib')
    return bundle


def warmup(report_path=Path('/kaggle/working/submission-preflight.json')):
    """Exercise actual SAM and Qwen vision while keeping SAM for all game threads."""
    import base64
    import io
    import time
    import urllib.request
    import torch
    from agent.rendering import frame_image
    from agent.cognition.sam_proposals import default_proposer
    from agent.cognition.hybrid_perception import HybridPerception
    started = time.perf_counter()
    grid = [[0] * 64 for _ in range(64)]
    for y in range(20, 40):
        grid[y][20:40] = [8] * 20
    provider = default_proposer()
    result = provider.propose(grid)
    if not result['boxes']:
        raise RuntimeError('SAM warmup produced no candidates')
    if not all(HybridPerception().proposer is provider for _ in range(2)):
        raise RuntimeError('SAM provider is not shared across game runtimes')
    buffer = io.BytesIO()
    frame_image(grid).resize((384, 384)).save(buffer, format='PNG')
    payload = dict(model='qwen3-vl-4b-instruct', temperature=0, max_tokens=1,
                   grammar='root ::= "1"', messages=[dict(role='user', content=[
                       dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
                            base64.b64encode(buffer.getvalue()).decode())),
                       dict(type='text', text='Acknowledge this image. Reply 1.')])])
    url = os.environ.get('VLM_API_BASE', 'http://127.0.0.1:8080/v1').rstrip('/')
    request = urllib.request.Request(url + '/chat/completions',
                                    data=json.dumps(payload).encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.load(response)
    if reply['choices'][0]['message']['content'].strip() != '1':
        raise RuntimeError('Qwen vision warmup failed')
    report = dict(status='ready', sam_shared=True, sam_candidates=len(result['boxes']),
                  sam_seconds=result['seconds'], total_seconds=time.perf_counter()-started,
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  sam_reserved_mib=torch.cuda.memory_reserved()/2**20,
                  checkpoint=os.environ['COGNITION_SAM_CHECKPOINT'])
    print(json.dumps(report), flush=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    warmup()
    if '--check-only' not in sys.argv:
        sys.argv = ['main.py', '--agent', 'myagent']
        runpy.run_path('main.py', run_name='__main__')
