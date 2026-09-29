"""Stage the pinned ADK wheels as a private Kaggle dataset locally; never upload.

Kaggle rejects kernel sources over 1 MB, so the wheels cannot be embedded in the notebook.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_sam_bundle import sha256

OUT = ROOT / '.cache/kaggle-adk-bundle'
WHEELS = [ROOT / 'third_party/wheels' / name for name in (
    'google_adk-2.0.0-py3-none-any.whl', 'google_genai-1.72.0-py3-none-any.whl',
    'google_auth-2.49.2-py3-none-any.whl',
)]
# A .whl is a zip archive; the suffix keeps Kaggle from extracting it on upload.
SUFFIX = '.bin'


def manifest():
    return {'schema': 1, 'files': {wheel.name + SUFFIX: sha256(wheel) for wheel in WHEELS}}


def build(out=OUT):
    for wheel in WHEELS:
        if not wheel.is_file():
            raise FileNotFoundError(wheel)
    expected = manifest()
    out.mkdir(parents=True, exist_ok=True)
    for wheel in WHEELS:
        target = out / (wheel.name + SUFFIX)
        if not target.is_file() or sha256(target) != expected['files'][target.name]:
            shutil.copyfile(wheel, target)
    (out / 'adk-bundle.json').write_text(json.dumps(expected, indent=2) + '\n')
    kernel = json.loads((ROOT / 'notebooks/kernel-metadata.json').read_text())
    dataset = kernel['id'].split('/')[0] + '/arc-agi-3-adk-wheels'
    if dataset not in kernel['dataset_sources']:
        raise ValueError(f'Attach {dataset} in kernel-metadata.json')
    metadata = dict(title='ARC AGI 3 ADK offline wheels', id=dataset,
                    licenses=[{'name': 'Apache 2.0'}])
    (out / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if {p.name for p in out.iterdir()} != set(expected['files']) | {'adk-bundle.json', 'dataset-metadata.json'}:
        raise ValueError('Unexpected files in ADK staging directory; inspect before uploading')
    print(f'ADK bundle ready: {out} ({sum(p.stat().st_size for p in out.iterdir()):,} bytes)')
    return out


if __name__ == '__main__':
    build()
