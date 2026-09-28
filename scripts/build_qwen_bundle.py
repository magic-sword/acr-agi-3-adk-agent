"""Stage only required Qwen assets; flatten shared-library symlinks for Kaggle."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.model_runtime import MODEL, PROJECTOR
from scripts.build_sam_bundle import sha256

SOURCE = ROOT / '.cache/model-cache/qwen3-vl-4b'
OUT = ROOT / '.cache/kaggle-qwen-bundle'


def build():
    names = [MODEL, PROJECTOR, 'llama-server', 'llama-LICENSE', 'llama-commit.txt',
             'MODEL-SOURCE.txt', 'Qwen-LICENSE', 'dataset-metadata.json']
    names += sorted(p.name for p in SOURCE.glob('lib*.so*'))
    for name in names:
        if not (SOURCE / name).is_file():
            raise FileNotFoundError(f'{name} missing; run make model-download and make model-runtime')
    if not any(name.startswith('libggml-cuda.so') for name in names):
        raise ValueError('CUDA runtime library missing; run make model-runtime')
    for name in (MODEL, PROJECTOR):
        with (SOURCE / name).open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError(f'Invalid GGUF: {name}')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'llama_commit': (SOURCE / 'llama-commit.txt').read_text().strip(), 'files': {}}
    for name in names:
        digest = sha256(SOURCE / name)
        target = OUT / name
        if not target.is_file() or sha256(target) != digest:
            shutil.copyfile(SOURCE / name, target, follow_symlinks=True)
        manifest['files'][name] = digest
    (OUT / 'qwen-bundle.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if {p.name for p in OUT.iterdir()} != set(names) | {'qwen-bundle.json'}:
        raise ValueError('Unexpected files in Qwen staging directory; inspect before uploading')
    print(f'Qwen bundle ready: {OUT} ({sum(p.stat().st_size for p in OUT.iterdir()):,} bytes)')


if __name__ == '__main__':
    build()
