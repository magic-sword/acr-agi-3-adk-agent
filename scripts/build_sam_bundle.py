"""Build the pinned, private Kaggle SAM dataset locally; never upload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.cache/kaggle-sam-bundle'


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def build(source, checkpoint, out=OUT):
    manifest = json.loads((ROOT / 'config/sam-bundle.json').read_text())
    if sha256(checkpoint) != manifest['files']['sam_vit_b_01ec64.pth']:
        raise ValueError('SAM checkpoint differs from the measured ViT-B weights')
    for name, digest in manifest['source_files'].items():
        if sha256(source / name) != digest:
            raise ValueError(f'SAM source differs from pinned version: {name}')
    out.mkdir(parents=True, exist_ok=True)
    # JSON avoids automatic archive extraction and directory skipping on Kaggle.
    sources = {name: (source / name).read_text() for name in manifest['source_files']}
    (out / 'sam-source.json').write_text(json.dumps(sources, ensure_ascii=True, indent=2)+'\n')
    if sha256(out / 'sam-source.json') != manifest['files']['sam-source.json']:
        raise ValueError('Non-reproducible SAM source bundle')
    target = out / 'sam_vit_b_01ec64.pth'
    if not target.is_file() or sha256(target) != manifest['files'][target.name]:
        shutil.copyfile(checkpoint, target)
    shutil.copyfile(source / 'LICENSE', out / 'LICENSE')
    (out / 'sam-bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    kernel = json.loads((ROOT / 'notebooks/kernel-metadata.json').read_text())
    owner = kernel['id'].split('/')[0]
    dataset = f'{owner}/arc-agi-3-sam-vit-b'
    if dataset not in kernel['dataset_sources']:
        raise ValueError(f'Attach {dataset} in kernel-metadata.json')
    metadata = dict(title='ARC AGI 3 SAM ViT B offline bundle', id=dataset,
                    licenses=[{'name': 'Apache 2.0'}])
    (out / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    expected = set(manifest['files']) | {'LICENSE', 'sam-bundle.json', 'dataset-metadata.json'}
    if {p.name for p in out.iterdir()} != expected:
        raise ValueError('Unexpected files in SAM staging directory; inspect before uploading')
    print(f'SAM bundle ready: {out} ({sum(p.stat().st_size for p in out.iterdir()):,} bytes)')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/sam-deps/segment-anything')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/sam-deps/sam_vit_b_01ec64.pth')
    args = parser.parse_args()
    build(args.source, args.checkpoint)
