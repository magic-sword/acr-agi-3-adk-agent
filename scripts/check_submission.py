"""Check staged submission locally; --remote is read-only; never upload."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.submission_runtime import configure_sam, warmup
from scripts.build_notebook import build, OUT
from scripts.build_adk_bundle import OUT as ADK_OUT, manifest as adk_manifest
from scripts.build_sam_bundle import sha256


def remote_check(manifest):
    if not os.environ.get('KAGGLE_API_TOKEN'):
        os.environ['KAGGLE_API_TOKEN'] = (ROOT / '.kaggle/access_token').read_text().strip()
    from kaggle import api
    metadata = json.loads((ROOT / 'notebooks/kernel-metadata.json').read_text())
    sam_id = metadata['id'].split('/')[0] + '/arc-agi-3-sam-vit-b'
    qwen = json.loads((ROOT / '.cache/kaggle-qwen-bundle/qwen-bundle.json').read_text())
    adk_id = metadata['id'].split('/')[0] + '/arc-agi-3-adk-wheels'
    bundles = {sam_id: ('sam-bundle.json', manifest),
               adk_id: ('adk-bundle.json', adk_manifest()),
               'magicsword001/arc-agi-3-qwen3-vl-4b': ('qwen-bundle.json', qwen)}
    for dataset, (manifest_name, expected) in bundles.items():
        required = (set(expected['files']) | {manifest_name}) - {'dataset-metadata.json'}
        response = api.dataset_list_files(dataset, page_size=100)
        if response.error_message:
            raise RuntimeError(response.error_message)
        names = {file.name for file in response.files}
        if required - names:
            raise RuntimeError(f'{dataset}: missing {sorted(required - names)}; upload the prepared assets first')
        print(f'Remote files OK: {dataset}')
        with tempfile.TemporaryDirectory() as directory:
            api.dataset_download_file(dataset, manifest_name, path=directory, force=True, quiet=True)
            path = Path(directory) / manifest_name
            if path.is_file():
                actual = json.loads(path.read_text())
            else:
                with zipfile.ZipFile(str(path) + '.zip') as archive:
                    actual = json.loads(archive.read(manifest_name))
            if actual != expected:
                raise RuntimeError(f'{dataset}: remote assets are stale; upload a new dataset version')
    print('Remote manifests match local staged assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()
    notebook = build()
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(cell['source'], f'cell_{index}', 'exec')
    if json.loads(OUT.read_text()) != notebook:
        raise RuntimeError('Generated notebook is stale; run make notebook')
    size = OUT.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f'Notebook is {size:,} bytes; Kaggle rejects kernel sources of 1 MB or more')
    manifest = json.loads((ROOT / 'config/sam-bundle.json').read_text())
    qwen_root = ROOT / '.cache/kaggle-qwen-bundle'
    qwen = json.loads((qwen_root / 'qwen-bundle.json').read_text())
    for name, digest in qwen['files'].items():
        if sha256(qwen_root / name) != digest:
            raise ValueError(f'Qwen asset checksum mismatch: {name}')
    print('Local Qwen bundle checksums: OK')
    adk = adk_manifest()
    for name, digest in adk['files'].items():
        if sha256(ADK_OUT / name) != digest:
            raise ValueError(f'ADK wheel checksum mismatch: {name}')
    print('Local ADK wheel checksums: OK')
    with tempfile.TemporaryDirectory(prefix='sam-submission-') as directory:
        configure_sam(manifest, ROOT / '.cache/kaggle-sam-bundle', Path(directory))
        print('Local notebook, SAM source and checkpoint checksums: OK')
        if args.gpu:
            warmup(ROOT / 'outputs/kaggle-ready/submission-preflight.json')
    if args.remote:
        remote_check(manifest)


if __name__ == '__main__':
    main()
