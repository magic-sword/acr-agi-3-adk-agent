"""Execute generated Save & Run cells in an isolated offline Kaggle GPU container.

Mount the prepared datasets under /kaggle/input, an empty competition directory
at the expected path, and a writable output directory at /kaggle/working. The
development image supplies ARC dependencies. Requires enough free GPU memory;
this script starts its own Qwen server and does not manage other GPU services.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    args = parser.parse_args()
    if os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
        raise RuntimeError('This check exercises Save & Run, not the competition gateway')
    scope = {'__name__': '__main__'}
    cells = []
    try:
        for index, cell in enumerate(json.loads(args.notebook.read_text())['cells']):
            if cell['cell_type'] != 'code':
                continue
            started = time.perf_counter()
            source = cell['source']
            exec(compile(''.join(source) if isinstance(source, list) else source,
                         f'notebook_cell_{index}', 'exec'), scope)
            cells.append(dict(cell=index, seconds=time.perf_counter()-started, status='passed'))
            print(json.dumps(cells[-1]), flush=True)
    finally:
        process = scope.get('model_process')
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        Path('/kaggle/working/notebook-smoke.json').write_text(json.dumps(cells, indent=2)+'\n')
    if not Path('/kaggle/working/submission.parquet').is_file():
        raise RuntimeError('Notebook did not produce the Save & Run placeholder')


if __name__ == '__main__':
    main()
