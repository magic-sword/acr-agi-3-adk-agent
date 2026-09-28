import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_qwen_bundle as bundle


class QwenBundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.out = self.root / 'out'
        for name in [bundle.MODEL, bundle.PROJECTOR, 'llama-server', 'llama-LICENSE',
                     'llama-commit.txt', 'MODEL-SOURCE.txt', 'Qwen-LICENSE', 'dataset-metadata.json',
                     'libggml-cuda.so.0']:
            (self.source / name).write_bytes(b'GGUF fixture')
        (self.source / 'libggml-cuda.so').symlink_to('libggml-cuda.so.0')
        self.addCleanup(patch.stopall)
        patch.object(bundle, 'SOURCE', self.source).start()
        patch.object(bundle, 'OUT', self.out).start()

    def test_shared_library_symlinks_become_regular_uploadable_files(self):
        bundle.build()
        self.assertFalse((self.out / 'libggml-cuda.so').is_symlink())
        self.assertEqual((self.out / 'libggml-cuda.so').read_bytes(), b'GGUF fixture')
        manifest = json.loads((self.out / 'qwen-bundle.json').read_text())
        self.assertEqual(set(manifest['files']), {p.name for p in self.source.iterdir()})

    def test_missing_cuda_runtime_rejected(self):
        (self.source / 'libggml-cuda.so').unlink()
        (self.source / 'libggml-cuda.so.0').unlink()
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            bundle.build()

    def test_missing_server_rejected_with_preparation_command(self):
        (self.source / 'llama-server').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'make model-runtime'):
            bundle.build()

    def test_invalid_gguf_rejected(self):
        (self.source / bundle.MODEL).write_bytes(b'HTML error')
        with self.assertRaisesRegex(ValueError, 'GGUF'):
            bundle.build()
