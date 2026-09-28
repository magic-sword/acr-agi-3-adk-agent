"""Offline asset corruption must stop submission startup, before model execution."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.submission_runtime import configure_sam
from scripts.build_notebook import build


class SubmissionAssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / 'input/datasets/owner/sam/1'
        self.bundle.mkdir(parents=True)
        self.source = self.root / 'extracted'
        self.sources = {'segment_anything/__init__.py': '# test fixture\n', 'LICENSE': 'license\n'}
        (self.bundle / 'sam-source.json').write_text(json.dumps(self.sources))
        (self.bundle / 'sam_vit_b_01ec64.pth').write_bytes(b'fixture checkpoint')
        self.manifest = dict(schema=1, files={name: hashlib.sha256((self.bundle / name).read_bytes()).hexdigest()
                                            for name in ['sam-source.json', 'sam_vit_b_01ec64.pth']},
                             source_files={name: hashlib.sha256(data.encode()).hexdigest()
                                           for name, data in self.sources.items()})
        (self.bundle / 'sam-bundle.json').write_text(json.dumps(self.manifest))

    def configure(self):
        return configure_sam(self.manifest, self.root / 'input', self.source)

    def test_kaggle_nested_assets_override_local_paths_without_loading_model(self):
        with patch.dict(os.environ, {'COGNITION_PROPOSALS': 'program',
                                     'COGNITION_SAM_SOURCE': '/old/local/path'}):
            self.assertEqual(self.configure(), self.bundle)
            self.assertEqual(os.environ['COGNITION_PROPOSALS'], 'sam_initial')
            self.assertEqual(os.environ['COGNITION_SAM_CHECKPOINT'], str(self.bundle / 'sam_vit_b_01ec64.pth'))
            self.assertEqual(os.environ['COGNITION_SAM_SOURCE'], str(self.source))
        self.assertEqual((self.source / 'segment_anything/__init__.py').read_text(), self.sources['segment_anything/__init__.py'])

    def test_missing_dataset_stops_instead_of_falling_back(self):
        (self.bundle / 'sam-bundle.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Attach'):
            self.configure()

    def test_corrupted_checkpoint_stops_before_source_extraction(self):
        (self.bundle / 'sam_vit_b_01ec64.pth').write_bytes(b'truncated')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.configure()
        self.assertFalse(self.source.exists())

    def test_corrupted_source_stops_before_executing_code(self):
        (self.bundle / 'sam-source.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.configure()

    def test_different_manifest_is_not_selected(self):
        changed = dict(self.manifest, schema=2)
        (self.bundle / 'sam-bundle.json').write_text(json.dumps(changed))
        with self.assertRaises(FileNotFoundError):
            self.configure()

    def test_missing_sam_attachment_rejected_by_notebook_builder(self):
        from scripts.build_notebook import METADATA
        metadata = json.loads(METADATA.read_text())
        metadata['dataset_sources'] = ['magicsword001/arc-agi-3-qwen3-vl-4b']
        path = self.root / 'kernel-metadata.json'
        path.write_text(json.dumps(metadata))
        with patch('scripts.build_notebook.METADATA', path):
            with self.assertRaisesRegex(ValueError, 'sam-vit-b'):
                build()

    def test_save_run_preflight_failure_blocks_submission_file_and_stops_qwen(self):
        from unittest.mock import Mock
        cell = build()['cells'][-1]['source']
        process = Mock()
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'is_dir', return_value=True), \
             patch('subprocess.run', side_effect=RuntimeError('GPU failure')), \
             patch('pathlib.Path.write_text', side_effect=AssertionError('must not write')):
            with self.assertRaisesRegex(RuntimeError, 'GPU failure'):
                exec(cell, {'model_process': process, 'sys': __import__('sys')})
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
