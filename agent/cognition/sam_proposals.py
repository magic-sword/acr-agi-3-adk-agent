"""Lazy, offline SAM ViT-B proposals; shared weights, never game semantics."""
from functools import lru_cache
from pathlib import Path
import os
import sys
import threading
import time
from .region_masks import native_mask


class SamProposer:
    def __init__(self, checkpoint, source, device='cuda'):
        self.checkpoint = Path(checkpoint)
        self.source = Path(source)
        self.device = device
        self.generator = None
        self.lock = threading.Lock()

    def _load(self):
        if self.generator is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'SAM checkpoint missing: {self.checkpoint}')
        import torch
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('SAM requires CUDA; program perception remains available')
        # The VLM is a separate service. Bound local SAM's CPU preprocessing.
        torch.set_num_threads(4)
        if self.source.is_dir() and str(self.source) not in sys.path:
            sys.path.insert(0, str(self.source))
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        model = sam_model_registry['vit_b'](checkpoint=str(self.checkpoint)).to(self.device).eval()
        self.generator = SamAutomaticMaskGenerator(
            model, points_per_side=32, points_per_batch=32,
            pred_iou_thresh=.88, stability_score_thresh=.95,
            box_nms_thresh=.7, crop_n_layers=0, min_mask_region_area=0,
            output_mode='binary_mask')

    def propose(self, grid):
        """Return up to 128 native, half-open boxes, sorted by mask quality.

        The entire call, including cold model load, is charged to the runtime.
        No download, crop recursion, semantic category, or gold labels are used.
        """
        start = time.perf_counter()
        with self.lock:
            self._load()
            import numpy as np
            import torch
            from PIL import Image
            from agent.rendering import frame_image
            h, w = len(grid), len(grid[0])
            image = np.asarray(frame_image(grid).resize((w*6, h*6), Image.Resampling.NEAREST))
            with torch.inference_mode():
                anns = self.generator.generate(image)
            ranked = sorted(enumerate(anns), key=lambda p: (
                -float(p[1]['predicted_iou']), -float(p[1]['stability_score']), p[0]))
            boxes, masks = [], []
            for _, ann in ranked[:128]:
                yy, xx = ann['segmentation'].nonzero()
                if len(xx):
                    boxes.append([int(xx.min())//6, int(yy.min())//6,
                                  (int(xx.max())+6)//6, (int(yy.max())+6)//6])
                    masks.append(native_mask(ann['segmentation'], h, w))
            if self.device.startswith('cuda'):
                torch.cuda.synchronize(self.device)
        return dict(boxes=boxes, masks=masks, mask_encoding='native_row_runs_majority_6x6', seconds=time.perf_counter()-start,
                    generated=len(anns), model='sam_vit_b', device=self.device)


@lru_cache(maxsize=1)
def _shared(checkpoint, source, device):
    return SamProposer(checkpoint, source, device)


def default_proposer():
    root = Path(__file__).resolve().parents[2] / 'outputs' / 'sam-deps'
    return _shared(os.getenv('COGNITION_SAM_CHECKPOINT', str(root/'sam_vit_b_01ec64.pth')),
                   os.getenv('COGNITION_SAM_SOURCE', str(root/'segment-anything')),
                   os.getenv('COGNITION_SAM_DEVICE', 'cuda'))
