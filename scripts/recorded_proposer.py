"""Replay cached SAM proposals without loading a model."""
from scripts.temporal_proposal_tracker import native_boxes

class RecordedProposer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def propose(self, grid):
        self.calls += 1
        return dict(boxes=native_boxes(self.result['boxes']), seconds=0,
                    recorded_detector_seconds=self.result['seconds'], model='recorded_sam_vit_b')
