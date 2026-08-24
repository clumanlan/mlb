from models.hit_predictor.processing.schema import PBP

# Walk as a completed PA outcome (play_result values). Intent Walk is a
# distinct play_result but the same true outcome as an unintentional walk
# for BB-prop purposes — the batter reaches on 4 balls either way.
WALKS = {"Walk", "Intent Walk"}

__all__ = ['PBP', 'WALKS']
