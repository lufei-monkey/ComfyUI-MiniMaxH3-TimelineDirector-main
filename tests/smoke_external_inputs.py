"""Run in ComfyUI's Python environment, with ComfyUI on PYTHONPATH."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
from smoke_finite_segments import _load_package, _plan
finite = _load_package(Path(__file__).resolve().parents[1])
director = sys.modules[finite.__package__ + '.minimax_h3_timeline_director']
source = _plan()['timeline']
source['segmentConfig']['mode'] = 'timeline'
for i, segment in enumerate(source['segmentConfig']['segments']):
    segment.update(startFrame=i*24, endFrame=(i+1)*24)
with patch.object(director, '_create_prompt_media_bundle', side_effect=lambda plan: plan):
    selected, bundle, complete = director.MiniMaxH3TimelinePlanner.execute(
        640, 352, 3, json.dumps(source), prompt_index=2,
        segment_prompts='["first", "second", "third"]', global_prompt='shared')
assert selected['timeline']['segmentConfig']['segments'][1]['prompt'] == 'second'
assert finite._require_finite_plan(complete)['prompts'] == ['first', 'second', 'third']
assert 'globalPrompt' not in source
try:
    director._external_timeline_inputs(source, segment_prompts='["only one"]')
except ValueError:
    pass
else:
    raise AssertionError('Prompt count mismatch accepted')
print('External prompt routing checks passed')
