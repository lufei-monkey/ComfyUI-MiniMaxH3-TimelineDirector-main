"""Dependency-free checks for external segment geometry and schema."""
import importlib.util
from pathlib import Path
import copy
spec=importlib.util.spec_from_file_location('external',Path(__file__).resolve().parents[1]/'external_input_nodes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
align=lambda seconds:5+17*max(0,round((seconds*24-5)/17))
timeline={'segmentConfig':{'count':3,'mode':'timeline','segments':[
 {'startFrame':0,'endFrame':158,'images':['old']},
 {'startFrame':136,'endFrame':294},
 {'startFrame':255,'endFrame':413}]}}
original=copy.deepcopy(timeline)
values={f'segment_{i}_segment_frames':158 for i in range(1,4)}
values.update(segment_1_overlap_frames=39,segment_2_overlap_frames=22,segment_3_overlap_frames=39,segment_2_duration=1.)
bundle={'type':m.SEGMENT_TYPE,'values':values}
result=m.apply_segments(timeline,bundle,5,None,align)
segments=result['segmentConfig']['segments']
assert [x['endFrame']-x['startFrame'] for x in segments]==[158]*3
assert segments[-1]['endFrame']==413
assert segments[0]['startFrame']==0
assert original==timeline
assert m.apply_segments(timeline,None,5,None,align) is timeline
# Unconnected overlap retains the original timeline geometry.
assert m.apply_segments(timeline,{'type':m.SEGMENT_TYPE,'values':{'segment_2_segment_frames':158}},5,None,align)==timeline
for invalid in [2,158,-1,22.0]:
 bad=copy.deepcopy(bundle);bad['values']['segment_2_overlap_frames']=invalid
 try:m.apply_segments(timeline,bad,5,None,align)
 except ValueError:pass
 else:raise AssertionError(invalid)
assert len(m.MiniMaxH3ExternalSegments.INPUT_TYPES()['optional'])==64*16
assert len(m.MiniMaxH3ExternalMedia.INPUT_TYPES()['optional'])==19
print('PASS: 413-frame example, precedence, first overlap, fallback, validation, schema, no mutation')
