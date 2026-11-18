"""Optional graph aggregators; tensors are materialized only by the planner."""
import copy
import math

MEDIA_TYPE = 'MINIMAX_H3_EXTERNAL_MEDIA'
SEGMENT_TYPE = 'MINIMAX_H3_EXTERNAL_SEGMENTS'

class MiniMaxH3ExternalMedia:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        for prefix, count, kind in [('image',9,'IMAGE'),('ref_video',3,'IMAGE'),('ref_audio',3,'AUDIO'),('ref_video_audio',3,'AUDIO')]:
            inputs.update({f'{prefix}_{i}': (kind,) for i in range(count)})
        inputs['global_prompt'] = ('STRING', {'forceInput': True})
        return {'required': {}, 'optional': inputs}
    RETURN_TYPES = (MEDIA_TYPE,)
    FUNCTION = 'collect'
    CATEGORY = 'model/conditioning/minimax'
    def collect(self, **values):
        return ({'type': MEDIA_TYPE, 'values': values},)

class MiniMaxH3ExternalSegments:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        for segment in range(1,65):
            for i in range(1,10):
                inputs[f'segment_{segment}_image_{i}'] = ('IMAGE',)
            for i in range(1,4):
                inputs[f'segment_{segment}_audio_{i}'] = ('AUDIO',)
            for field, kind in [('segment_frames','INT'),('segment_prompt','STRING'),('overlap_frames','INT'),('duration','FLOAT')]:
                inputs[f'segment_{segment}_{field}'] = (kind, {'forceInput': True})
        return {'required': {}, 'optional': inputs}
    RETURN_TYPES = (SEGMENT_TYPE,)
    FUNCTION = 'collect'
    CATEGORY = 'model/conditioning/minimax'
    def collect(self, **values):
        return ({'type': SEGMENT_TYPE, 'values': values},)

def apply_segments(timeline, bundle, seconds, adapter, align):
    if bundle is None:
        return timeline
    if not isinstance(bundle, dict) or bundle.get('type') != SEGMENT_TYPE:
        raise ValueError('segment_inputs requires MiniMax H3 External Segments')
    values = bundle['values']
    used = [i for i in range(1,65) if any(k.startswith(f'segment_{i}_') and v is not None for k,v in values.items())]
    if not used:
        return timeline
    result = copy.deepcopy(timeline)
    config = result['segmentConfig']
    segments = config.setdefault('segments', [])
    count = max(int(config.get('count',1)), max(used))
    if not 1 <= count <= 64:
        raise ValueError('Segment count must be 1..64')
    while len(segments) < count:
        segments.append({})
    previous_end = previous_length = 0
    previous_original_end = 0
    for number, segment in enumerate(segments[:count],1):
        prefix = f'segment_{number}_'
        original_start = int(segment.get('startFrame', previous_original_end))
        original_end = int(segment.get('endFrame', original_start + align(seconds)))
        frames = values.get(prefix+'segment_frames')
        duration = values.get(prefix+'duration')
        if frames is not None:
            if type(frames) is not int:
                raise ValueError(f'Segment {number}: frames must be an integer')
            length = frames
        elif duration is not None:
            if not math.isfinite(duration) or duration < 5/24 or duration > 150:
                raise ValueError(f'Segment {number}: duration must be 5/24..150 seconds')
            length = align(duration)
        else:
            length = original_end-original_start
        if length < 5 or length > 3592 or (length-5)%17:
            raise ValueError(f'Segment {number}: frames must be 5 + 17*n (up to 3592)')
        overlap = 0 if number == 1 else values.get(prefix+'overlap_frames', previous_original_end-original_start)
        if type(overlap) is not int or overlap < 0 or (number > 1 and overlap >= min(length, previous_length)) or (overlap not in (0,1) and (overlap < 5 or (overlap-5)%17)):
            raise ValueError(f'Segment {number}: invalid H3 overlap or overlap exceeds adjacent lengths')
        start = previous_end-overlap
        segment.update(startFrame=start, endFrame=start+length)
        if values.get(prefix+'segment_prompt') is not None:
            segment['prompt'] = values[prefix+'segment_prompt']
        for kind, argument, limit in [('images','images',9),('audios','audio',3)]:
            singular = 'image' if kind == 'images' else 'audio'
            supplied = [(i,values.get(prefix+f'{singular}_{i}')) for i in range(1,limit+1)]
            supplied = [(i,v) for i,v in supplied if v is not None]
            if supplied:
                ids = []
                for i,value in supplied:
                    if kind == 'images' and value.shape[0] != 1:
                        raise ValueError('Each segment image socket accepts exactly one image')
                    scratch = {'segmentConfig': {'count':1,'segments':[{}]}}
                    saved = adapter(scratch, **{argument:value})
                    for asset in saved[kind]:
                        asset['id'] = f'external-segment-{number}-{singular}-{i}'
                        result.setdefault(kind,[]).append(asset)
                        ids.append(asset['id'])
                segment[kind] = ids
        previous_end, previous_length = start+length, length
        previous_original_end = original_end
    config.update(count=count, mode='timeline', segments=segments[:count])
    return result
