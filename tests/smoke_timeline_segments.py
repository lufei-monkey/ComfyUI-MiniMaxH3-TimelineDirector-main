"""Variable windows, per-seam overlap and zero-overlap boundary regression."""
import copy
from pathlib import Path
import sys
import json
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_finite_segments import _load_package, _plan

finite = _load_package(Path(__file__).resolve().parents[1])

single=_plan()
single['segment_count']=0
single['timeline']['globalPrompt']='A material-free text-to-video prompt'
single['timeline']['videoClips']=[]
single['timeline']['images']=[]
single['timeline']['audios']=[]
single['timeline']['segmentConfig']={'count':0,'segments':[]}
single_plan=finite._require_finite_plan(single)
assert single_plan['mode']=='single_segment'
assert single_plan['prompts']==['A material-free text-to-video prompt']
assert single_plan['segment_lengths']==[192] and single_plan['segment_overlaps']==[0]
single_result=finite.MiniMaxH3FiniteSegmentSampler.execute(
    model='model',clip='clip',vae='vae',audio_vae='audio_vae',finite_plan=single,
    sampler='sampler',sigmas=torch.tensor([1.,.5,0.]),seed=7,continue_audio_latent=True,
)
single_nodes=list(single_result.expand.values())
single_encoders=[n for n in single_nodes if n['class_type']=='MiniMaxH3TimelineEncoder']
assert len(single_encoders)==1
assert single_encoders[0]['inputs']['prompt']=='A material-free text-to-video prompt'
assert single_encoders[0]['inputs']['plan']['timeline']['images']==[]

source = _plan()
source['timeline']['segmentConfig'] = {'mode':'timeline','count':3,'segments':[
    {'startFrame':0,'endFrame':56,'prompt':'First','images':['p2','p1'],'audios':['a1']},
    {'startFrame':34,'endFrame':107,'prompt':'Second','images':['p1'],'audios':[]},
    {'startFrame':107,'endFrame':146,'prompt':'Third','images':[],'audios':['a2']},
]}
p = finite._require_finite_plan(source)
assert p['segment_lengths'] == [56,73,39]
assert p['segment_overlaps'] == [0,22,0]
assert p['target_output_frames'] == 146
assert [a['id'] for a in p['segment_plans'][0]['timeline']['images']] == ['p2','p1']
assert p['segment_plans'][1]['timeline']['selection']['start'] == 34/24
assert p['prompts'] == ['First','Second','Third']

# Global prompt is reused only when every segment prompt is empty. As soon as
# one local prompt exists, all local prompts are required and the global value
# is deliberately ignored.
global_only=copy.deepcopy(source)
global_only['timeline']['globalPrompt']='One shared prompt'
for segment in global_only['timeline']['segmentConfig']['segments']:segment['prompt']=''
assert finite._require_finite_plan(global_only)['prompts']==['One shared prompt']*3
local_override=copy.deepcopy(global_only)
for index,segment in enumerate(local_override['timeline']['segmentConfig']['segments']):segment['prompt']=f'Local {index+1}'
assert finite._require_finite_plan(local_override)['prompts']==['Local 1','Local 2','Local 3']
partial=copy.deepcopy(global_only);partial['timeline']['segmentConfig']['segments'][0]['prompt']='Only one'
try:finite._require_finite_plan(partial)
except ValueError as error:assert 'every segment' in str(error)
else:raise AssertionError('partial segment prompt mode accepted')
director=sys.modules[finite.__package__+'.minimax_h3_timeline_director']
source['timeline']['segmentConfig']['activeIndex']=1
with patch.object(director,'_create_prompt_media_bundle',side_effect=lambda plan:plan):
    selected,bundle,complete=director.MiniMaxH3TimelinePlanner.execute(
        640,352,8,json.dumps(source['timeline']),
    )
    assert selected['prompt_index']==2 and bundle['prompt_index']==2
    assert selected['length']==73 and selected['generation_seconds']==73/24
    assert [a['id'] for a in selected['timeline']['images']]==['p1']
    assert complete['prompt_index'] is None
    assert finite._require_finite_plan(complete)['segment_lengths']==[56,73,39]
# Source clips stay on their global clock and expose only this window's intersection.
source['timeline']['videoClips']=[{'id':'v','file':'clip.mp4','start':1.,'trimStart':4.,'duration':10.,'hasAudio':False,'referenceMode':'edit'}]
video_plan=finite._require_finite_plan(source)['segment_plans'][1]
spec=director._video_reference_specs(video_plan['timeline'])[0]
assert abs(spec['source_start']-(4+34/24-1))<1e-6
assert abs(spec['duration']-73/24)<1e-6
for field,value in [('startFrame',108),('prompt',''),('endFrame',145)]:
    bad=copy.deepcopy(source);bad['timeline']['segmentConfig']['segments'][2][field]=value
    try:finite._require_finite_plan(bad)
    except ValueError:pass
    else:raise AssertionError(f'Invalid {field} accepted')

result=finite.MiniMaxH3FiniteSegmentSampler.execute(
    model='model',clip='clip',vae='vae',audio_vae='audio_vae',finite_plan=source,
    sampler='sampler',sigmas=torch.tensor([1.,.5,0.]),seed=7,continue_audio_latent=True,
)
graph=result.expand
nodes=list(graph.values())
continuations=[n['inputs'] for n in nodes if n['class_type']=='MiniMaxH3FiniteLatentContinuation']
assert [n['overlap_frames'] for n in continuations]==[0,22,0]
assert 'previous_images' not in continuations[2] and 'previous_latent' not in continuations[2]
assert len([n for n in nodes if n['class_type']=='MiniMaxH3FiniteAudioTrimTail'])==1
assert [n['inputs']['plan']['length'] for n in nodes if n['class_type']=='MiniMaxH3TimelineEncoder']==[56,73,39]
images=torch.zeros(39,2,2,3);audio={'sample_rate':24000,'waveform':torch.ones(1,1,39000)}
out=finite.MiniMaxH3FiniteSegmentFinalize.execute({},images,2,0,True,audio)
assert out[1].shape[0]==39 and out[2]['waveform'].shape[-1]==39000
previous=torch.rand(73,2,2,3)
with patch.object(finite,'_apply_h3_guides',return_value='anchored') as guide:
    out=finite.MiniMaxH3FiniteLatentContinuation.execute('positive',{},2,0,True,'model',None,{},previous,'vae','audio_vae')
    assert out[0]=='positive' and out[2]==0 and out[3]=='model'
    guide.assert_not_called()

# A two-stage result carries its native low-resolution prediction as metadata.
# The continuation node must forward it into the next target latent without
# changing the existing high-resolution AV-tail and Drift-Control setup.
low_carry=torch.randn(1,24,9,4,6)
fake_previous={'samples':'high','selflift_low_resolution_carry':low_carry}
fake_masked={'samples':'masked','noise_mask':'mask'}
with patch.object(
    finite,'_apply_linear_temporal_noise_mask',
    return_value=(fake_masked,{'frames':22,'video_tokens':7}),
), patch.object(finite,'install_drift_control_av_model',return_value='patched'):
    carried=finite.MiniMaxH3FiniteLatentContinuation.execute(
        'positive',{'samples':'target'},1,22,True,'model','sigmas',fake_previous,
    )
assert carried[1]['selflift_previous_low_resolution_carry'] is low_carry
assert carried[1]['samples']=='masked' and carried[3]=='patched'

# The planner switch replaces every internal segment sampler with SelfLift;
# users do not add a second sampler node to the workflow manually.
second_pass=copy.deepcopy(source)
second_pass['timeline']['secondPass']=True
second_pass['timeline']['secondPassModel']='minimax_h3_latent_upscaler_3d_bf16.safetensors'
second_pass['timeline']['secondPassHighSteps']=5
with patch.object(
    finite.folder_paths, 'get_filename_list',
    return_value=['minimax_h3_latent_upscaler_3d_bf16.safetensors'],
):
    second_pass_result=finite.MiniMaxH3FiniteSegmentSampler.execute(
        model='model',clip='clip',vae='vae',audio_vae='audio_vae',finite_plan=second_pass,
        sampler='sampler',sigmas=torch.tensor([1.,.85,.7,.55,.4,.25,.15,.08,0.]),
        seed=7,continue_audio_latent=True,
    )
second_pass_nodes=list(second_pass_result.expand.values())
selflift=[n for n in second_pass_nodes if n['class_type']=='MiniMaxH3TimelineSelfLiftSampler']
assert len(selflift)==3
assert all(n['inputs']['transition_step']==3 for n in selflift)
assert all(n['inputs']['upscaler_model']=='minimax_h3_latent_upscaler_3d_bf16.safetensors' for n in selflift)
assert not [n for n in second_pass_nodes if n['class_type']=='SamplerCustomAdvanced']
assert 'Two-stage sampling ran on every segment' in second_pass_result[3]
assert '3 low-resolution step(s), 5 full-resolution step(s)' in second_pass_result[3]
for invalid_high_steps in (0, 8, 9):
    try:finite._selflift_settings(8,'minimax_h3_latent_upscaler_3d_bf16.safetensors',invalid_high_steps)
    except ValueError as error:assert 'lower than the Basic Scheduler step count' in str(error) or invalid_high_steps==0
    else:raise AssertionError(f'Invalid high-resolution step count accepted: {invalid_high_steps}')
print('timeline segment planning and execution graph: PASS')
