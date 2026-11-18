"""Memory-bounded finite sampling that spills decoded segments to temporary video files."""

from __future__ import annotations

import copy
import gc
import shutil
import time
import uuid
import weakref
from fractions import Fraction
from pathlib import Path

import torch

import folder_paths
from comfy_api.latest import InputImpl, Types, io
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_audio import vae_decode_audio

from .drift_control_av import drift_control_step_count
from .experimental_latent_guide import _valid_guide_frames
from .minimax_h3_finite_segments import (
    H3_FPS,
    FiniteLoopState,
    FiniteSegmentPlan,
    _finite_locked_audio_asset,
    _finite_plan_for_segment,
    _finite_video_audio_muted,
    _locked_audio_interval,
    _require_finite_plan,
    _selflift_settings,
    _silent_audio_interval,
)


SPILL_ROOT = "minimax_h3_segment_spill"


def _spill_root() -> Path:
    root = Path(folder_paths.get_temp_directory()).resolve() / SPILL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 24 * 60 * 60
    for path in root.iterdir():
        if path.is_dir() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
    return root


def _inside_spill(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(_spill_root())
    except ValueError as error:
        raise ValueError("Segment spill path is outside ComfyUI's temporary directory") from error
    return resolved


def _trim_audio_head(audio: dict, frames: int) -> dict:
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    sample_rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
    if waveform is None or sample_rate <= 0:
        raise ValueError("Decoded segment audio must contain waveform and sample_rate")
    count = round(frames / H3_FPS * sample_rate)
    if count and waveform.shape[-1] <= count:
        raise ValueError("Decoded segment audio is shorter than its overlap")
    result = dict(audio)
    result["waveform"] = waveform[..., count:].clone() if count else waveform
    return result


def _trim_audio_tail(audio: dict, frames: int) -> dict:
    waveform = audio["waveform"]
    count = round(frames / H3_FPS * int(audio["sample_rate"]))
    if count and waveform.shape[-1] <= count:
        raise ValueError("Accumulated audio is shorter than its overlap")
    result = dict(audio)
    result["waveform"] = waveform[..., :-count].clone() if count else waveform
    return result


def _concat_audio(left: dict | None, right: dict) -> dict:
    if left is None:
        return right
    if int(left["sample_rate"]) != int(right["sample_rate"]):
        raise ValueError("Decoded segment audio sample rates do not match")
    return {
        "waveform": torch.cat((left["waveform"], right["waveform"]), dim=-1),
        "sample_rate": int(right["sample_rate"]),
    }


class MiniMaxH3SegmentVideoInitialize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentVideoInitialize",
            display_name="MiniMax H3 Initialize Temporary Segment Video",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                FiniteSegmentPlan.Input("finite_plan"),
                io.Boolean.Input("continue_audio_latent", default=True),
            ],
            outputs=[FiniteLoopState.Output()],
        )

    @classmethod
    def execute(cls, finite_plan, continue_audio_latent):
        finite = _require_finite_plan(finite_plan)
        directory = _spill_root() / uuid.uuid4().hex
        directory.mkdir()
        return io.NodeOutput({
            "finite_plan": finite,
            "next_iteration": 0,
            "previous_latent": None,
            "previous_images": None,
            "video_files": [],
            "merged_audio": None,
            "written_frames": 0,
            "spill_directory": str(directory),
            "continue_audio_latent": bool(continue_audio_latent),
        })


class MiniMaxH3SegmentVideoArchive(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentVideoArchive",
            display_name="MiniMax H3 Archive Temporary Segment Video",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                FiniteSegmentPlan.Input("finite_plan"),
                FiniteLoopState.Input("state"),
                io.Latent.Input("sampled_latent"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("iteration_index", force_input=True),
            ],
            outputs=[FiniteLoopState.Output()],
        )

    @classmethod
    def execute(cls, finite_plan, state, sampled_latent, vae, audio_vae, iteration_index):
        finite = _require_finite_plan(finite_plan)
        index = int(iteration_index)
        if int(state.get("next_iteration", -1)) != index:
            raise ValueError("Temporary segment-video state is out of sequence")
        latent = sampled_latent["samples"]
        video_latent = latent.unbind()[0] if getattr(latent, "is_nested", False) else latent
        images = vae.decode(video_latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        locked = _finite_locked_audio_asset(finite)
        muted = _finite_video_audio_muted(finite)
        audio = None if locked is not None or muted else vae_decode_audio(audio_vae, sampled_latent)
        overlaps = finite.get("segment_overlaps", [finite.get("overlap_frames", 0)] * int(finite["segment_count"]))
        overlap = int(overlaps[index]) if index else 0
        trim_frames = _valid_guide_frames(overlap) if overlap else 0
        if int(images.shape[0]) <= trim_frames:
            raise ValueError(f"Segment {index + 1} is shorter than its opening overlap")
        current = images[trim_frames:]
        target = int(finite.get("target_output_frames") or 0)
        written = int(state.get("written_frames") or 0)
        if target and written + int(current.shape[0]) > target:
            current = current[:target - written]
        directory = _inside_spill(state["spill_directory"])
        path = directory / f"segment_{index + 1:03d}.mp4"
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=current, frame_rate=Fraction(H3_FPS)),
            bit_depth=8,
            color_space="sRGB",
        )
        video.save_to(path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=18.0)

        merged_audio = state.get("merged_audio")
        if locked is None and not muted:
            soft = bool(state.get("continue_audio_latent"))
            current_audio = audio if soft else _trim_audio_head(audio, trim_frames)
            if merged_audio is not None and soft and trim_frames:
                merged_audio = _trim_audio_tail(merged_audio, trim_frames)
            merged_audio = _concat_audio(merged_audio, current_audio)

        files = list(state.get("video_files", []))
        files.append(str(path))
        right_overlap = int(overlaps[index + 1]) if index + 1 < len(overlaps) else 0
        next_state = dict(state)
        next_state.update(
            next_iteration=index + 1,
            previous_latent=sampled_latent,
            previous_images=images[-1:].clone() if right_overlap == 1 else None,
            video_files=files,
            merged_audio=merged_audio,
            written_frames=written + int(current.shape[0]),
        )
        del current, images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return io.NodeOutput(next_state)


class MiniMaxH3SegmentVideoOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentVideoOutput",
            display_name="MiniMax H3 Finish Temporary Segment Video",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[FiniteLoopState.Input("state")],
            outputs=[
                io.Latent.Output(display_name="Last Sampled Latent"),
                io.Video.Output(display_name="Final Video"),
                io.Audio.Output(display_name="Final Audio"),
                io.String.Output(display_name="Status"),
            ],
        )

    @classmethod
    def execute(cls, state):
        finite = _require_finite_plan(state.get("finite_plan"))
        if int(state.get("next_iteration", 0)) != int(finite["segment_count"]):
            raise ValueError("Not every segment was archived before final video assembly")
        directory = _inside_spill(state["spill_directory"])
        locked = _finite_locked_audio_asset(finite)
        if locked is not None:
            first = _finite_plan_for_segment(finite, 1)
            plan = copy.deepcopy(first)
            if locked.get("lockKind") != "timeline_video_audio":
                plan["timeline"]["audios"] = [locked]
            audio = _locked_audio_interval(plan, output_frames=int(finite.get("target_output_frames") or 0))
        elif _finite_video_audio_muted(finite):
            audio = _silent_audio_interval(
                _finite_plan_for_segment(finite, 1),
                output_frames=int(finite.get("target_output_frames") or state.get("written_frames") or 0),
            )
        else:
            audio = state.get("merged_audio")
            if audio is None:
                raise ValueError("Generated segment audio is missing")
            target = int(finite.get("target_output_frames") or 0)
            wanted = round(target / H3_FPS * int(audio["sample_rate"])) if target else 0
            if wanted and audio["waveform"].shape[-1] > wanted:
                audio = dict(audio)
                audio["waveform"] = audio["waveform"][..., :wanted].clone()
        sources = [InputImpl.VideoFromFile(str(_inside_spill(path))) for path in state["video_files"]]
        final_path = directory / "assembled.mp4"
        InputImpl.VideoFromList(sources, audio, Types.VideoCodec.H264).save_to(
            final_path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.AUTO,
        )
        for path in state["video_files"]:
            Path(path).unlink(missing_ok=True)
        result = InputImpl.VideoFromFile(str(final_path))
        try:
            weakref.finalize(result, shutil.rmtree, directory, True)
        except TypeError:
            pass
        status = (
            f"Decoded and encoded {finite['segment_count']} segments one at a time, then joined them without "
            "holding the full RGB timeline in memory. No latent checkpoint was written."
        )
        return io.NodeOutput(state.get("previous_latent"), result, audio, status)


class MiniMaxH3FiniteSegmentVideoSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FiniteSegmentVideoSampler",
            display_name="MiniMax H3 有限分段采样(低内存版本)",
            category="MiniMax H3/Long Video",
            description=(
                "Decode and encode one segment at a time in ComfyUI's temporary folder, then join the encoded "
                "segments as VIDEO. It does not save latent checkpoints or provide segment reruns."
            ),
            enable_expand=True,
            inputs=[
                io.Model.Input("model"), io.Clip.Input("clip"), io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"), FiniteSegmentPlan.Input("finite_plan"),
                io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Boolean.Input("continue_audio_latent", default=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
            ],
            outputs=[
                io.Latent.Output(display_name="Last Sampled Latent"),
                io.Video.Output(display_name="Final Video"),
                io.Audio.Output(display_name="Final Audio"),
                io.String.Output(display_name="Status"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, finite_plan, sampler, sigmas, seed, continue_audio_latent, ref_image_size="match"):
        finite = _require_finite_plan(finite_plan)
        graph = GraphBuilder()
        initialized = graph.node(
            "MiniMaxH3SegmentVideoInitialize", id="segment_video_initialize",
            finite_plan=finite, continue_audio_latent=bool(continue_audio_latent),
        )
        state = initialized.out(0)
        steps = drift_control_step_count(sigmas)
        if steps < 1:
            raise ValueError("Sampling requires a sigma schedule with at least one step")
        lift = _selflift_settings(
            steps, finite.get("second_pass_model", ""), finite.get("second_pass_high_steps"),
        ) if finite.get("second_pass") else None
        for index, prompt in enumerate(finite["prompts"]):
            number = index + 1
            segment_plan = _finite_plan_for_segment(finite, number)
            encoder = graph.node(
                "MiniMaxH3TimelineEncoder", id=f"video_encode_{number}", clip=clip,
                vae=vae, audio_vae=audio_vae, plan=segment_plan, prompt=prompt,
                ref_image_size=ref_image_size,
            )
            prepared = graph.node(
                "MiniMaxH3FiniteLoopPrepare", id=f"video_prepare_{number}", finite_plan=finite,
                loop_state=state, segment_plan=segment_plan, positive=encoder.out(0),
                target_latent=encoder.out(1), model=model, sigmas=sigmas, vae=vae,
                audio_vae=audio_vae, iteration_index=index,
                continue_audio_latent=bool(continue_audio_latent),
            )
            if lift is not None:
                sampled = graph.node(
                    "MiniMaxH3TimelineSelfLiftSampler", id=f"video_sample_{number}",
                    model=prepared.out(2), positive=prepared.out(0), negative=prepared.out(0),
                    vae=vae, latent_image=prepared.out(1), sampler=sampler, sigmas=sigmas,
                    seed=int(seed), cfg=1.0, **lift,
                )
            else:
                noise = graph.node("RandomNoise", id=f"video_noise_{number}", noise_seed=int(seed))
                guider = graph.node(
                    "BasicGuider", id=f"video_guider_{number}",
                    model=prepared.out(2), conditioning=prepared.out(0),
                )
                sampled = graph.node(
                    "SamplerCustomAdvanced", id=f"video_sample_{number}", noise=noise.out(0),
                    guider=guider.out(0), sampler=sampler, sigmas=sigmas,
                    latent_image=prepared.out(1),
                )
            archived = graph.node(
                "MiniMaxH3SegmentVideoArchive", id=f"video_archive_{number}", finite_plan=finite,
                state=state, sampled_latent=sampled.out(0), vae=vae, audio_vae=audio_vae,
                iteration_index=index,
            )
            state = archived.out(0)
        output = graph.node("MiniMaxH3SegmentVideoOutput", id="segment_video_output", state=state)
        return io.NodeOutput(
            output.out(0), output.out(1), output.out(2), output.out(3), expand=graph.finalize(),
        )
