#!/usr/bin/env python3
"""
ltx-flash: LTX-2.3 DiT SSD Block Streaming

Flash-MoE의 SSD 오프로딩 기법을 LTX-2.3 비디오 디퓨전 모델에 적용.
Transformer Block을 SSD에서 하나씩 on-demand 로드 → 연산 → 해제.

기존: transformer 전체(10.5GB) → RAM (~38GB)
ltx-flash: 블록 1개씩 (~225MB) → SSD 스트리밍 → RAM ~2GB

사용법:
    # T2V: 텍스트 → 영상
    python generate.py generate -p "A cat on a windowsill" -o out.mp4

    # I2V: 이미지 + 텍스트 → 영상
    python generate.py generate -p "The cat starts to move" --image cat.jpg -o out.mp4

    # FLF2V: 첫 프레임 + 마지막 프레임 → 영상
    python generate.py generate -p "Transition" --first-frame first.jpg --last-frame last.jpg -o out.mp4

    python generate.py benchmark
    python generate.py stream-test
"""

import argparse
import sys
import time
from pathlib import Path

MODEL_DIR = Path.home() / "git/ltx-2-mlx/models/ltx-2.3-mlx-q4"


def cmd_benchmark(args):
    """SSD 블록 스트리밍 속도 벤치마크."""
    import mlx.core as mx
    from ltx_flash.ssd_stream import SSDBlockLoader, BlockIndex
    from ltx_core_mlx.model.transformer.model import LTXModelConfig

    sft_path = MODEL_DIR / "transformer-distilled.safetensors"
    config = LTXModelConfig()

    print("ltx-flash — SSD Block Streaming Benchmark")
    print(f"파일: {sft_path.name}  ({sft_path.stat().st_size/1024**3:.1f} GB)")
    print(f"블록: {config.num_layers}개, 블록당 ~{sft_path.stat().st_size/config.num_layers/1024**2:.0f} MB")
    print()

    loader = SSDBlockLoader(sft_path, prefetch=True)
    idx = loader._index

    print(f"텐서 구조: 블록당 {len(idx.block_keys(0))}개, 비블록 {len(idx.non_block_keys())}개")
    print()
    print(f"{'블록':>5}  {'I/O (ms)':>10}  {'MB/s':>8}")
    print("-" * 32)

    import os
    block_mb = sft_path.stat().st_size / config.num_layers / 1024**2

    for i in range(min(8, config.num_layers)):
        t0 = time.perf_counter()
        weights = loader.load_block_weights(i)
        for v in list(weights.values())[:5]:
            mx.eval(v)
        t = (time.perf_counter() - t0) * 1000
        print(f"{i:>5}  {t:>10.1f}  {block_mb/(t/1000):>8.0f}")
        del weights
        mx.clear_cache()

    loader.close()
    print()
    print(f"예상 step당 SSD I/O: 48 × ~50ms = ~2.4초  (cold)")
    print(f"예상 step당 SSD I/O: 48 × ~5ms  = ~0.2초  (OS page cache warm)")


def cmd_stream_test(args):
    """블록 스트리밍 + 메모리 사용량 측정."""
    import mlx.core as mx
    from ltx_flash.ssd_stream import SSDBlockLoader
    from ltx_core_mlx.model.transformer.model import LTXModelConfig

    sft_path = MODEL_DIR / "transformer-distilled.safetensors"
    config = LTXModelConfig()
    n = args.num_blocks

    print(f"블록 스트리밍 테스트 ({n}개, prefetch={'on' if args.prefetch else 'off'})")
    print(f"{'블록':>5}  {'SSD (ms)':>9}  {'Metal (MB)':>11}")
    print("-" * 32)

    loader = SSDBlockLoader(sft_path, prefetch=args.prefetch)
    times = []

    for block, weights, t_load in loader.stream_blocks(config, num_blocks=n):
        for t in list(weights.values())[:10]:
            mx.eval(t)
        mem = mx.get_active_memory() / 1024**2
        print(f"{len(times):>5}  {t_load*1000:>9.0f}  {mem:>11.0f}")
        times.append(t_load)

    loader.close()
    avg = sum(times) / len(times) * 1000
    print()
    print(f"평균 블록 로딩: {avg:.0f}ms")
    print(f"48블록 예상: {avg*48/1000:.1f}초/step")


def _detect_mode(args) -> str:
    """인자로 생성 모드 자동 감지.

    - flf2v : --first-frame + --last-frame
    - ti2v  : --image 여러 장 + --image-index 지정 (다중 이미지 키프레임)
    - i2v   : --image 1장
    - t2v   : 이미지 없음
    """
    images = getattr(args, "image", None) or []
    has_first = bool(getattr(args, "first_frame", None))
    has_last = bool(getattr(args, "last_frame", None))

    if has_first and has_last:
        return "flf2v"
    if len(images) > 1:
        return "ti2v"
    if len(images) == 1:
        return "i2v"
    return "t2v"


def _build_ssd_model(args, model_dir):
    """SSDStreamingLTXModel 초기화."""
    import mlx.core as mx
    from ltx_flash.ssd_stream import SSDStreamingLTXModel

    prefetch_depth = getattr(args, "prefetch_depth", 2)
    ssd_model = SSDStreamingLTXModel(
        model_dir=model_dir,
        variant="distilled",
        prefetch_depth=prefetch_depth,
        verbose=not args.quiet,
    )
    print(f"초기화: Metal: {mx.get_active_memory()/1024**3:.2f} GB\n")
    return ssd_model


def _patch_pipeline(pipe_module, ssd_model):
    """LTXModel을 SSDWrappedLTXModel로 패치."""
    import mlx.nn as nn
    from ltx_core_mlx.model.transformer.model import LTXModel

    class SSDWrappedLTXModel(LTXModel):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self._ssd = ssd_model
            self.config = ssd_model.config
            self._shell = ssd_model._model_shell

        def load_weights(self, weights):
            pass

        def __call__(self, *a, **kw):
            return ssd_model(*a, **kw)

        def __getattr__(self, name):
            if name.startswith("_") or name == "config":
                raise AttributeError(name)
            try:
                return getattr(self._shell, name)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    original = pipe_module.LTXModel

    class Factory:
        def __new__(cls, *a, **kw):
            return SSDWrappedLTXModel()

    pipe_module.LTXModel = Factory
    return original


def cmd_generate(args):
    """SSD 스트리밍으로 영상 생성 (T2V / I2V / FLF2V)."""
    import mlx.core as mx
    import ltx_pipelines_mlx.ti2vid_one_stage as pipe_module
    from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
    from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT

    model_dir = Path(args.model) if args.model else MODEL_DIR
    if not model_dir.exists():
        print(f"모델 경로 없음: {model_dir}")
        print("  huggingface-cli download dgrauet/ltx-2.3-mlx-q4 --local-dir ~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4")
        sys.exit(1)

    mode = _detect_mode(args)
    cfg_scale = getattr(args, "cfg_scale", 1.0)
    negative_prompt = getattr(args, "negative_prompt", None) or DEFAULT_NEGATIVE_PROMPT

    print("ltx-flash: SSD 스트리밍 모드")
    print(f"모델: {model_dir}")
    print(f"모드: {mode.upper()}")

    ssd_model = _build_ssd_model(args, model_dir)

    # TextToVideoPipeline에 커스텀 메서드 추가
    TextToVideoPipeline._encode_text_with_negative_custom = _encode_text_with_negative_custom

    # LTXModel → SSDWrappedLTXModel 패치
    original_ltxmodel = _patch_pipeline(pipe_module, ssd_model)

    try:
        if mode == "t2v":
            _run_t2v(args, model_dir, cfg_scale, negative_prompt)
        elif mode == "i2v":
            _run_i2v(args, model_dir, cfg_scale, negative_prompt)
        elif mode == "ti2v":
            _run_ti2v(args, model_dir, cfg_scale, negative_prompt)
        elif mode == "flf2v":
            _run_flf2v(args, model_dir, cfg_scale, negative_prompt)
    finally:
        pipe_module.LTXModel = original_ltxmodel


# ──────────────────────────────────────────────
# T2V: 텍스트 → 영상
# ──────────────────────────────────────────────

def _run_t2v(args, model_dir, cfg_scale, negative_prompt):
    import mlx.core as mx
    import os
    from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
    from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
    from ltx_core_mlx.model.transformer.model import X0Model
    from ltx_core_mlx.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
    from ltx_core_mlx.conditioning.types.latent_cond import create_initial_state
    from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
    from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
    from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

    if not args.quiet:
        print(f"프롬프트: {args.prompt}")
        if cfg_scale > 1.0:
            print(f"네거티브: {negative_prompt[:60]}...")
            print(f"CFG scale: {cfg_scale}")
        print(f"해상도: {args.width}×{args.height}, {args.frames}프레임, {args.steps}steps\n")

    pipe = TextToVideoPipeline(model_dir=str(model_dir))
    t0 = time.perf_counter()

    if cfg_scale <= 1.0:
        pipe.generate_and_save(
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_steps=args.steps,
            seed=args.seed,
        )
    else:
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = (
            pipe._encode_text_with_negative_custom(args.prompt, negative_prompt)
        )
        pipe.load()

        F, H, W = compute_video_latent_shape(args.frames, args.height, args.width)
        audio_T = compute_audio_token_count(args.frames)

        video_state = create_initial_state((1, F * H * W, 128), args.seed,
                                           positions=compute_video_positions(F, H, W))
        audio_state = create_initial_state((1, audio_T, 128), args.seed + 1,
                                           positions=compute_audio_positions(audio_T))

        sigmas = DISTILLED_SIGMAS[:args.steps + 1] if args.steps else DISTILLED_SIGMAS
        video_guider = MultiModalGuiderFactory.constant(
            MultiModalGuiderParams(cfg_scale=cfg_scale, stg_scale=0.0),
            negative_context=neg_video_embeds,
        )
        audio_guider = MultiModalGuiderFactory.constant(
            MultiModalGuiderParams(cfg_scale=cfg_scale, stg_scale=0.0),
            negative_context=neg_audio_embeds,
        )

        output = guided_denoise_loop(
            model=X0Model(pipe.dit),
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_guider,
            audio_guider_factory=audio_guider,
            sigmas=sigmas,
        )

        video_latent = pipe.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = pipe.audio_patchifier.unpatchify(output.audio_latent)

        if pipe.low_memory:
            from ltx_core_mlx.utils.memory import aggressive_cleanup
            pipe.dit = None
            pipe.text_encoder = None
            pipe.feature_extractor = None
            pipe._loaded = False
            aggressive_cleanup()

        pipe._decode_and_save_video(video_latent, audio_latent, args.output)

    _print_result(args.output, t0)


# ──────────────────────────────────────────────
# I2V: 이미지 1장 + 텍스트 → 영상
# ──────────────────────────────────────────────

def _run_i2v(args, model_dir, cfg_scale, negative_prompt):
    from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

    image_path = args.image[0]
    if not Path(image_path).exists():
        print(f"이미지 없음: {image_path}")
        sys.exit(1)

    if not args.quiet:
        print(f"프롬프트: {args.prompt}")
        print(f"이미지: {image_path}")
        if cfg_scale > 1.0:
            print(f"네거티브: {negative_prompt[:60]}...")
            print(f"CFG scale: {cfg_scale}")
        print(f"해상도: {args.width}×{args.height}, {args.frames}프레임, {args.steps}steps\n")

    pipe = TextToVideoPipeline(model_dir=str(model_dir))
    t0 = time.perf_counter()

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        image=image_path,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_steps=args.steps,
        seed=args.seed,
    )

    _print_result(args.output, t0)


# ──────────────────────────────────────────────
# TI2V: 이미지 여러 장 + 텍스트 → 영상 (키프레임 보간)
# ──────────────────────────────────────────────

def _run_ti2v(args, model_dir, cfg_scale, negative_prompt):
    """Text + Multi-Image to Video: KeyframeInterpolationPipeline 사용."""
    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    images = args.image  # list[str]
    indices = getattr(args, "image_index", None) or []

    # 인덱스 미지정 시 자동 균등 배분
    if not indices:
        n = len(images)
        if n == 1:
            indices = [0]
        else:
            step = (args.frames - 1) / (n - 1)
            indices = [round(i * step) for i in range(n)]

    if len(indices) != len(images):
        print(f"--image {len(images)}장과 --image-index {len(indices)}개 수가 맞지 않습니다.")
        sys.exit(1)

    for p in images:
        if not Path(p).exists():
            print(f"이미지 없음: {p}")
            sys.exit(1)

    if not args.quiet:
        print(f"프롬프트: {args.prompt}")
        for img, idx in zip(images, indices):
            print(f"  프레임 {idx:3d}: {img}")
        if cfg_scale > 1.0:
            print(f"CFG scale: {cfg_scale}")
        print(f"해상도: {args.width}×{args.height}, {args.frames}프레임, steps={args.steps}\n")

    pipe = KeyframeInterpolationPipeline(model_dir=str(model_dir))
    t0 = time.perf_counter()

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        keyframe_images=images,
        keyframe_indices=indices,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        fps=getattr(args, "fps", 24.0),
        seed=args.seed,
        stage1_steps=args.steps,
        cfg_scale=cfg_scale,
    )

    _print_result(args.output, t0)


# ──────────────────────────────────────────────
# FLF2V: 첫 프레임 + 마지막 프레임 → 영상
# ──────────────────────────────────────────────

def _run_flf2v(args, model_dir, cfg_scale, negative_prompt):
    """First-Last-Frame-to-Video: KeyframeInterpolationPipeline 사용."""
    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    first_path = args.first_frame
    last_path = args.last_frame
    for p in [first_path, last_path]:
        if not Path(p).exists():
            print(f"이미지 없음: {p}")
            sys.exit(1)

    if not args.quiet:
        print(f"프롬프트: {args.prompt}")
        print(f"첫 프레임 (index=0): {first_path}")
        print(f"마지막 프레임 (index={args.frames-1}): {last_path}")
        if cfg_scale > 1.0:
            print(f"CFG scale: {cfg_scale}")
        print(f"해상도: {args.width}×{args.height}, {args.frames}프레임, steps={args.steps}\n")

    pipe = KeyframeInterpolationPipeline(model_dir=str(model_dir))
    t0 = time.perf_counter()

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        keyframe_images=[first_path, last_path],
        keyframe_indices=[0, args.frames - 1],
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        fps=getattr(args, "fps", 24.0),
        seed=args.seed,
        stage1_steps=args.steps,
        cfg_scale=cfg_scale,
    )

    _print_result(args.output, t0)


# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def _print_result(output_path: str, t0: float):
    import mlx.core as mx
    import os
    elapsed = time.perf_counter() - t0
    size = os.path.getsize(output_path)
    print(f"\n저장: {output_path}  ({size/1024:.0f} KB)")
    print(f"시간: {elapsed:.1f}초  |  RAM: {mx.get_active_memory()/1024**3:.2f} GB")


def _encode_text_with_negative_custom(self, prompt: str, negative_prompt: str):
    """positive + 커스텀 negative 프롬프트 인코딩."""
    self._load_text_encoder()
    import mlx.core as mx
    video_embeds, audio_embeds = self._encode_text(prompt)
    neg_video_embeds, neg_audio_embeds = self._encode_text(negative_prompt)
    mx.eval(video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds)
    self.text_encoder = None
    self.feature_extractor = None
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    aggressive_cleanup()
    return video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds


def main():
    parser = argparse.ArgumentParser(
        prog="ltx-flash",
        description="LTX-2.3 DiT SSD Block Streaming — T2V / I2V / FLF2V 지원",
    )
    sub = parser.add_subparsers(dest="command")

    # ── generate (T2V / I2V / FLF2V 통합) ──────────────────────
    p = sub.add_parser("generate", help="영상 생성 (T2V / I2V / FLF2V)")

    # 공통
    p.add_argument("--prompt", "-p", required=True, help="생성 프롬프트")
    p.add_argument("--output", "-o", default="output.mp4", help="출력 파일")
    p.add_argument("--model", "-m", default=None, help="모델 경로")
    p.add_argument("--height", "-H", type=int, default=480, help="영상 높이 (px)")
    p.add_argument("--width",  "-W", type=int, default=704, help="영상 너비 (px)")
    p.add_argument("--frames", "-f", type=int, default=49, help="프레임 수 (8k+1, 49=2초@24fps)")
    p.add_argument("--steps",  "-s", type=int, default=8, help="디노이징 스텝 (기본:8, 품질↑→20~30)")
    p.add_argument("--fps",          type=float, default=24.0, help="출력 FPS (FLF2V용, 기본:24)")
    p.add_argument("--prefetch-depth", type=int, default=2, help="N-ahead prefetch 블록 수 (기본:2)")
    p.add_argument("--cfg-scale",    type=float, default=1.0, help="CFG scale (1.0=없음, 3.0~7.0=품질↑, 속도 2배)")
    p.add_argument("--negative-prompt", "-n", type=str, default=None, help="네거티브 프롬프트")
    p.add_argument("--seed",         type=int, default=42, help="랜덤 시드")
    p.add_argument("--quiet", "-q",  action="store_true", help="로그 숨김")

    # I2V / TI2V: --image는 여러 번 지정 가능
    p.add_argument("--image", "-i", type=str, action="append", default=[],
                   metavar="PATH",
                   help="[I2V/TI2V] 이미지 경로 (여러 번 지정 가능). "
                        "1장=I2V(첫 프레임 고정), 여러 장=TI2V(키프레임 보간)")
    p.add_argument("--image-index", type=int, action="append", default=[],
                   metavar="FRAME",
                   help="[TI2V] 각 --image가 고정될 프레임 번호 (미지정 시 균등 자동 배분). "
                        "예: --image a.jpg --image b.jpg --image-index 0 --image-index 48")

    # FLF2V: 첫/마지막 프레임 단축 옵션
    p.add_argument("--first-frame", type=str, default=None,
                   help="[FLF2V] 첫 번째 프레임 이미지 경로")
    p.add_argument("--last-frame",  type=str, default=None,
                   help="[FLF2V] 마지막 프레임 이미지 경로")

    p.set_defaults(func=cmd_generate)

    # ── benchmark ───────────────────────────────────────────────
    p = sub.add_parser("benchmark", help="SSD I/O 속도 측정")
    p.set_defaults(func=cmd_benchmark)

    # ── stream-test ─────────────────────────────────────────────
    p = sub.add_parser("stream-test", help="블록 스트리밍 + 메모리 테스트")
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--prefetch", action="store_true", default=True)
    p.set_defaults(func=cmd_stream_test)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
