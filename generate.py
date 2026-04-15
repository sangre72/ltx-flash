#!/usr/bin/env python3
"""
ltx-flash: LTX-2.3 DiT SSD Block Streaming

Flash-MoE의 SSD 오프로딩 기법을 LTX-2.3 비디오 디퓨전 모델에 적용.
Transformer Block을 SSD에서 하나씩 on-demand 로드 → 연산 → 해제.

기존: transformer 전체(10.5GB) → RAM (~38GB)
ltx-flash: 블록 1개씩 (~225MB) → SSD 스트리밍 → RAM ~2GB

사용법:
    python generate.py generate -p "A cat on a windowsill" -o out.mp4
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


def cmd_generate(args):
    """SSD 스트리밍으로 영상 생성."""
    import mlx.core as mx
    from ltx_flash.ssd_stream import SSDStreamingLTXModel

    model_dir = Path(args.model) if args.model else MODEL_DIR
    if not model_dir.exists():
        print(f"모델 경로 없음: {model_dir}")
        print("모델 다운로드:")
        print("  huggingface-cli download dgrauet/ltx-2.3-mlx-q4 --local-dir ~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4")
        sys.exit(1)

    print("ltx-flash: SSD 스트리밍 모드")
    print(f"모델: {model_dir}")
    t0 = time.perf_counter()
    ssd_model = SSDStreamingLTXModel(model_dir=model_dir, variant="distilled", verbose=not args.quiet)
    print(f"초기화: {time.perf_counter()-t0:.1f}초  Metal: {mx.get_active_memory()/1024**3:.2f} GB\n")

    _run_pipeline(ssd_model, args, model_dir)


def _run_pipeline(ssd_model, args, model_dir):
    import mlx.core as mx
    import mlx.nn as nn
    import ltx_pipelines_mlx.ti2vid_one_stage as pipe_module
    from ltx_core_mlx.model.transformer.model import LTXModel
    from ltx_flash.ssd_stream import SSDStreamingLTXModel

    class SSDWrappedLTXModel(LTXModel):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self._ssd = ssd_model
            self.config = ssd_model.config
            self._shell = ssd_model._model_shell

        def load_weights(self, weights):
            pass  # 이미 SSD 스트리머가 처리

        def __call__(self, *args, **kwargs):
            return ssd_model(*args, **kwargs)

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

    try:
        from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
        pipe = TextToVideoPipeline(model_dir=str(model_dir))

        if not args.quiet:
            print(f"프롬프트: {args.prompt}")
            print(f"해상도: {args.width}×{args.height}, {args.frames}프레임, {args.steps}steps\n")

        t0 = time.perf_counter()
        pipe.generate_and_save(
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_steps=args.steps,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - t0

        import os
        size = os.path.getsize(args.output)
        print(f"\n저장: {args.output}  ({size/1024:.0f} KB)")
        print(f"시간: {elapsed:.1f}초  |  RAM: {mx.get_active_memory()/1024**3:.2f} GB")
    finally:
        pipe_module.LTXModel = original


def main():
    parser = argparse.ArgumentParser(
        prog="ltx-flash",
        description="LTX-2.3 DiT SSD Block Streaming — Flash-MoE 기법을 비디오 디퓨전에 적용",
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    p = sub.add_parser("generate", help="SSD 스트리밍으로 영상 생성")
    p.add_argument("--prompt", "-p", required=True, help="생성 프롬프트")
    p.add_argument("--output", "-o", default="output.mp4", help="출력 파일")
    p.add_argument("--model", "-m", default=None, help="모델 경로 (기본: ~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4)")
    p.add_argument("--height", "-H", type=int, default=480)
    p.add_argument("--width",  "-W", type=int, default=704)
    p.add_argument("--frames", "-f", type=int, default=49, help="프레임 수 (8k+1, 49=2초)")
    p.add_argument("--steps",  "-s", type=int, default=8)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--quiet", "-q",  action="store_true")
    p.set_defaults(func=cmd_generate)

    # benchmark
    p = sub.add_parser("benchmark", help="SSD I/O 속도 측정")
    p.set_defaults(func=cmd_benchmark)

    # stream-test
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
