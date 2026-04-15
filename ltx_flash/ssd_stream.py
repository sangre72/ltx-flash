"""
SSD Block Streaming for LTX-2.3 DiT.

Flash-MoE 원리를 DiT Transformer Block에 적용:
  - MoE: SSD에서 활성 expert K개를 pread()로 on-demand 로드
  - LTX:  SSD에서 Transformer Block 1개를 on-demand 로드 → 연산 → 해제

구조:
  transformer-distilled.safetensors (10.5GB)
    transformer.transformer_blocks.0.*   (154 tensors, ~225MB)
    transformer.transformer_blocks.1.*
    ...
    transformer.transformer_blocks.47.*
  + 비블록 가중치 (58 tensors, ~0.8GB) → RAM 상주

실행 시 메모리:
  - RAM 상주: 비블록 가중치 ~0.8GB + 활성화(activation) ~2GB
  - 스트리밍: 블록 1개 ~225MB (연산 후 즉시 해제)
  - 총 RAM: ~3GB (기존 38GB 대비 92% 절약)
  - SSD I/O: 블록 로드 ~225MB × 8 steps × 48 blocks = ~86GB/step

성능 예상:
  - NVMe 읽기: ~7GB/s → 블록당 ~32ms
  - GPU 연산: 블록당 ~10ms (bf16 행렬곱)
  - 병렬화: 다음 블록 prefetch 중 현재 블록 GPU 연산 (ping-pong)
  - 총 step당: ~48 × max(32, 10)ms = ~1.5초/step
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core_mlx.model.transformer.model import LTXModelConfig


def _strip_prefix(key: str, prefix: str = "transformer.") -> str:
    return key[len(prefix):] if key.startswith(prefix) else key


def _block_prefix(block_idx: int) -> str:
    return f"transformer.transformer_blocks.{block_idx}."


class BlockIndex:
    """
    safetensors 파일에서 각 블록 텐서의 바이트 오프셋 인덱스.

    safetensors 포맷:
      [헤더 크기: u64][JSON 헤더][텐서 데이터...]
      헤더에 각 텐서의 (dtype, shape, data_offsets) 포함

    pread()로 특정 텐서만 직접 읽기 위해 오프셋을 미리 인덱싱.
    """

    def __init__(self, sft_path: str | Path):
        self.path = Path(sft_path)
        self._offsets: dict[str, tuple[int, int, str, tuple]] = {}  # key → (start, end, dtype, shape)
        self._header_size: int = 0
        self._fd: int = -1
        self._build_index()

    def _build_index(self):
        import struct, json
        with open(self.path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_len).decode("utf-8")
            self._header_size = 8 + header_len
        header = json.loads(header_json)
        for key, info in header.items():
            if key == "__metadata__":
                continue
            dtype = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            # 실제 파일 오프셋 = header_size + data_start
            self._offsets[key] = (
                self._header_size + start,
                self._header_size + end,
                dtype,
                shape,
            )

    def block_keys(self, block_idx: int) -> list[str]:
        prefix = _block_prefix(block_idx)
        return [k for k in self._offsets if k.startswith(prefix)]

    def non_block_keys(self) -> list[str]:
        return [k for k in self._offsets if "transformer_blocks." not in k]

    def get_offset(self, key: str) -> tuple[int, int, str, tuple]:
        return self._offsets[key]

    def open_fd(self) -> int:
        if self._fd < 0:
            self._fd = os.open(str(self.path), os.O_RDONLY)
        return self._fd

    def close_fd(self):
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


# safetensors dtype 문자열 → MLX dtype 매핑
_DTYPE_MAP = {
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
    "I32": mx.int32,
    "U32": mx.uint32,
    "U8": mx.uint8,
    "I8": mx.int8,
}


def _pread_tensor(fd: int, start: int, end: int, dtype_str: str, shape: tuple) -> mx.array:
    """
    pread()로 SSD에서 텐서를 직접 읽어 mx.array로 변환.

    Flash-MoE expert loading과 동일한 원리.
    seek 없이 offset 지정 → 멀티스레드 안전, OS page cache 활용.
    """
    import numpy as np
    n_bytes = end - start
    buf = bytearray(n_bytes)
    view = memoryview(buf)
    total = 0
    while total < n_bytes:
        chunk = os.pread(fd, n_bytes - total, start + total)
        if not chunk:
            break
        view[total:total + len(chunk)] = chunk
        total += len(chunk)

    # dtype 변환
    dtype_str_upper = dtype_str.upper()
    if dtype_str_upper == "BF16":
        # numpy에 bfloat16 없음 → uint16로 읽어 MLX에서 해석
        arr = np.frombuffer(bytes(buf), dtype=np.uint16).reshape(shape)
        return mx.array(arr).view(mx.bfloat16)
    elif dtype_str_upper == "U32":
        arr = np.frombuffer(bytes(buf), dtype=np.uint32).reshape(shape)
        return mx.array(arr)
    elif dtype_str_upper == "F16":
        arr = np.frombuffer(bytes(buf), dtype=np.float16).reshape(shape)
        return mx.array(arr)
    elif dtype_str_upper == "F32":
        arr = np.frombuffer(bytes(buf), dtype=np.float32).reshape(shape)
        return mx.array(arr)
    else:
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(shape)
        return mx.array(arr)


class SSDBlockLoader:
    """
    LTX-2.3 Transformer Block SSD 스트리머.

    사용법:
        loader = SSDBlockLoader("models/ltx-2.3-mlx-q4/transformer-distilled.safetensors")
        for block, weights in loader.stream_blocks(config):
            output = block(hidden, **kwargs)
            # 블록은 자동으로 다음 iteration에서 새로 로드됨
            # Python GC가 이전 블록 weights 해제

    Ping-pong buffering:
        블록 N 연산 중 블록 N+1 prefetch (백그라운드 스레드)
        → GPU 연산과 SSD I/O 겹치기
    """

    def __init__(
        self,
        sft_path: str | Path,
        num_io_threads: int = 8,
        prefetch: bool = True,
    ):
        self.sft_path = Path(sft_path)
        self.prefetch = prefetch
        self._index = BlockIndex(sft_path)
        self._executor = ThreadPoolExecutor(max_workers=num_io_threads, thread_name_prefix="ssd-block")
        self._fd = self._index.open_fd()

    def load_block_weights(self, block_idx: int) -> dict[str, mx.array]:
        """블록 N의 모든 텐서를 병렬 pread()로 로드."""
        keys = self._index.block_keys(block_idx)
        prefix = _block_prefix(block_idx)

        # 각 텐서를 별도 스레드에서 pread()
        futures: dict[str, Future] = {}
        for key in keys:
            start, end, dtype, shape = self._index.get_offset(key)
            local_key = key[len(prefix):]  # 블록 prefix 제거
            futures[local_key] = self._executor.submit(
                _pread_tensor, self._fd, start, end, dtype, shape
            )

        return {k: f.result() for k, f in futures.items()}

    def load_non_block_weights(self) -> dict[str, mx.array]:
        """비블록 가중치(adaln, proj 등)를 로드 — RAM에 상주."""
        keys = self._index.non_block_keys()
        futures: dict[str, Future] = {}
        for key in keys:
            start, end, dtype, shape = self._index.get_offset(key)
            local_key = _strip_prefix(key)
            futures[local_key] = self._executor.submit(
                _pread_tensor, self._fd, start, end, dtype, shape
            )
        return {k: f.result() for k, f in futures.items()}

    def stream_blocks(
        self,
        config: LTXModelConfig,
        num_blocks: int | None = None,
    ) -> Iterator[tuple[BasicAVTransformerBlock, dict]]:
        """
        Transformer Block을 SSD에서 순차 스트리밍.

        Ping-pong buffering:
          - 현재 블록 연산 중 다음 블록 prefetch
          - GPU ↔ SSD I/O 겹치기

        Yields:
            (block_module, weights_dict) — 연산 후 다음 yield까지 weights가 RAM에 유지됨.
            다음 yield 시점에 이전 weights 참조 사라지면 GC가 해제.
        """
        n = num_blocks or config.num_layers

        # 첫 블록 로드 시작
        prefetch_future: Future | None = self._executor.submit(
            self.load_block_weights, 0
        )

        for block_idx in range(n):
            t_load_start = time.perf_counter()

            # 현재 블록 weights 가져오기
            weights = prefetch_future.result()
            t_load = time.perf_counter() - t_load_start

            # 다음 블록 prefetch (ping-pong)
            if block_idx + 1 < n:
                prefetch_future = self._executor.submit(
                    self.load_block_weights, block_idx + 1
                )
            else:
                prefetch_future = None

            # 블록 모듈 생성 + 가중치 주입
            from ltx_core_mlx.utils.weights import apply_quantization
            block = BasicAVTransformerBlock(
                video_dim=config.video_dim,
                audio_dim=config.audio_dim,
                video_num_heads=config.video_num_heads,
                audio_num_heads=config.audio_num_heads,
                video_head_dim=config.video_head_dim,
                audio_head_dim=config.audio_head_dim,
                av_cross_num_heads=config.av_cross_num_heads,
                av_cross_head_dim=config.av_cross_head_dim,
                ff_mult=config.ff_mult,
                norm_eps=config.norm_eps,
            )
            # int4 양자화 적용 후 가중치 로드
            apply_quantization(block, weights)
            block.load_weights(list(weights.items()))
            mx.eval(block.parameters())

            yield block, weights, t_load

            # 이 시점에서 block, weights 참조가 사라지면 GC가 메모리 해제
            # mx.clear_cache()를 명시적으로 호출해 Metal 메모리도 해제
            del block
            mx.clear_cache()

    def close(self):
        self._index.close_fd()
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SSDStreamingLTXModel:
    """
    SSD 스트리밍 LTX-2.3 추론 엔진.

    기존 LTXModel은 48블록을 전부 RAM에 로드 (~35GB).
    이 클래스는 블록을 하나씩 SSD에서 로드 → 연산 → 해제.

    RAM 사용량:
      - 비블록 가중치: ~0.8GB
      - 활성화(activation): ~2GB (video/audio hidden states)
      - 현재 블록 가중치: ~225MB
      - 총: ~3GB (기존 35GB 대비 91% 절약)
    """

    def __init__(
        self,
        model_dir: str | Path,
        variant: str = "distilled",  # "distilled" or "dev"
        config: LTXModelConfig | None = None,
        prefetch: bool = True,
        verbose: bool = True,
    ):
        from ltx_core_mlx.model.transformer.model import LTXModel, AdaLayerNormSingle
        from ltx_core_mlx.loader.sft_loader import SafetensorsStateDictLoader

        self.model_dir = Path(model_dir)
        self.config = config or LTXModelConfig()
        self.prefetch = prefetch
        self.verbose = verbose

        sft_name = f"transformer-{variant}.safetensors"
        self.sft_path = self.model_dir / sft_name
        if not self.sft_path.exists():
            raise FileNotFoundError(f"Transformer 가중치 없음: {self.sft_path}")

        # 비블록 가중치를 RAM에 로드 (adaln, patchify_proj, proj_out 등, ~0.8GB)
        if verbose:
            print(f"비블록 가중치 로드 중... (~0.8GB)", flush=True)
        t0 = time.perf_counter()

        self._loader = SSDBlockLoader(self.sft_path, prefetch=prefetch)
        non_block_weights = self._loader.load_non_block_weights()

        # 비블록 전용 LTXModel (transformer_blocks 없음)
        # 실제로는 상위 레벨 LTXModel에서 블록만 제거한 구조
        self._model_shell = LTXModel(self.config)
        # 블록 제거 (메모리 절약)
        self._model_shell.transformer_blocks = []

        # 비블록 가중치 주입
        # key에서 "transformer." prefix 제거
        filtered = {k: v for k, v in non_block_weights.items() if "transformer_blocks" not in k}
        # LTXModel의 키 형식에 맞게 변환
        shell_weights = {}
        for k, v in filtered.items():
            # "transformer.adaln_single.*" → "adaln_single.*"
            new_k = _strip_prefix(k, "transformer.")
            shell_weights[new_k] = v
        self._model_shell.load_weights(list(shell_weights.items()))
        mx.eval(self._model_shell.parameters())

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"비블록 가중치 로드 완료 ({elapsed:.1f}초, {len(non_block_weights)}개 텐서)")
            print(f"블록 가중치: SSD 스트리밍 모드 (블록당 ~225MB on-demand)")

    def forward_blocks(
        self,
        video_hidden: mx.array,
        audio_hidden: mx.array,
        **block_kwargs,
    ) -> tuple[mx.array, mx.array]:
        """
        48개 블록을 SSD에서 하나씩 스트리밍하며 순차 처리.

        block_kwargs: BasicAVTransformerBlock.__call__에 전달되는 나머지 인자
          (adaln_params, rope_freqs, attention_mask 등)

        각 블록:
          1. SSD에서 ~225MB pread() (병렬, ~30ms)
          2. GPU 연산 (~10ms)
          3. Metal 메모리 해제
        """
        block_times = []

        for block_idx, (block, weights, t_load) in enumerate(
            self._loader.stream_blocks(self.config)
        ):
            t_gpu_start = time.perf_counter()

            video_hidden, audio_hidden = block(
                video_hidden=video_hidden,
                audio_hidden=audio_hidden,
                block_idx=block_idx,
                **block_kwargs,
            )
            mx.eval(video_hidden, audio_hidden)  # GPU 연산 완료 대기

            t_gpu = time.perf_counter() - t_gpu_start
            block_times.append((block_idx, t_load * 1000, t_gpu * 1000))

            if self.verbose and (block_idx % 12 == 0 or block_idx == self.config.num_layers - 1):
                print(
                    f"  block {block_idx:2d}/{self.config.num_layers}: "
                    f"SSD={t_load*1000:.0f}ms GPU={t_gpu*1000:.0f}ms",
                    flush=True,
                )

        return video_hidden, audio_hidden, block_times

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        timestep: mx.array,
        video_text_embeds=None,
        audio_text_embeds=None,
        video_positions=None,
        audio_positions=None,
        video_attention_mask=None,
        audio_attention_mask=None,
        video_timesteps=None,
        audio_timesteps=None,
        perturbations=None,
    ) -> tuple[mx.array, mx.array]:
        """LTXModel.__call__과 동일한 인터페이스."""
        from ltx_core_mlx.model.transformer.model import LTXModel

        # 비블록 연산 (adaln, patchify 등)은 shell model이 처리
        # 블록 forward는 SSD 스트리밍으로 처리

        # cast
        video_latent = video_latent.astype(mx.bfloat16)
        audio_latent = audio_latent.astype(mx.bfloat16)
        if video_text_embeds is not None:
            video_text_embeds = video_text_embeds.astype(mx.bfloat16)
        if audio_text_embeds is not None:
            audio_text_embeds = audio_text_embeds.astype(mx.bfloat16)

        m = self._model_shell
        timestep = timestep.astype(mx.bfloat16)

        from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding
        t_emb = m._embed_timestep_scalar(timestep)
        av_ca_factor = m.config.av_ca_timestep_scale_multiplier / m.config.timestep_scale_multiplier
        t_emb_av_gate = get_timestep_embedding(
            timestep * m.config.timestep_scale_multiplier * av_ca_factor,
            m.config.timestep_embedding_dim,
        )

        video_hidden = m.patchify_proj(video_latent)
        audio_hidden = m.audio_patchify_proj(audio_latent)

        if video_timesteps is not None:
            vt_emb = m._embed_timestep_per_token(video_timesteps)
            video_adaln_emb, video_embedded_ts = m._adaln_per_token(m.adaln_single, vt_emb)
            av_ca_video_emb, _ = m._adaln_per_token(m.av_ca_video_scale_shift_adaln_single, vt_emb)
        else:
            video_adaln_emb, video_embedded_ts = m.adaln_single(t_emb)
            av_ca_video_emb, _ = m.av_ca_video_scale_shift_adaln_single(t_emb)

        av_ca_a2v_gate_emb, _ = m.av_ca_a2v_gate_adaln_single(t_emb_av_gate)
        video_prompt_emb, _ = m.prompt_adaln_single(t_emb)

        if audio_timesteps is not None:
            at_emb = m._embed_timestep_per_token(audio_timesteps)
            audio_adaln_emb, audio_embedded_ts = m._adaln_per_token(m.audio_adaln_single, at_emb)
            av_ca_audio_emb, _ = m._adaln_per_token(m.av_ca_audio_scale_shift_adaln_single, at_emb)
        else:
            audio_adaln_emb, audio_embedded_ts = m.audio_adaln_single(t_emb)
            av_ca_audio_emb, _ = m.av_ca_audio_scale_shift_adaln_single(t_emb)

        av_ca_v2a_gate_emb, _ = m.av_ca_v2a_gate_adaln_single(t_emb_av_gate)
        audio_prompt_emb, _ = m.audio_prompt_adaln_single(t_emb)

        # RoPE freqs
        video_rope_freqs = None
        audio_rope_freqs = None
        video_cross_rope_freqs = None
        audio_cross_rope_freqs = None
        if video_positions is not None:
            video_rope_freqs = m._compute_rope_freqs(
                video_positions, m.config.video_num_heads, m.config.video_head_dim
            )
            cross_pe_max_pos = max(
                m.config.positional_embedding_max_pos[0],
                m.config.audio_positional_embedding_max_pos[0],
            )
            video_cross_rope_freqs = m._compute_rope_freqs(
                video_positions[:, :, 0:1],
                m.config.av_cross_num_heads, m.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )
        if audio_positions is not None:
            audio_rope_freqs = m._compute_rope_freqs(
                audio_positions, m.config.audio_num_heads, m.config.audio_head_dim,
                max_pos_override=list(m.config.audio_positional_embedding_max_pos),
            )
            cross_pe_max_pos = max(
                m.config.positional_embedding_max_pos[0],
                m.config.audio_positional_embedding_max_pos[0],
            )
            audio_cross_rope_freqs = m._compute_rope_freqs(
                audio_positions[:, :, 0:1],
                m.config.av_cross_num_heads, m.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )

        mx.eval(
            video_hidden, audio_hidden,
            video_adaln_emb, audio_adaln_emb,
            video_prompt_emb, audio_prompt_emb,
            av_ca_video_emb, av_ca_audio_emb,
            av_ca_a2v_gate_emb, av_ca_v2a_gate_emb,
        )

        # ── SSD 스트리밍 블록 forward ──────────────────────────
        video_hidden, audio_hidden, block_times = self.forward_blocks(
            video_hidden=video_hidden,
            audio_hidden=audio_hidden,
            video_adaln_params=video_adaln_emb,
            audio_adaln_params=audio_adaln_emb,
            video_prompt_adaln_params=video_prompt_emb,
            audio_prompt_adaln_params=audio_prompt_emb,
            av_ca_video_params=av_ca_video_emb,
            av_ca_audio_params=av_ca_audio_emb,
            av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
            av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_rope_freqs=video_rope_freqs,
            audio_rope_freqs=audio_rope_freqs,
            video_cross_rope_freqs=video_cross_rope_freqs,
            audio_cross_rope_freqs=audio_cross_rope_freqs,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
            perturbations=perturbations,
        )

        video_out = m._output_block(video_hidden, video_embedded_ts, m.scale_shift_table, m.proj_out)
        audio_out = m._output_block(audio_hidden, audio_embedded_ts, m.audio_scale_shift_table, m.audio_proj_out)

        return video_out, audio_out

    def close(self):
        self._loader.close()
