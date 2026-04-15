# ltx-flash

LTX-2.3 DiT SSD Block Streaming — Flash-MoE 기법을 비디오 디퓨전 모델에 적용한 Apple Silicon 메모리 최적화 라이브러리.

## 프로젝트 개요

기존 LTX-2.3 모델은 Transformer 전체 가중치(~10.5GB)를 RAM에 로드해 ~38GB RAM이 필요하다.
ltx-flash는 Transformer Block 1개씩(~225MB)을 SSD에서 on-demand pread()로 로드 → GPU 연산 → 즉시 해제하여 RAM을 ~3GB 수준으로 줄인다.

**메모리 구조:**
- 비블록 가중치(adaln, patchify_proj 등): ~0.8GB RAM 상주
- 현재 처리 중인 블록 가중치: ~225MB (연산 후 해제)
- 활성화(activation) 버퍼: ~2GB
- 총 RAM: ~3GB (기존 38GB 대비 92% 절약)

## 기술 스택

- **Python** ≥ 3.11
- **MLX** ≥ 0.30.0 (Apple Silicon Metal GPU 프레임워크)
- **safetensors** ≥ 0.4 (가중치 파일 포맷)
- **numpy** ≥ 1.24
- **서브모듈**: `ltx-2-mlx` (ltx_core_mlx, ltx_pipelines_mlx 패키지 포함)

## 파일 구조

```
ltx-flash/
├── ltx_flash/
│   ├── __init__.py          # 패키지 진입점, 공개 API
│   └── ssd_stream.py        # 핵심 구현: BlockIndex, SSDBlockLoader, SSDStreamingLTXModel
├── generate.py              # CLI 진입점 (benchmark / stream-test / generate 서브커맨드)
├── pyproject.toml
└── ltx-2-mlx/              # git submodule (ltx_core_mlx, ltx_pipelines_mlx)
    └── models/ltx-2.3-mlx-q4/
        └── transformer-distilled.safetensors  # 10.5GB, 48블록
```

## 핵심 클래스

### `BlockIndex` (`ssd_stream.py:52`)
safetensors 헤더를 파싱해 각 텐서의 바이트 오프셋을 인덱싱한다.
- `block_keys(block_idx)`: 특정 블록의 텐서 키 목록
- `non_block_keys()`: 비블록 가중치 키 목록
- `open_fd()` / `close_fd()`: 파일 디스크립터 관리

### `SSDBlockLoader` (`ssd_stream.py:163`)
블록 단위 SSD 스트리머. `_pread_tensor()`로 seek 없이 직접 오프셋 읽기.
- `load_block_weights(block_idx)`: 한 블록 텐서를 ThreadPoolExecutor로 병렬 pread()
- `stream_blocks(config, num_blocks)`: ping-pong prefetch 제너레이터
- `num_io_threads=8`, `prefetch=True`가 기본값

### `SSDStreamingLTXModel` (`ssd_stream.py:294`)
LTXModel 드롭인 대체. 비블록 가중치는 RAM에 상주하고 블록은 스트리밍.
- `forward_blocks(video_hidden, audio_hidden, **block_kwargs)`: 48블록 순차 스트리밍
- `__call__(...)`: `LTXModel.__call__`과 동일한 인터페이스

## CLI 사용법

```bash
# 영상 생성
ltx-flash generate -p "A cat on a windowsill" -o out.mp4

# SSD I/O 벤치마크
ltx-flash benchmark

# 블록 스트리밍 + 메모리 테스트
ltx-flash stream-test --num-blocks 4
```

## 개발 가이드

### 환경 설정

```bash
pip install -e .
# 서브모듈 초기화
git submodule update --init --recursive
pip install -e ltx-2-mlx/packages/ltx-core-mlx
```

### 모델 경로

기본 모델 경로: `~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4/`
`generate.py:22`의 `MODEL_DIR` 상수로 설정됨.

### 핵심 설계 원칙

1. **pread() 사용**: `seek()` 없이 오프셋 직접 지정 → 멀티스레드 안전, OS page cache 활용
2. **ping-pong buffering**: 블록 N GPU 연산 중 블록 N+1 SSD I/O prefetch
3. **즉시 해제**: `del block; mx.clear_cache()` — Metal GPU 메모리도 명시적으로 해제
4. **BF16 처리**: numpy에 bfloat16 없으므로 uint16로 읽어 `mx.array().view(mx.bfloat16)` 변환

### 성능 기준치

| 상태 | 블록당 SSD I/O | step당 시간 |
|------|--------------|------------|
| Cold (캐시 없음) | ~50ms | ~2.4초 |
| Warm (OS page cache) | ~5ms | ~0.2초 |
| NVMe 이론값 (~7GB/s) | ~32ms | ~1.5초 |

### 주의사항

- `mx.eval()`로 GPU 연산 완료를 명시적으로 대기한 후 블록을 해제해야 함
- `stream_blocks()`는 제너레이터 — 한 번에 블록 1개만 메모리에 존재
- BF16 텐서는 `np.uint16` → `mx.bfloat16` 경로로 처리 (numpy BF16 미지원)
- `SSDStreamingLTXModel`은 `LTXModel` 서브클래스가 아닌 드롭인 대체 패턴 사용
