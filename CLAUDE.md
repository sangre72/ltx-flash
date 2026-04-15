# ltx-flash

LTX-2.3 DiT SSD Block Streaming — Flash-MoE 기법을 비디오 디퓨전 모델에 적용한 Apple Silicon 메모리 최적화 라이브러리.

## 프로젝트 개요

기존 LTX-2.3 모델은 Transformer 전체 가중치(~10.5GB)를 RAM에 로드해 ~38GB RAM이 필요하다.
ltx-flash는 Transformer Block 1개씩(~225MB)을 SSD에서 on-demand mmap/pread()로 로드 → GPU 연산 → 즉시 해제하여 RAM을 ~3GB 수준으로 줄인다.

**메모리 구조:**
- 비블록 가중치(adaln, patchify_proj 등): ~0.8GB RAM 상주
- 현재 처리 중인 블록 가중치: ~225MB (연산 후 해제)
- 활성화(activation) 버퍼: ~2GB
- 총 RAM: ~3GB (기존 38GB 대비 92% 절약)

## 기술 스택

- **Python** ≥ 3.11
- **MLX** ≥ 0.31.0 (Apple Silicon Metal GPU 프레임워크)
- **safetensors** ≥ 0.4 (가중치 파일 포맷)
- **numpy** ≥ 1.26
- **uv** (패키지 매니저, 권장)
- **서브모듈**: `ltx-2-mlx` (ltx_core_mlx, ltx_pipelines_mlx 패키지 포함)

## 파일 구조

```
ltx-flash/
├── ltx_flash/
│   ├── __init__.py          # 패키지 진입점, 공개 API
│   └── ssd_stream.py        # 핵심 구현
├── generate.py              # CLI 진입점 (generate / benchmark / stream-test)
├── pyproject.toml
└── ltx-2-mlx/              # git submodule (ltx_core_mlx, ltx_pipelines_mlx)
```

**모델 경로 (기본값):**  
`~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4/`  
`generate.py:22`의 `MODEL_DIR` 상수로 설정됨.

## 핵심 클래스

### `BlockIndex` (`ssd_stream.py:55`)
safetensors 헤더를 파싱해 각 텐서의 바이트 오프셋을 인덱싱한다.
- `block_keys(block_idx)`: 특정 블록의 텐서 키 목록
- `non_block_keys()`: 비블록 가중치 키 목록
- `open_fd()` / `close_fd()`: 파일 디스크립터 관리
- `open_mmap()` / `close_fd()`: mmap 열기/닫기

### `SSDBlockLoader` (`ssd_stream.py:202`)
블록 단위 SSD 스트리머.
- `load_block_weights(block_idx)`: 한 블록 텐서를 ThreadPoolExecutor로 병렬 로드
- `stream_blocks(config, num_blocks)`: N-ahead prefetch 슬라이딩 윈도우 제너레이터
- 기본값: `num_io_threads=8`, `prefetch=True`, `prefetch_depth=2`, `use_mmap=True`
- **블록 객체 풀링**: `BasicAVTransformerBlock` 1개를 재사용, 가중치만 교체

### `BlockKVCache` (`ssd_stream.py:360`)
step 간 attention KV 캐시 (실험적). `BasicAVTransformerBlock.__call__`에 KV 주입 API가 없어 현재 미사용.

### `SSDStreamingLTXModel` (`ssd_stream.py:414`)
LTXModel 드롭인 대체. 비블록 가중치는 RAM에 상주하고 블록은 스트리밍.
- `forward_blocks(video_hidden, audio_hidden, **block_kwargs)`: 48블록 순차 스트리밍
- `__call__(...)`: `LTXModel.__call__`과 동일한 인터페이스

## 생성 모드 (generate.py)

`--image` / `--first-frame` / `--last-frame` 인자 조합으로 자동 감지:

| 모드 | 조건 | 파이프라인 |
|------|------|-----------|
| T2V | 이미지 없음 | `TextToVideoPipeline` |
| I2V | `--image` 1장 | `TextToVideoPipeline.generate_from_image()` |
| TI2V | `--image` 여러 장 | `KeyframeInterpolationPipeline` |
| FLF2V | `--first-frame` + `--last-frame` | `KeyframeInterpolationPipeline` |

## 개발 가이드

### 환경 설정

```bash
git clone --recurse-submodules https://github.com/sangre72/ltx-flash
cd ltx-flash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ltx-2-mlx/packages/ltx-core-mlx
uv pip install -e ltx-2-mlx/packages/ltx-pipelines-mlx
uv pip install -e .
```

### 모델 파일 관리 방식

**물리적 분리 없이 논리적 주소 접근** 방식을 사용한다.

- `transformer-distilled.safetensors` 1개 파일을 디스크에 그대로 유지 (분리 안 함)
- 시작 시 safetensors 헤더(JSON)를 파싱해 각 텐서의 byte offset을 `BlockIndex._offsets` dict에 인덱싱
- 블록 N이 필요하면 `"transformer.transformer_blocks.N.*"` 키의 offset으로 `mmap[offset:offset+size]` 슬라이스만 읽음
- 파일은 1개 10.5GB 그대로 — 읽는 위치만 달라짐

### 블록 처리 순서 — 항상 고정

Transformer는 입력이 블록 0 → 47까지 **순차적으로 통과**하는 고정 구조다.  
"어떤 블록이 필요한지" 동적으로 판단하는 로직이 없다 — 항상 0→47을 전부 순서대로 처리한다.

```
각 디노이징 step: 블록 0 → 1 → 2 → ... → 47 (48개 전부)
× N step 반복 (기본 30 step)
```

N-ahead prefetch가 가능한 이유: 다음 블록 번호가 항상 현재+1로 정해져 있으므로 예측 없이 미리 읽을 수 있다.

> **Dense vs MoE**: LTX-2.3은 Dense Transformer — 모든 파라미터가 모든 입력에 대해 항상 활성화된다. 가중치 값이 크든 작든 블록은 스킵되지 않으며, 출력값에 강하게/약하게 반영될 뿐 연산 자체는 동일하게 수행된다.  
> MoE(GPT-4, Mixtral 등)는 라우터가 입력마다 일부 expert만 선택해 나머지는 스킵하는 Sparse 구조. 비디오 DiT는 수천 개의 시공간 패치가 attention으로 전부 연결되어 있어 라우팅이 구조적으로 어렵기 때문에, 현재 오픈소스 비디오 생성 모델(LTX, Wan, CogVideoX 등)은 대부분 Dense다.  
> "Flash-MoE"는 SSD 오프로딩 기법만 차용한 것이고 실제 동적 라우팅은 없다.

### 핵심 설계 원칙

1. **mmap 우선**: OS page cache 직접 참조, bytearray 복사 없음 → warm 상태에서 극적으로 빠름
2. **N-ahead prefetch**: 슬라이딩 윈도우로 `prefetch_depth`개 블록 미리 읽기 (deque 기반)
3. **블록 객체 풀링**: `BasicAVTransformerBlock` 1개 생성 후 재사용, 가중치만 `load_weights()` 교체
4. **즉시 해제**: `del weights; mx.clear_cache()` — Metal GPU 메모리도 명시적으로 해제
5. **BF16 처리**: numpy에 bfloat16 없으므로 uint16로 읽어 `mx.array().view(mx.bfloat16)` 변환
6. **Metal 스레드 안전**: `mx.eval()`은 반드시 메인 스레드에서 호출 — IO 스레드에서 호출 시 Metal command buffer 충돌

### RAM 상주 방식과 성능 비교

| | RAM 전체 상주 | ltx-flash |
|---|---|---|
| 필요 RAM | ~38GB | ~3GB |
| M4 Max 36GB에서 실행 | ❌ OOM | ✅ |
| 첫 step | ~12초 | ~14초 (+2초) |
| 이후 step (warm) | ~12초 | ~13초 (거의 동일) |

- **warm 상태(2번째 step~)에서는 RAM 상주와 거의 동일한 속도** — GPU(~125ms/블록)가 압도적 병목이고, SSD I/O는 OS page cache로 ~0ms가 되어 숨겨지기 때문
- **cold 상태(첫 step)만 ~2초 손해** — SSD I/O ~45ms/블록이 추가되지만 이후 수렴
- **ltx-flash의 가치는 속도가 아니라 실행 가능/불가능의 차이** — RAM이 충분해도 속도 손해는 미미

### 다른 모델 적용 가능성

**같은 모델의 비양자화(BF16) 버전** — 코드 변경 없음. `--model` 경로만 교체하면 됩니다.

- safetensors 헤더에 dtype 정보가 포함되어 있고, ltx-flash는 이미 BF16 처리 경로(`np.uint16 → mx.bfloat16`)를 갖고 있음
- 블록 키 패턴, 파일 구조 동일 → 블록당 크기만 ~225MB → ~450MB로 증가

| | q4 (양자화) | BF16 (원본) |
|---|---|---|
| 코드 변경 | 없음 | 없음 |
| 블록당 크기 | ~225MB | ~450MB |
| 속도 | 현재 | SSD I/O 소폭 증가, GPU 연산 동일 |

**다른 모델(FLUX.1, SD3, Wan 등)** — 부분 포팅 필요.

- `BlockIndex`, `SSDBlockLoader` — **재사용 가능** (핵심 엔진)
- `SSDStreamingLTXModel` — **새로 작성** (모델별 블록 키 패턴, 비블록 가중치 목록, forward() 방식이 다름)
- `generate.py` — **새로 작성** (파이프라인 연결)
- 전제 조건: 해당 모델의 MLX 포팅이 존재해야 함

가장 유력한 다음 타겟: **FLUX.1** (`mlx-community/FLUX.1-dev-4bit`, DiT 57블록, 블록 구조 LTX와 유사)

### 성능 기준치 (M4 Max 36GB)

| 상태 | 블록당 SSD I/O | 블록당 GPU | step당 시간 |
|------|--------------|-----------|------------|
| Cold (캐시 없음) | ~45ms | ~125ms | ~14초 (CFG) |
| Warm (OS page cache) | ~0ms | ~125ms | ~13초 (CFG) |
| GPU가 병목 | SSD는 이미 0ms | 125ms × 48블록 × 2(CFG) | — |

> GPU 연산(~125ms/블록)이 현재 주요 병목. Flash Attention(`mx.fast.scaled_dot_product_attention`)은 이미 적용됨.

### 주의사항

- `mx.eval()`로 GPU 연산 완료를 명시적으로 대기한 후 블록을 해제해야 함
- `stream_blocks()`는 제너레이터 — 한 번에 블록 1개만 메모리에 존재
- BF16 텐서는 `np.uint16` → `mx.bfloat16` 경로로 처리 (numpy BF16 미지원)
- `SSDStreamingLTXModel`은 `LTXModel` 서브클래스가 아닌 드롭인 대체 패턴 사용
- mmap은 멀티스레드에서 동시 슬라이스 시 내부 포지션 공유로 안전하지 않음 → `bytes(mm[start:end])`로 복사 후 사용
- CFG 사용 시 매 step마다 모델 2회 실행 → 속도 약 2배 느림
- TI2V / FLF2V는 `KeyframeInterpolationPipeline` 사용 — dev 모델 필요
