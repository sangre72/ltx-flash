# ltx-flash

> LTX-2.3 비디오 생성 모델을 **Apple Silicon Mac에서 RAM 2GB로 실행**하는 SSD 스트리밍 라이브러리

![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20Mac%20only-black?logo=apple)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Framework](https://img.shields.io/badge/framework-MLX-orange)

> ⚠️ **Mac 전용입니다.**  
> Apple Silicon (M1 / M2 / M3 / M4) + MLX 프레임워크 기반으로 만들어졌습니다.  
> NVIDIA GPU(CUDA) 및 Intel Mac은 지원하지 않습니다.

---

## 어떤 문제를 해결하나요?

LTX-2.3은 Lightricks가 만든 최신 오픈소스 비디오 생성 AI입니다.  
문제는 모델 크기가 너무 커서 일반 Mac으로는 실행이 불가능하다는 점입니다.

| | RAM 전체 상주 | ltx-flash |
|---|---|---|
| 필요 RAM | ~38 GB | **~3 GB** |
| M4 Max 36GB에서 | ❌ OOM | ✅ 동작 |
| 첫 번째 step | ~12초 | ~14초 (+2초) |
| 이후 step (warm) | ~12초 | ~13초 (거의 동일) |
| 영상 생성 (2초) | 불가 | 22.7초 |
| 영상 생성 (10초, 고화질) | 불가 | ~157초 |

> **속도 손해는 거의 없습니다.**  
> GPU 연산(~125ms/블록)이 압도적인 병목이고, SSD I/O는 2번째 step부터 OS page cache로 ~0ms가 됩니다.  
> ltx-flash의 가치는 속도가 아니라 **"실행 불가 → 실행 가능"** 의 차이에 있습니다.

---

## 어떻게 동작하나요?

LTX-2.3의 Transformer는 **48개의 블록**으로 구성되어 있고, 전체 크기는 약 10.5GB입니다.  
기존에는 이 48개를 전부 RAM에 올려야 했습니다.

ltx-flash는 아이디어를 바꿉니다:

```
기존:  모델 전체 10.5GB → RAM에 올림 → 38GB 필요
ltx-flash: 블록 1개(225MB) → SSD에서 읽기 → 연산 → 해제 → 다음 블록
```

이 방식은 LLM에서 쓰이는 [Flash-MoE](https://github.com/danveloper/flash-moe) 기법을  
비디오 디퓨전 모델(DiT)에 최초로 적용한 구현입니다.

### 최적화 기법

- **mmap + OS page cache**: SSD에서 읽되, 운영체제가 자주 쓰는 데이터를 자동으로 메모리에 캐싱 → 2번째 step부터 SSD I/O 거의 0ms
- **N-ahead prefetch**: 현재 블록 GPU 연산 중 다음 2~3개 블록을 미리 읽기 (병렬 처리)
- **블록 객체 풀링**: `TransformerBlock` 객체 1개를 재사용, 가중치만 교체 → 객체 생성 오버헤드 제거
- **CFG 지원**: Classifier-Free Guidance로 품질 향상 (`--cfg-scale 3.0`)

---

## 설치

### 요구 사항

- Apple Silicon Mac (M1 이상)
- macOS 14 이상
- Python 3.11 이상
- 디스크 여유 공간 40GB 이상 (모델 파일)

### 1. 저장소 클론

```bash
git clone --recurse-submodules https://github.com/sangre72/ltx-flash
cd ltx-flash
```

> `--recurse-submodules` 를 반드시 붙여야 합니다. `ltx-2-mlx` 서브모듈이 함께 받아집니다.

### 2. Python 환경 설정 (uv 권장)

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ltx-2-mlx/packages/ltx-core-mlx
uv pip install -e ltx-2-mlx/packages/ltx-pipelines-mlx
uv pip install -e .
```

### 3. 모델 다운로드 (~40GB)

```bash
huggingface-cli download dgrauet/ltx-2.3-mlx-q4 \
  --local-dir ~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4
```

다운로드에 수십 분이 걸릴 수 있습니다. 완료되면 아래 파일들이 생성됩니다:

```
~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4/
├── transformer-distilled.safetensors  (11GB) ← 주력 모델
├── transformer-dev.safetensors        (11GB)
├── connector.safetensors              (5.9GB)
├── ltx-2.3-22b-distilled-lora-384...  (7.1GB)
└── vae, vocoder, upscaler 등...
```

---

## 사용법

### 생성 모드 자동 감지

인자 조합에 따라 모드가 자동으로 선택됩니다.

| 모드 | 사용 인자 | 설명 |
|------|----------|------|
| **T2V** | (기본) | 텍스트만으로 영상 생성 |
| **I2V** | `--image` 1장 | 이미지를 첫 프레임으로 고정 |
| **TI2V** | `--image` 여러 장 | 여러 이미지를 키프레임으로 고정 후 보간 |
| **FLF2V** | `--first-frame` + `--last-frame` | 첫/마지막 프레임 고정 후 사이를 보간 |

---

### T2V — 텍스트 → 영상

```bash
python generate.py generate \
  -p "A Labrador Retriever running across a wide open field, large tree in background, golden hour lighting, cinematic wide shot" \
  -o output.mp4 \
  --height 720 --width 480 \
  --frames 137 \
  --steps 30 \
  --cfg-scale 3.0 \
  --negative-prompt "blurry, low quality, distorted"
```

### I2V — 이미지 → 영상

이미지를 첫 프레임으로 고정하고 이후 움직임을 생성합니다.

```bash
python generate.py generate \
  -p "The dog starts running forward energetically" \
  --image dog.jpg \
  -o output.mp4 \
  --frames 97 --steps 20
```

### TI2V — 여러 이미지 → 영상 (키프레임 보간)

여러 이미지를 특정 프레임 위치에 고정하고 자연스럽게 보간합니다.

```bash
# 이미지 3장 → 자동으로 0, 24, 48 프레임에 균등 배치
python generate.py generate \
  -p "Smooth transition between scenes" \
  --image scene1.jpg \
  --image scene2.jpg \
  --image scene3.jpg \
  --frames 49 -o output.mp4 --steps 20

# 프레임 위치 직접 지정
python generate.py generate \
  -p "Transition" \
  --image opening.jpg --image-index 0 \
  --image closing.jpg --image-index 96 \
  --frames 97 -o output.mp4 --steps 20
```

### FLF2V — 첫/마지막 프레임 → 영상 보간

시작 장면과 끝 장면을 주면 사이를 자동으로 채워줍니다.

```bash
python generate.py generate \
  -p "A smooth cinematic transition from day to night" \
  --first-frame morning.jpg \
  --last-frame night.jpg \
  --frames 97 -o output.mp4 --steps 20 --cfg-scale 3.0
```

---

## 옵션 전체 목록

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-p`, `--prompt` | **(필수)** | 생성할 영상 설명 (영어 권장) |
| `-o`, `--output` | `output.mp4` | 출력 파일 경로 |
| `-H`, `--height` | `480` | 영상 높이 (px). 32 배수 권장 |
| `-W`, `--width` | `704` | 영상 너비 (px). 32 배수 권장 |
| `-f`, `--frames` | `49` | 프레임 수 (`8k+1` 형식 필수) |
| `-s`, `--steps` | `8` | 디노이징 스텝. 높을수록 품질↑ 속도↓ |
| `--fps` | `24.0` | 출력 FPS (TI2V/FLF2V 인덱스 계산용) |
| `--prefetch-depth` | `2` | 미리 읽을 블록 수. 높을수록 SSD 대기↓ (2~4 권장) |
| `--cfg-scale` | `1.0` | CFG 강도. `1.0`=없음(빠름), `3.0~7.0`=품질↑ (속도 약 2배) |
| `-n`, `--negative-prompt` | 내장 기본값 | 결과에서 제외할 요소. `--cfg-scale > 1.0`일 때만 동작 |
| `-i`, `--image` | — | 이미지 경로. 여러 번 지정 가능 (I2V / TI2V) |
| `--image-index` | 자동 균등 배분 | 각 이미지가 고정될 프레임 번호 (`--image`와 1:1 대응) |
| `--first-frame` | — | FLF2V 시작 프레임 이미지 |
| `--last-frame` | — | FLF2V 끝 프레임 이미지 |
| `--seed` | `42` | 랜덤 시드. 같은 값이면 동일한 결과 재현 |
| `-m`, `--model` | `~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4` | 모델 디렉토리 경로 |
| `-q`, `--quiet` | `False` | 진행 로그 숨김 |

---

## 프레임 수 참고표 (8k+1 규칙)

LTX 모델은 `8×k + 1` 형식의 프레임 수만 허용합니다 (예: 9, 17, 25, 33, 41, 49...).

| 영상 길이 | FPS | 프레임 수 |
|-----------|-----|-----------|
| 2초 | 24 | 49 |
| 4초 | 24 | 97 |
| 5초 | 14 | 57 |
| 10초 | 14 | 137 |
| 10초 | 24 | 241 |

---

## 속도 vs 품질 가이드

```bash
# 빠른 미리보기 (~15초)
python generate.py generate -p "..." -o preview.mp4 --frames 49 --steps 8

# 균형 (~60초)
python generate.py generate -p "..." -o balanced.mp4 \
  --frames 97 --steps 20 --cfg-scale 3.0

# 고품질 (~157초)
python generate.py generate -p "..." -o hq.mp4 \
  --frames 137 --steps 30 --prefetch-depth 3 \
  --cfg-scale 3.0 --negative-prompt "blurry, low quality, distorted"
```

---

## 실제 동작 방식

영상을 생성하면 내부적으로 다음 순서로 동작합니다.

### 모델 파일 관리 방식 — 논리적 주소 접근

ltx-flash는 모델 파일을 블록별로 **물리적으로 분리하지 않습니다.**  
`transformer-distilled.safetensors` 1개 파일을 디스크에 그대로 두고, 필요한 위치만 골라 읽습니다.

```
파일: transformer-distilled.safetensors (10.5GB, 분리 없음)
                  ↓
safetensors 헤더(JSON) 파싱
  → 각 텐서의 byte offset, size, dtype, shape를 BlockIndex에 인덱싱
                  ↓
블록 0 필요 시: "transformer.transformer_blocks.0.*" offset 조회
               → mmap[offset:offset+size] 로 해당 위치만 읽기
블록 1 필요 시: "transformer.transformer_blocks.1.*" offset 조회
               → mmap[offset:offset+size] 로 해당 위치만 읽기
               ...
```

책 전체를 챕터별로 찢지 않고, **페이지 번호를 기억해뒀다가 그 페이지만 펼쳐 읽는 방식**입니다.

### 블록 처리 순서 — 항상 0→47 고정

"어떤 블록이 필요한지" 판단하는 로직이 없습니다.  
Transformer는 입력이 블록 0부터 47까지 **순차적으로 통과**하는 고정 구조라서, 항상 전부 순서대로 씁니다.

```
디노이징 1 step: 블록 0 → 블록 1 → ... → 블록 47  (48블록 전부)
디노이징 2 step: 블록 0 → 블록 1 → ... → 블록 47  (또 48블록 전부)
...총 N step 반복 (기본 30 step)
```

N-ahead prefetch가 가능한 이유도 같습니다.  
"다음 블록이 뭔지 예측"하는 게 아니라, **다음 번호가 이미 정해져 있기 때문에** 미리 읽어둘 수 있습니다.

```
현재: 블록 3 GPU 연산 중
prefetch: 블록 4, 5 백그라운드에서 SSD 읽기 시작
→ 블록 3 끝나면 블록 4는 이미 RAM에 대기
```

### Dense Transformer — 블록 스킵 없음

LTX-2.3은 **Dense Transformer** 구조입니다.  
가중치 값이 크든 작든, 입력이 무엇이든 상관없이 **48개 블록 전부를 예외 없이 연산**합니다.

```
Dense (LTX-2.3):        모든 파라미터가 매번 전부 활성화
                         블록 0 → 1 → ... → 47, 스킵 없음

Sparse / MoE (GPT-4):   라우터가 입력마다 일부 expert만 선택
                         → 나머지는 이번 step엔 연산 안 함
```

가중치 값이 크면 출력에 **강하게 반영**되고, 작으면 **약하게 반영**될 뿐입니다.  
값이 작다고 해당 블록이 스킵되지 않습니다 — 작은 기여도 합산됩니다.

비디오 DiT는 수천 개의 시공간 패치가 attention으로 전부 연결되어 있어  
"이 패치는 이 블록이 담당" 식의 라우팅이 구조적으로 어렵습니다.  
현재 오픈소스 비디오 생성 모델(LTX, Wan, CogVideoX 등)은 대부분 Dense입니다.

> **"Flash-MoE"** 이름은 SSD 오프로딩 기법을 MoE에서 차용했다는 의미입니다.  
> 실제 동적 라우팅(어떤 블록을 쓸지 결정)은 없습니다.

---

### 1단계 — 초기화 (~1초)

```
safetensors 헤더 파싱
  → 10.5GB 파일에서 각 텐서의 위치(offset)를 미리 인덱싱
  → 비블록 가중치 58개(~0.8GB) RAM에 로드 (adaln, patchify_proj 등)
  → 나머지 블록 가중치(~9.7GB)는 SSD에 그대로 유지
```

### 2단계 — 텍스트 인코딩

```
프롬프트 → Gemma 3 12B 텍스트 인코더 → 임베딩 벡터 생성
  → 인코딩 완료 후 Gemma 메모리 해제 (low_memory 모드)
```

### 3단계 — 디노이징 루프 (steps만큼 반복)

각 step마다 48개 블록을 순서대로 처리합니다:

```
┌─────────────────────────────────────────────────────────┐
│ Block 0                                                 │
│   SSD → mmap → numpy → mx.array  (block 0 로드, ~45ms) │
│   [prefetch] block 1, 2 백그라운드 로드 시작            │
│   GPU 연산: attention + feedforward          (~125ms)   │
│   mx.eval() → 연산 완료 대기                            │
│   del block → mx.clear_cache() → 메모리 해제           │
├─────────────────────────────────────────────────────────┤
│ Block 1                                                 │
│   prefetch 완료된 가중치 즉시 사용           (~0ms)     │
│   [prefetch] block 3 백그라운드 로드 시작               │
│   GPU 연산                                   (~125ms)   │
│   ...                                                   │
└─────────────────────────────────────────────────────────┘
  × 48블록 반복
  × steps 횟수만큼 전체 반복 (2번째 step부터 OS cache로 SSD ~0ms)
```

**CFG 사용 시**: 각 step에서 positive + negative 프롬프트로 2회 실행 후 결합

### 4단계 — 디코딩 & 저장

```
video latent → VAE 디코더 → 픽셀 프레임
audio latent → Audio VAE → 멜 스펙트로그램 → Vocoder → 파형
  → ffmpeg으로 mp4 합성 → 파일 저장
```

### 메모리 사용 분포

```
RAM 상주 (항상):
  비블록 가중치   ~0.8 GB  (adaln, patchify_proj, proj_out 등)
  활성화 버퍼     ~2.0 GB  (video/audio hidden states)

순간 사용 (연산 후 즉시 해제):
  현재 블록       ~0.2 GB  (225MB, 연산 완료 즉시 del)

총합:            ~3.0 GB  (기존 38GB 대비 92% 절감)
```

---

## 진단 명령어

### SSD I/O 속도 측정

```bash
python generate.py benchmark
```

```
블록     I/O (ms)     MB/s
--------------------------------
    0        48.2      4632   ← cold (처음 읽기)
    1         3.1     72000   ← warm (OS page cache)
    2         1.2    185000
    ...
```

### 메모리 사용량 확인

```bash
python generate.py stream-test --num-blocks 8
```

---

## 프로젝트 구조

```
ltx-flash/
├── generate.py              # CLI 진입점 (generate / benchmark / stream-test)
├── ltx_flash/
│   ├── __init__.py
│   └── ssd_stream.py        # 핵심 구현
│       ├── BlockIndex           — safetensors 헤더 파싱 & 오프셋 인덱싱
│       ├── SSDBlockLoader       — mmap/pread 로딩, N-ahead prefetch, 블록 풀링
│       ├── SSDStreamingLTXModel — 비블록 가중치 RAM 상주 + 블록 스트리밍 엔진
│       └── BlockKVCache         — step 간 KV 캐시 (실험적)
├── ltx-2-mlx/               # git submodule
│   └── packages/
│       ├── ltx-core-mlx/        — LTX-2.3 모델 구현 (MLX)
│       └── ltx-pipelines-mlx/   — T2V / I2V / FLF2V 파이프라인
└── pyproject.toml
```

---

## 다른 모델에 적용 가능한가?

### 같은 모델의 비양자화(BF16) 버전 — 코드 변경 없음

`--model` 경로만 BF16 모델 디렉토리로 바꾸면 그대로 실행됩니다.

ltx-flash는 safetensors 헤더의 dtype을 읽어 자동으로 처리하며, BF16 경로(`np.uint16 → mx.bfloat16`)가 이미 구현되어 있습니다.

| | q4 (양자화) | BF16 (원본) |
|---|---|---|
| 코드 변경 | 없음 | 없음 |
| 블록당 크기 | ~225MB | ~450MB |
| 속도 | 현재 | SSD I/O 소폭 증가, GPU 연산 동일 |

### 다른 모델(FLUX.1, SD3, Wan 등) — 부분 포팅 필요

핵심 엔진(`BlockIndex`, `SSDBlockLoader`)은 재사용 가능하고, 모델 연결 레이어만 새로 작성하면 됩니다.

| 구성요소 | 작업 |
|---------|------|
| `BlockIndex` | 재사용 |
| `SSDBlockLoader` | 재사용 |
| `SSDStreamingLTXModel` | 새로 작성 (블록 키 패턴, forward() 방식이 모델마다 다름) |
| `generate.py` | 새로 작성 (파이프라인 연결) |

전제 조건: 해당 모델의 **MLX 포팅이 존재**해야 합니다.

가장 유력한 다음 타겟: **FLUX.1** — DiT 57블록, 블록 구조가 LTX와 유사, `mlx-community/FLUX.1-dev-4bit` 존재

---

## 한계

- **Apple Silicon 전용**: MLX 프레임워크 기반, NVIDIA GPU 미지원
- **int4 양자화 모델 전용**: `dgrauet/ltx-2.3-mlx-q4` 포맷에 최적화
- **GPU 병목**: 블록당 GPU 연산 ~125ms가 현재 주요 병목 (Apple M4 Max 기준)
- **CFG 사용 시 2배 느림**: CFG는 매 step마다 모델을 2회 실행

---

## 의존성

- Python ≥ 3.11
- [MLX](https://github.com/ml-explore/mlx) ≥ 0.31.0 (Apple Silicon GPU 프레임워크)
- [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) (서브모듈)
- safetensors ≥ 0.4
- numpy ≥ 1.26

---

## 관련 프로젝트

| 프로젝트 | 설명 |
|---------|------|
| [flash-moe](https://github.com/danveloper/flash-moe) | LLM용 SSD pread 스트리밍 (C/Metal, 원조 기법) |
| [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) | LTX-2.3 MLX 파이프라인 (ltx-flash 서브모듈) |
| [scope-ltx-2](https://github.com/daydreamlive/scope-ltx-2) | LTX-2.3 블록 스트리밍 (CUDA 환경) |
| [LTX-Video](https://github.com/Lightricks/LTX-Video) | Lightricks 공식 LTX-2.3 |
