# ltx-flash

**Flash-MoE SSD 블록 스트리밍을 LTX-2.3 비디오 디퓨전에 적용**

LLM 전용으로 알려진 SSD 오프로딩 기법([Flash-MoE](https://github.com/bumsuklee/flash-moe-mlx))을  
LTX-2.3 DiT(Diffusion Transformer) 비디오 생성 모델에 이식한 구현체입니다.  
Apple Silicon + MLX 환경에서 **38GB RAM → 2GB RAM**으로 영상을 생성합니다.

---

## 아이디어

Flash-MoE는 MoE LLM에서 활성 expert K개만 SSD에서 pread()로 on-demand 로드하는 기법입니다.  
LTX-2.3은 48개의 `BasicAVTransformerBlock`으로 구성된 DiT 구조인데,  
"MoE의 expert" 대신 "DiT의 transformer block"을 하나씩 스트리밍하면 같은 원리가 적용됩니다.

```
기존:  transformer 전체(10.5GB) → RAM 38GB
ltx-flash: 블록 1개(~225MB) → 연산 → 해제 → RAM 2GB
```

### 핵심 기법

- **safetensors pread**: 헤더에서 텐서별 byte offset 파싱 → `pread(fd, size, offset)`으로 직접 로드
- **Ping-pong prefetch**: 블록 N 연산 중 블록 N+1을 백그라운드에서 미리 로드
- **OS page cache**: 1회 통과 후 캐시 warm → 후속 denoising step에서 SSD I/O ~1ms/블록

---

## 결과

| | 기존 | ltx-flash |
|---|---|---|
| RAM 사용 | ~38 GB | ~2 GB |
| 블록 로딩 (cold) | — | ~50 ms |
| 블록 로딩 (warm) | — | ~1 ms |
| 영상 생성 (49프레임) | OOM | 22.7초 |
| 영상 생성 (2초, 고화질) | OOM | 88.4초 |

테스트 환경: MacBook Pro M4 Max 36GB

---

## 설치

```bash
git clone --recurse-submodules https://github.com/bumsuklee/ltx-flash
cd ltx-flash
pip install -e .

# 모델 다운로드 (~19.4GB)
huggingface-cli download dgrauet/ltx-2.3-mlx-q4 --local-dir ~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4
```

---

## 사용법

### 영상 생성

```bash
python generate.py generate -p "A golden retriever running in a field" -o output.mp4

# 옵션
python generate.py generate \
  -p "A cat sitting on a windowsill" \
  -o cat.mp4 \
  --height 480 --width 704 \
  --frames 49 \   # 49 = 2초 (8k+1 규칙)
  --steps 8
```

### 벤치마크 (SSD I/O 속도)

```bash
python generate.py benchmark
```

```
블록     I/O (ms)     MB/s
--------------------------------
    0        48.2      4632
    1         3.1     72000   ← OS page cache
    2         1.2    185000
    ...
```

### 스트리밍 테스트 (메모리 측정)

```bash
python generate.py stream-test --num-blocks 8
```

---

## 구조

```
ltx-flash/
├── generate.py          # CLI: generate / benchmark / stream-test
├── ltx_flash/
│   ├── __init__.py
│   └── ssd_stream.py    # BlockIndex, SSDBlockLoader, SSDStreamingLTXModel
├── ltx-2-mlx/           # submodule: ltx-core-mlx, ltx-pipelines-mlx
└── pyproject.toml
```

### 핵심 클래스

**`BlockIndex`** — safetensors 헤더 파싱, 텐서별 (offset, size, shape) 인덱스

**`SSDBlockLoader`** — 블록 단위 pread 로딩, ping-pong prefetch

**`SSDStreamingLTXModel`** — 비블록 weight(~0.8GB) RAM 상주 + 블록 스트리밍

---

## 의존성

- Python >= 3.11
- [MLX](https://github.com/ml-explore/mlx) >= 0.30.0
- safetensors >= 0.4
- [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) (submodule)

---

## 한계

- Apple Silicon + MLX 전용 (CUDA 미지원)
- int4 양자화 모델 전용 (`dgrauet/ltx-2.3-mlx-q4`)
- 오디오 생성 미구현 (비디오만)
- 생성 속도: 48블록 × 8steps = 384회 I/O → SSD 성능에 비례

---

## 참고

- [Flash-MoE](https://github.com/bumsuklee/flash-moe-mlx) — 원본 SSD expert 스트리밍 (MoE LLM용)
- [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) — LTX-2.3 MLX 구현
- [LTX-Video](https://github.com/Lightricks/LTX-Video) — 원본 LTX-2.3
