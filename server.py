"""LTX-2.3 I2V HTTP 서버.

이미지 + 텍스트 → MP4 영상 생성 (on-demand, 생성 후 프로세스 종료).
프로세스 종료 = Metal GPU 메모리 완전 반환.

실행:
    uv run python server.py

포트: 18189
엔드포인트:
    GET  /health    — 상태 확인
    POST /generate  — I2V 영상 생성

요청 JSON:
    {
        "prompt": str,
        "image_b64": "data:image/png;base64,...",  # 첫 프레임
        "width": 768,       # 선택 (기본 768)
        "height": 512,      # 선택 (기본 512)
        "frames": 25,       # 선택 (기본 25)
        "steps": 30,        # 선택 (기본 30)
        "seed": null        # 선택
    }

응답 JSON:
    {
        "success": true,
        "video_b64": "data:video/mp4;base64,...",
        "elapsed": 42.3
    }
"""

import base64
import io
import json
import logging
import os
import signal
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ltx-server")

PORT = 18189
MODEL_DIR = Path.home() / "git/ltx-2-mlx/models/ltx-2.3-mlx-q4"


def _shutdown():
    """응답 전송 후 프로세스 종료 — Metal GPU 메모리 OS 반환."""
    logger.info("프로세스 종료 (Metal GPU 메모리 반환)")
    os.kill(os.getpid(), signal.SIGTERM)


def _generate_i2v(image_path: str, prompt: str, output_path: str,
                  width: int, height: int, frames: int, steps: int, seed=None):
    """LTX I2V 생성."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    import mlx.core as mx
    import ltx_pipelines_mlx.ti2vid_one_stage as pipe_module
    from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
    from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT
    from ltx_core_mlx.model.transformer.model import LTXModelConfig
    from ltx_flash.ssd_stream import SSDStreamingLTXModel

    logger.info(f"모델 로딩: {MODEL_DIR}")
    sft_path = MODEL_DIR / "transformer-distilled.safetensors"
    config = LTXModelConfig()
    ssd_model = SSDStreamingLTXModel(sft_path, config)

    # LTXModel 패치
    original = pipe_module.LTXModel
    pipe_module.LTXModel = lambda *a, **kw: ssd_model

    try:
        pipe = TextToVideoPipeline(model_dir=str(MODEL_DIR))
        logger.info(f"I2V 생성: {width}x{height}, {frames}f, {steps}steps, prompt={prompt[:60]}...")
        pipe.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            image=image_path,
            height=height,
            width=width,
            num_frames=frames,
            num_steps=steps,
            seed=seed,
        )
    finally:
        pipe_module.LTXModel = original
        mx.metal.clear_cache()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info(format % args)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": str(MODEL_DIR)})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._send_json(400, {"success": False, "error": "invalid JSON"})
            return

        prompt = body.get("prompt", "")
        image_b64 = body.get("image_b64", "")
        width = int(body.get("width", 768))
        height = int(body.get("height", 512))
        frames = int(body.get("frames", 25))
        steps = int(body.get("steps", 30))
        seed = body.get("seed", None)

        if not image_b64:
            self._send_json(400, {"success": False, "error": "image_b64 필요"})
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # 입력 이미지 저장
                raw = image_b64.split(",", 1)[-1]
                img_path = os.path.join(tmpdir, "input.png")
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(raw))

                # 출력 MP4 경로
                out_path = os.path.join(tmpdir, "output.mp4")

                t = time.time()
                _generate_i2v(img_path, prompt, out_path, width, height, frames, steps, seed)
                elapsed = time.time() - t

                if not Path(out_path).exists():
                    self._send_json(500, {"success": False, "error": "출력 파일 없음"})
                    return

                video_bytes = Path(out_path).read_bytes()
                video_b64 = base64.b64encode(video_bytes).decode()
                size_kb = len(video_bytes) // 1024
                logger.info(f"생성 완료: {elapsed:.1f}s, {size_kb}KB")

                self._send_json(200, {
                    "success": True,
                    "video_b64": f"data:video/mp4;base64,{video_b64}",
                    "elapsed": elapsed,
                })

        except Exception as e:
            logger.error(f"생성 실패: {e}", exc_info=True)
            self._send_json(500, {"success": False, "error": str(e)})

        finally:
            # 응답 후 프로세스 종료 → GPU 메모리 반환
            Timer(0.5, _shutdown).start()


if __name__ == "__main__":
    if not MODEL_DIR.exists():
        logger.error(f"모델 없음: {MODEL_DIR}")
        logger.error("huggingface-cli download dgrauet/ltx-2.3-mlx-q4 --local-dir ~/git/ltx-2-mlx/models/ltx-2.3-mlx-q4")
        exit(1)

    logger.info(f"ltx-server 시작 (port {PORT})")
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    logger.info(f"Ready: http://127.0.0.1:{PORT}")
    server.serve_forever()
