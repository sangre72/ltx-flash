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
        "image_b64": "data:image/png;base64,...",
        "width": 512,
        "height": 768,
        "frames": 25,
        "steps": 30,
        "seed": null
    }

응답 JSON:
    {
        "success": true,
        "video_b64": "data:video/mp4;base64,...",
        "elapsed": 42.3
    }
"""

import base64
import json
import logging
import os
import signal
import subprocess
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
_THIS_DIR = Path(__file__).parent
_GENERATE_PY = _THIS_DIR / "generate.py"
MODEL_DIR = Path.home() / "git/ltx-2-mlx/models/ltx-2.3-mlx-q4"


def _shutdown():
    """응답 전송 후 프로세스 종료 — Metal GPU 메모리 OS 반환."""
    logger.info("프로세스 종료 (Metal GPU 메모리 반환)")
    os.kill(os.getpid(), signal.SIGTERM)


def _generate_i2v(image_path: str, prompt: str, output_path: str,
                  width: int, height: int, frames: int, fps: int, steps: int,
                  seed=None, cfg_scale: float = 4.0, negative_prompt: str = "",
                  last_frame_path: str = None):
    """generate.py subprocess로 I2V 생성.

    last_frame_path 지정 시 flf2v 모드 (first+last frame 보간).
    """
    if last_frame_path:
        # flf2v: first-frame + last-frame 보간
        cmd = [
            "uv", "run", "python", str(_GENERATE_PY), "generate",
            "--prompt", prompt,
            "--first-frame", image_path,
            "--last-frame", last_frame_path,
            "--output", output_path,
            "--width", str(width),
            "--height", str(height),
            "--frames", str(frames),
            "--fps", str(fps),
            "--steps", str(steps),
            "--model", str(MODEL_DIR),
            "--cfg-scale", str(cfg_scale),
        ]
    else:
        cmd = [
            "uv", "run", "python", str(_GENERATE_PY), "generate",
            "--prompt", prompt,
            "--image", image_path,
            "--output", output_path,
            "--width", str(width),
            "--height", str(height),
            "--frames", str(frames),
            "--fps", str(fps),
            "--steps", str(steps),
            "--model", str(MODEL_DIR),
            "--cfg-scale", str(cfg_scale),
        ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if negative_prompt:
        cmd += ["--negative-prompt", negative_prompt]

    mode = "FLF2V" if last_frame_path else "I2V"
    logger.info(f"{mode} 생성: {width}x{height}, {frames}f@{fps}fps, {steps}steps, cfg={cfg_scale}, prompt={prompt[:60]}...")
    result = subprocess.run(
        cmd,
        cwd=str(_THIS_DIR),
        timeout=1200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generate.py 실패 (exit={result.returncode})")


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
        image_last_b64 = body.get("image_last_b64", "")  # flf2v용 마지막 프레임
        width = int(body.get("width", 512))
        height = int(body.get("height", 768))
        frames = int(body.get("frames", 145))
        fps = int(body.get("fps", 14))
        steps = int(body.get("steps", 30))
        seed = body.get("seed", None)
        cfg_scale = float(body.get("cfg_scale", 4.0))
        negative_prompt = body.get("negative_prompt", "")

        if not image_b64:
            self._send_json(400, {"success": False, "error": "image_b64 필요"})
            return
        if not prompt:
            self._send_json(400, {"success": False, "error": "prompt 필요"})
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # 첫 프레임 이미지 저장
                raw = image_b64.split(",", 1)[-1]
                img_path = os.path.join(tmpdir, "input.png")
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(raw))

                # 마지막 프레임 이미지 (flf2v 모드)
                last_path = None
                if image_last_b64:
                    raw_last = image_last_b64.split(",", 1)[-1]
                    last_path = os.path.join(tmpdir, "input_last.png")
                    with open(last_path, "wb") as f:
                        f.write(base64.b64decode(raw_last))

                out_path = os.path.join(tmpdir, "output.mp4")

                t = time.time()
                _generate_i2v(img_path, prompt, out_path, width, height, frames, fps, steps, seed, cfg_scale, negative_prompt, last_path)
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
        logger.error(f"필요: {MODEL_DIR}")
        exit(1)
    if not _GENERATE_PY.exists():
        logger.error(f"generate.py 없음: {_GENERATE_PY}")
        exit(1)

    logger.info(f"ltx-server 시작 (port {PORT})")
    logger.info(f"모델: {MODEL_DIR}")
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    logger.info(f"Ready: http://127.0.0.1:{PORT}")
    server.serve_forever()
