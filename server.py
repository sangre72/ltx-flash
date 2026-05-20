"""LTX-2.3 HTTP 서버.

T2V / I2V / FLF2V / TI2V 영상 생성.

포트: 18190
엔드포인트:
    GET  /health    — 상태 확인
    POST /generate  — 영상 생성

요청 JSON (mode 자동 감지):
    T2V  : { "prompt": str, ... }
    I2V  : { "prompt": str, "image_b64": "data:...", ... }
    FLF2V: { "prompt": str, "image_b64": "...", "image_last_b64": "...", ... }
    TI2V : { "prompt": str, "keyframes": [{"image_b64":"...", "frame_index":0}, ...], ... }
           frame_index 생략 시 균등 자동 배분.
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

PORT = 18190
_THIS_DIR = Path(__file__).parent
_GENERATE_PY = _THIS_DIR / "generate.py"
MODEL_DIR = Path.home() / "git/ltx-2-mlx/models/ltx-2.3-mlx-q4"


def _shutdown():
    """응답 전송 후 프로세스 종료 — Metal GPU 메모리 OS 반환."""
    logger.info("프로세스 종료 (Metal GPU 메모리 반환)")
    os.kill(os.getpid(), signal.SIGTERM)


def _base_cmd(prompt: str, output_path: str, width: int, height: int,
              frames: int, fps: int, steps: int, cfg_scale: float,
              prefetch_depth: int, seed=None, negative_prompt: str = "") -> list:
    cmd = [
        "uv", "run", "python", str(_GENERATE_PY), "generate",
        "--prompt", prompt,
        "--output", output_path,
        "--width", str(width), "--height", str(height),
        "--frames", str(frames), "--fps", str(fps),
        "--steps", str(steps), "--model", str(MODEL_DIR),
        "--cfg-scale", str(cfg_scale),
        "--prefetch-depth", str(prefetch_depth),
        "--quiet",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if negative_prompt:
        cmd += ["--negative-prompt", negative_prompt]
    return cmd


def _run_cmd(cmd: list, mode: str, prompt: str):
    logger.info(f"{mode}: prompt={prompt[:60]}...")
    result = subprocess.run(cmd, cwd=str(_THIS_DIR), timeout=1200)
    if result.returncode != 0:
        raise RuntimeError(f"generate.py 실패 (exit={result.returncode})")


def _generate(prompt: str, output_path: str, width: int, height: int,
              frames: int, fps: int, steps: int, seed=None,
              cfg_scale: float = 4.0, negative_prompt: str = "",
              image_path: str = None, last_frame_path: str = None,
              keyframe_paths: list = None, keyframe_indices: list = None,
              prefetch_depth: int = 1):
    """통합 생성 함수.

    mode 자동 감지:
      TI2V  : keyframe_paths 2장 이상
      FLF2V : image_path + last_frame_path
      I2V   : image_path만
      T2V   : image_path 없음
    """
    cmd = _base_cmd(prompt, output_path, width, height, frames, fps, steps,
                    cfg_scale, prefetch_depth, seed, negative_prompt)

    if keyframe_paths and len(keyframe_paths) >= 2:
        # TI2V — 여러 키프레임 보간
        for p in keyframe_paths:
            cmd += ["--image", p]
        if keyframe_indices:
            for idx in keyframe_indices:
                cmd += ["--image-index", str(idx)]
        _run_cmd(cmd, "TI2V", prompt)
    elif image_path and last_frame_path:
        # FLF2V
        cmd += ["--first-frame", image_path, "--last-frame", last_frame_path]
        _run_cmd(cmd, "FLF2V", prompt)
    elif image_path:
        # I2V
        cmd += ["--image", image_path]
        _run_cmd(cmd, "I2V", prompt)
    else:
        # T2V
        _run_cmd(cmd, "T2V", prompt)


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
        if self.path == "/shutdown":
            self._send_json(200, {"success": True})
            Timer(0.5, _shutdown).start()
            return
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
        keyframes = body.get("keyframes", [])  # TI2V: [{image_b64, frame_index?}]
        width = int(body.get("width", 512))
        height = int(body.get("height", 768))
        frames = int(body.get("frames", 145))
        fps = int(body.get("fps", 14))
        steps = int(body.get("steps", 20))
        seed = body.get("seed", None)
        cfg_scale = float(body.get("cfg_scale", 1.0))
        negative_prompt = body.get("negative_prompt", "")
        prefetch_depth = int(body.get("prefetch_depth", 2))

        if not prompt:
            self._send_json(400, {"success": False, "error": "prompt 필요"})
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                def _save_b64(b64str: str, name: str) -> str:
                    raw = b64str.split(",", 1)[-1]
                    p = os.path.join(tmpdir, name)
                    with open(p, "wb") as f:
                        f.write(base64.b64decode(raw))
                    return p

                img_path = _save_b64(image_b64, "input.png") if image_b64 else None
                last_path = _save_b64(image_last_b64, "input_last.png") if image_last_b64 else None

                # TI2V 키프레임 처리
                kf_paths, kf_indices = [], []
                for i, kf in enumerate(keyframes):
                    if not kf.get("image_b64"):
                        continue
                    kp = _save_b64(kf["image_b64"], f"kf_{i:02d}.png")
                    kf_paths.append(kp)
                    if kf.get("frame_index") is not None:
                        kf_indices.append(int(kf["frame_index"]))

                out_path = os.path.join(tmpdir, "output.mp4")
                t = time.time()
                _generate(
                    prompt=prompt, output_path=out_path,
                    width=width, height=height, frames=frames, fps=fps,
                    steps=steps, seed=seed, cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    image_path=img_path, last_frame_path=last_path,
                    keyframe_paths=kf_paths or None,
                    keyframe_indices=kf_indices or None,
                    prefetch_depth=prefetch_depth,
                )
                elapsed = time.time() - t

                if not Path(out_path).exists():
                    self._send_json(500, {"success": False, "error": "출력 파일 없음"})
                    return

                video_bytes = Path(out_path).read_bytes()
                video_b64_out = base64.b64encode(video_bytes).decode()
                logger.info(f"생성 완료: {elapsed:.1f}s, {len(video_bytes)//1024}KB")
                self._send_json(200, {
                    "success": True,
                    "video_b64": f"data:video/mp4;base64,{video_b64_out}",
                    "elapsed": elapsed,
                })

        except Exception as e:
            logger.error(f"생성 실패: {e}", exc_info=True)
            self._send_json(500, {"success": False, "error": str(e)})

        # 데몬 상주 방식: 요청 후 자살하지 않음.
        # generate.py 서브프로세스가 종료되면서 Metal GPU 메모리는 자동 해제됨.
        # 명시 종료가 필요하면 클라이언트가 POST /shutdown 호출.


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
