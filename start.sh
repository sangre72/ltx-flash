#!/bin/bash
# LTX I2V HTTP 서버 시작
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
exec uv run python server.py
