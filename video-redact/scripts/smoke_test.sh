#!/usr/bin/env bash
set -euo pipefail
PROJ="${PROJ:-/root/autodl-tmp/video-redact}"
cd "$PROJ"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"
python scripts/make_sample.py -o samples/synthetic_smoke.mp4
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_smoke.mp4 -v --device cuda:0 || \
  python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_smoke.mp4 -v --cpu
ls -lh jobs/out_smoke.mp4
echo SMOKE_OK
