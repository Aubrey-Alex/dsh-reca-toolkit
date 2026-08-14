#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example; fill provider credentials before running."
fi
python3 -m pip install -r requirements.txt
echo "ReCA Director dependencies installed."
