#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v dsh >/dev/null 2>&1 || {
  echo "dsh is not installed; install @deepseek-ai/dsh first" >&2
  exit 1
}
dsh plugin --profile web add "$ROOT/dsh-plugin"
