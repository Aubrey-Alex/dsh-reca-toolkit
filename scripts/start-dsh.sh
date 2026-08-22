#!/usr/bin/env bash
# Optional convenience: install the skill once, then open DSH Web.
# The plugin starts the bundled runtime itself. Do not treat this as a second product.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v dsh >/dev/null 2>&1 || {
  echo "error: dsh is not installed; install @deepseek-ai/dsh first" >&2
  exit 1
}
bash "$ROOT/scripts/install_dsh_plugin.sh"
exec dsh web "$@"
