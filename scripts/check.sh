#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m py_compile gateway/*.py scripts/*.py videorlm/integrations/director/*.py videorlm/framework/pipeline.py videorlm/framework/_scripts/_smoke.py
node --check dsh-plugin/src/index.js
node --check dsh-plugin/src/skill.js
node --check dsh-plugin/src/client.js
node --check dsh-plugin/src/runtime.js
for file in dsh-plugin/src/tools/*.js dsh-plugin/src/renderers/*.js; do node --check "$file"; done
if [[ -f demo/app.js ]]; then node --check demo/app.js; fi
node tests/test_plugin_runtime.mjs
bash -n scripts/*.sh
git diff --check
echo "ReCA Director checks passed."
