#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${MEM_ENV_DIR:+$MEM_ENV_DIR/bin/}python3" -m utils.launch eval "$@"
