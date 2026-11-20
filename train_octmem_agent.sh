#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec "${MEM_ENV_DIR:+$MEM_ENV_DIR/bin/}python3" -m utils.launch train "$@"
