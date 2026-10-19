#!/usr/bin/env bash
# Recreate machine-local state after a fresh clone (notebook or desktop).
# Everything this touches is deliberately NOT in git.
set -euo pipefail
cd "$(dirname "$0")"

# 1. datasets live outside the repo; notebooks reach them via this symlink
mkdir -p "$HOME/data/idl"
ln -sfn "$HOME/data/idl" datasets

# 2. one env for the whole course, from the newest exercise's pins
req=$(ls -d exercise_*/requirements/macOS_and_linux/requirements.txt 2>/dev/null | sort | tail -1)
[ -n "$req" ] || { echo "no requirements found"; exit 1; }
echo "using $req"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -r "$req"
else
  echo "uv not found, falling back to system python $(python3 --version)"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -r "$req"
fi

mkdir -p output && touch output/.gitkeep
echo
echo "done. activate with:  source .venv/bin/activate"
