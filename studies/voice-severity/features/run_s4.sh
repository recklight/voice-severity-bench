#!/usr/bin/env bash
# S4 batch: clinical measures, then the SSL models, /a/ first. Resumable (SSL cache).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY features/extract_clinical.py --materials a --native
$PY features/extract_clinical.py --materials i sentences whole
for model in A4 A5 A6; do $PY features/extract_ssl.py --model $model --materials a; done
for material in i sentences whole; do
  for model in A4 A5 A6; do $PY features/extract_ssl.py --model $model --materials $material; done
done
echo "S4 batch finished"
