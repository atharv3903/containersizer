#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
out="${1:-artifacts/stage-4}"
mkdir -p "$out"

scripts/csizer --data data/v1 report --out "$out"
scripts/csizer --data data/v1 plots --out "$out/plots"
scripts/csizer --data data/v1 models --out "$out"
scripts/csizer --data data/v1 stability --out "$out"
scripts/csizer --data data/v1 sizer --out "$out"
