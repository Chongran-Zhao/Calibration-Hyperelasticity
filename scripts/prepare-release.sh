#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare-release.sh <version> [--sha256 <dmg_sha256>]

Examples:
  scripts/prepare-release.sh 0.3.0
  scripts/prepare-release.sh 0.3.0 --sha256 <64-hex-sha>

Updates:
  - pyproject.toml
  - hyperfit/__init__.py
  - frontend/package.json
  - frontend/package-lock.json
  - packaging/hyperelastic-calibration-cask.rb
EOF
}

version=""
sha256=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --sha256)
      [[ $# -ge 2 ]] || { echo "error: --sha256 requires a value" >&2; exit 2; }
      sha256="$2"
      shift 2
      ;;
    -*)
      echo "error: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      [[ -z "$version" ]] || { echo "error: version already set to $version" >&2; exit 2; }
      version="$1"
      shift
      ;;
  esac
done

[[ -n "$version" ]] || { usage >&2; exit 2; }
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.+-][0-9A-Za-z.-]+)?$ ]] || {
  echo "error: version must look like 0.3.0" >&2
  exit 2
}
[[ -z "$sha256" || "$sha256" =~ ^[0-9a-fA-F]{64}$ ]] || {
  echo "error: --sha256 must be a 64-character hex digest" >&2
  exit 2
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HYPERFIT_RELEASE_VERSION="$version"
export HYPERFIT_RELEASE_SHA="${sha256:-REPLACE_WITH_AARCH64_APPLE_DARWIN_DMG_SHA256}"

python3 <<'PY'
import json
import os
import re
from pathlib import Path

version = os.environ["HYPERFIT_RELEASE_VERSION"]
sha256 = os.environ["HYPERFIT_RELEASE_SHA"]


def replace_once(path, pattern, replacement):
    target = Path(path)
    text = target.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"expected one replacement in {path}: {pattern}")
    target.write_text(updated)


replace_once("pyproject.toml", r'^version = "[^"]+"', f'version = "{version}"')
replace_once("hyperfit/__init__.py", r'^__version__ = "[^"]+"', f'__version__ = "{version}"')
replace_once(
    "packaging/hyperelastic-calibration-cask.rb",
    r'^  version "[^"]+"',
    f'  version "{version}"',
)
replace_once(
    "packaging/hyperelastic-calibration-cask.rb",
    r'^  sha256 "[^"]+"',
    f'  sha256 "{sha256}"',
)

for filename in ("frontend/package.json", "frontend/package-lock.json"):
    path = Path(filename)
    document = json.loads(path.read_text())
    document["version"] = version
    if filename.endswith("package-lock.json"):
        document["packages"][""]["version"] = version
    path.write_text(json.dumps(document, indent=2) + "\n")

print(f"updated Hyperelastic Calibration release metadata to {version}")
PY

echo "done"
