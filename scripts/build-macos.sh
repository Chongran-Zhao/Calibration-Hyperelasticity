#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

version="$(python3 - <<'PY'
import re
from pathlib import Path

text = Path("pyproject.toml").read_text()
match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
if not match:
    raise SystemExit("project version not found")
print(match.group(1))
PY
)"

case "$(uname -m)" in
  arm64) target="aarch64-apple-darwin" ;;
  x86_64) target="x86_64-apple-darwin" ;;
  *) echo "unsupported macOS architecture: $(uname -m)" >&2; exit 1 ;;
esac

npm --prefix frontend run build

build_root="build/macos"
app="$build_root/dist/Hyperelastic Calibration.app"
dmg="dist/Hyperelastic-Calibration-v${version}-${target}.dmg"
stage="$build_root/dmg"

rm -rf "$build_root" "$dmg" "${dmg}.sha256"
mkdir -p "$build_root" "$stage" dist

python3 -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$build_root/dist" \
  --workpath "$build_root/work" \
  packaging/HyperelasticCalibration.spec

executable="$app/Contents/MacOS/Hyperelastic Calibration"
"$executable" --check
codesign --verify --deep --strict "$app"

ditto "$app" "$stage/Hyperelastic Calibration.app"
ln -s /Applications "$stage/Applications"
hdiutil create \
  -volname "Hyperelastic Calibration" \
  -srcfolder "$stage" \
  -ov \
  -format UDZO \
  "$dmg"

shasum -a 256 "$dmg" | tee "${dmg}.sha256"
echo "built $dmg"
