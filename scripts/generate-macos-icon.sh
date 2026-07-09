#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source_svg="frontend/public/logo.svg"
output="assets/icons/app.icns"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

mkdir -p "$workdir/rendered" "$workdir/AppIcon.iconset" "$(dirname "$output")"
qlmanage -t -s 1024 -o "$workdir/rendered" "$source_svg" >/dev/null
source_png="$workdir/rendered/$(basename "$source_svg").png"

for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$source_png" \
    --out "$workdir/AppIcon.iconset/icon_${size}x${size}.png" >/dev/null
  retina=$((size * 2))
  sips -z "$retina" "$retina" "$source_png" \
    --out "$workdir/AppIcon.iconset/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$workdir/AppIcon.iconset" -o "$output"
echo "generated $output"
