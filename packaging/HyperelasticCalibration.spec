# -*- mode: python ; coding: utf-8 -*-

import os
import platform
import re
from pathlib import Path

ROOT = Path(SPECPATH).parent
VERSION_TEXT = (ROOT / "pyproject.toml").read_text()
VERSION = re.search(r'^version = "([^"]+)"', VERSION_TEXT, re.MULTILINE).group(1)
ICON = ROOT / "assets" / "icons" / "app.icns"
ENTRY = ROOT / "packaging" / "macos_entry.py"

signing_identity = os.environ.get("APPLE_CODESIGN_IDENTITY") or None
target_arch = platform.machine()

datas = [
    (str(ROOT / "data" / "data.h5"), "data"),
    (str(ROOT / "frontend" / "dist"), "frontend/dist"),
    (str(ROOT / "hyperfit" / "lebedev" / "Lebedev.txt"), "hyperfit/lebedev"),
]

hiddenimports = [
    "uvicorn.lifespan.on",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
]

a = Analysis(
    [str(ENTRY)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pandas", "pytest", "tkinter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Hyperelastic Calibration",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=target_arch,
    codesign_identity=signing_identity,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Hyperelastic Calibration",
)

app = BUNDLE(
    coll,
    name="Hyperelastic Calibration.app",
    icon=str(ICON),
    bundle_identifier="io.github.chongranzhao.hyperelastic-calibration",
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": "Hyperelastic Calibration",
        "CFBundleName": "Hyperelastic Calibration",
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
    },
)
