# Development Notes

## Repository Structure (Key Paths)

- `qt_app.py`: Main desktop GUI (PySide6).
- `src/`: Core logic (material models, optimization, plotting, kinematics).
- `data/data.h5`: Packaged datasets.
- `assets/icons/`: App icons and icon generation script.
- `.github/workflows/release.yml`: Multi-platform release pipeline.
- `README.md`: User-facing install and usage instructions.

## Release Flow (v1.0+)

This project uses a GitHub Actions workflow to build and publish release
artifacts for macOS, Linux, and Windows. Releases are created from tags.

Steps:

1) Tag a release:
```
git tag -a v1.0 -m "v1.0"
git push origin v1.0
```

2) GitHub Actions builds and uploads (from tag):
- `HyperelasticCalibration-macos.zip`
- `HyperelasticCalibration-linux.tar.gz`
- `HyperelasticCalibration-windows.zip`

3) The release is published automatically by the workflow.

## Release Workflow Details

Workflow file:
- `.github/workflows/release.yml`

Behavior:
- On tag push (`v*`), build runs on macOS, Linux, Windows.
- Each job builds the app with PyInstaller and uploads its artifact.
- A final job downloads all artifacts and creates/updates the GitHub release.

Notes:
- macOS build includes `--icon assets/icons/app.icns`.
- `tqdm` is bundled explicitly via `--hidden-import tqdm`.
- Sympy is included via `--collect-all sympy`.

## Homebrew (macOS)

The cask is maintained in:
- `https://github.com/Chongran-Zhao/homebrew-hyperelastic`

After each release, update the cask:

1) Set `version` to the new tag.
2) Update `sha256` for the macOS zip from the GitHub release asset.
3) Commit and push to the tap repo.

Example:
```
cd ~/tmp_homebrew_hyperelastic
git pull
edit Casks/hyperelastic-calibration.rb
git commit -am "Bump hyperelastic-calibration to v1.0"
git push
```

## Updating the Homebrew SHA

1) Fetch the new macOS asset hash:
```
gh release view v1.0 --json assets
```

2) Use the `HyperelasticCalibration-macos.zip` digest as the cask `sha256`.

## Versioning Policy

- `v1.0`, `v2.0`, ... for major stable releases.
- Tags should match the release version exactly.
- Each tag rebuilds all artifacts.

## Building Locally (macOS)

Suggested clean build:
```
rm -rf build dist
/opt/homebrew/anaconda3/bin/pyinstaller --noconfirm --windowed \
  --name HyperelasticCalibration --exclude PyQt5 --icon assets/icons/app.icns \
  --add-data "src:src" --add-data "data/data.h5:data" --add-data "assets/icons:assets/icons" \
  --collect-all sympy --hidden-import sympy --hidden-import tqdm qt_app.py
```

Output:
- `dist/HyperelasticCalibration.app`

## App Icon Notes

- App icon source: `assets/icons/app.icns`.
- GUI uses `assets/icons/app.png` for window icon.
- To regenerate icons, use `assets/icons/generate_icons.py`.

## Troubleshooting

### App fails to launch: missing module
If you see errors like:
```
ModuleNotFoundError: No module named 'tqdm'
```
Add the module to the PyInstaller command via `--hidden-import`.

### Homebrew SHA mismatch
If Homebrew reports a SHA mismatch:
1) Update the cask SHA in the tap repo.
2) Clear the cached download:
```
rm -f ~/Library/Caches/Homebrew/downloads/*HyperelasticCalibration-macos.zip
```
3) Retry:
```
brew update
brew install --cask hyperelastic-calibration
```

## Homebrew (macOS)

The cask is maintained in:
- `https://github.com/Chongran-Zhao/homebrew-hyperelastic`

After each release, update the cask:

1) Set `version` to the new tag.
2) Update `sha256` for the macOS zip from the GitHub release asset.
3) Commit and push to the tap repo.

Example:
```
cd ~/tmp_homebrew_hyperelastic
git pull
edit Casks/hyperelastic-calibration.rb
git commit -am "Bump hyperelastic-calibration to v1.0"
git push
```
