# Development Notes

To continue this session, run `codex resume 019b3fbc-4d66-72f2-a04a-4d516a39a49b`.

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

## Session Log (2025-12-23)

High-level summary of this session:
- Added a README example using Zhan (non-Gaussian) and moved screenshots to `assets/examples/zhan-non-gaussian-james-1975`.
- Built a multi-platform release workflow and iterated until release publishing was reliable.
- Re-released as `v1.0` and updated the Homebrew cask to the new macOS asset SHA.
- Added explicit bundling of `tqdm` in the PyInstaller build to fix runtime errors.
- Ensured app startup shows a progress dialog and sets the app name/icon during initialization.
- Documented release and Homebrew procedures in this file.

Key commits pushed:
- `Add README example for Zhan non-Gaussian`
- `Add multi-platform release workflow and install docs`
- `Add quick user guide`
- `Fix app startup icon and bundle tqdm`
- `Make release workflow reliable`
- `Fix release job checkout`
- `Publish release from collected artifacts`
- `Add development release notes`
- `Expand development notes`

Release workflow notes:
- Final workflow uploads build artifacts and publishes the release from a single job to avoid race conditions.
- The release is triggered by pushing a tag like `v1.0`.

Homebrew cask update (v1.0):
- Tap repo: `Chongran-Zhao/homebrew-hyperelastic`
- Cask file: `Casks/hyperelastic-calibration.rb`
- SHA updated to match `HyperelasticCalibration-macos.zip` from the v1.0 release.

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
