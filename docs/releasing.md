# Releasing Hyperelastic Calibration

The release path mirrors Tensbook: a version tag builds an Apple Silicon DMG,
publishes it on GitHub, renders the Homebrew cask with the real checksum, and
syncs that cask to a dedicated tap.

## One-time setup

1. Create the public repository `Chongran-Zhao/homebrew-hyperfit`.
2. Add a repository secret named `HOMEBREW_TAP_TOKEN` to
   `Calibration-Hyperelasticity`. It needs write access to
   `Chongran-Zhao/homebrew-hyperfit`.
3. Keep GitHub Actions enabled for this repository.

The app currently follows Tensbook's Apple Silicon-only release policy. The
PyInstaller bundle is ad-hoc signed, and the cask removes the quarantine
attribute after installation. A Developer ID certificate can replace this
with notarization later without changing the Homebrew command.

## Publish a version

Update every version-bearing file:

```sh
scripts/prepare-release.sh <version>
```

To reproduce the DMG locally:

```sh
python3 -m pip install -e ".[desktop,release]"
scripts/build-macos.sh
```

Review and commit the result, then push an annotated tag:

```sh
git tag -a v<version> -m "Hyperelastic Calibration v<version>"
git push origin v<version>
```

The release workflow:

1. builds `frontend/dist`;
2. freezes `Hyperelastic Calibration.app` with PyInstaller;
3. runs the frozen app smoke test;
4. creates an Apple Silicon DMG and SHA-256 file;
5. publishes the GitHub Release;
6. writes `Casks/hyperelastic-calibration.rb` to the tap.

Users can then install or upgrade with:

```sh
brew install --cask Chongran-Zhao/hyperfit/hyperelastic-calibration
brew upgrade --cask hyperelastic-calibration
```
