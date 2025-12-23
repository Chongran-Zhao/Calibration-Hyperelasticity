# Development Notes

## Release Flow (v1.0+)

This project uses a GitHub Actions workflow to build and publish release
artifacts for macOS, Linux, and Windows. Releases are created from tags.

Steps:

1) Tag a release:
```
git tag -a v1.0 -m "v1.0"
git push origin v1.0
```

2) GitHub Actions builds and uploads:
- `HyperelasticCalibration-macos.zip`
- `HyperelasticCalibration-linux.tar.gz`
- `HyperelasticCalibration-windows.zip`

3) The release is published automatically by the workflow.

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
