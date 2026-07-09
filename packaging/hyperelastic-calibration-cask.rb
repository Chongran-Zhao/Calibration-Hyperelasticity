# Homebrew cask for the Hyperelastic Calibration desktop app (prebuilt DMG).
#
# Install from the tap:
#   brew install --cask Chongran-Zhao/hyperfit/hyperelastic-calibration
#
# This is a template. scripts/prepare-release.sh resets its checksum, and the
# release workflow writes the real checksum before syncing the tap repository.
cask "hyperelastic-calibration" do
  version "0.2.0"
  sha256 "REPLACE_WITH_AARCH64_APPLE_DARWIN_DMG_SHA256"

  url "https://github.com/Chongran-Zhao/Calibration-Hyperelasticity/releases/download/v#{version}/Hyperelastic-Calibration-v#{version}-aarch64-apple-darwin.dmg"
  name "Hyperelastic Calibration"
  desc "Calibrate hyperelastic models from stress-stretch data"
  homepage "https://github.com/Chongran-Zhao/Calibration-Hyperelasticity"

  app "Hyperelastic Calibration.app"

  postflight do
    system_command "/usr/bin/xattr",
                   args: ["-dr", "com.apple.quarantine", "#{appdir}/Hyperelastic Calibration.app"],
                   sudo: false
  end

  zap trash: "~/.hyperfit"
end
