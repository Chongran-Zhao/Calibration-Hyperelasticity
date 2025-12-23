# Hyperelastic Material Calibration

## Install (macOS App)

```
brew install --cask Chongran-Zhao/hyperelastic/hyperelastic-calibration
```

## Update (macOS App)

```
brew update && brew upgrade --cask hyperelastic-calibration
```

## Launch (Python)

1) Install dependencies:
```
pip install -r requirement.txt
```

2) Run the GUI:
```
python qt_app.py
```

## Install (Windows)

1) Download `HyperelasticCalibration-windows.zip` from the latest release.
2) Unzip and run `HyperelasticCalibration.exe`.

## Install (Linux)

1) Download `HyperelasticCalibration-linux.tar.gz` from the latest release.
2) Extract and run `./HyperelasticCalibration/HyperelasticCalibration`.

## Example: Zhan (non-Gaussian)

This example reproduces Fig. 7 from Zhan (JMPS) by fitting the Zhan (non-Gaussian)
model to James 1975 uniaxial tension data, then predicting biaxial tension.

Step 1: Select the experimental data.
![Step 1](assets/examples/zhan-non-gaussian-james-1975/step1.jpg)

Step 2: Configure the material model.
![Step 2](assets/examples/zhan-non-gaussian-james-1975/step2.jpg)

Step 3: Run calibration.
![Step 3](assets/examples/zhan-non-gaussian-james-1975/step3.jpg)

Step 4: Run prediction.
![Step 4](assets/examples/zhan-non-gaussian-james-1975/step4.jpg)
