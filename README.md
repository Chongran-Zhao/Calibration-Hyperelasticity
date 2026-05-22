# Hyperelastic Material Calibration

Core Python routines for fitting hyperelastic material models to experimental
data and running prediction workflows.

The previous Streamlit and PySide desktop interfaces have been removed while a
new web UI is being prepared. The current repository is intentionally focused on
the computational layer, packaged datasets, and design direction for the next
interface.

## Install

```bash
git clone https://github.com/Chongran-Zhao/Calibration-Hyperelasticity.git
cd Calibration-Hyperelasticity
pip install -r requirements.txt
```

## Project Structure

- `src/`: material models, kinematics, optimization, plotting, and utilities.
- `data/data.h5`: packaged experimental datasets.
- `assets/examples/`: example screenshots and animations.
- `DESIGN.md`: design direction for the upcoming web interface.

## Example: Zhan (Non-Gaussian)

This example reproduces Fig. 7 from Zhan (JMPS) by fitting the Zhan
(non-Gaussian) model to James 1975 uniaxial tension data, then predicting
biaxial tension.

![Example](assets/examples/example.gif)
