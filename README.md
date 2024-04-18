# Optimal operation of a solar MED plant. Case study: Plataforma Solar de Almería

![](docs/models/attachments/solarMED_optimization-general_diagram.svg)

Repository with source code of a variety of models of a solar field - thermal storage and MED plant combined system located at Plataforma Solar de Almería.
The model of the complete system, called `SolarMED` is contained in the [models_psa package](./models_psa).

- Documentation and results of the modelling of the different components can be found in the [docs](./docs). For better visualization, using [Obsidian](https://obsidian.md) is recommended.
- The model calibration for the different components is contained in [model_calibrations](./model_calibrations).
- To get started, first check the [installation](#installation) section to set up the environment.
- For examples on how to use it check the [examples](#examples) section.


## Package structure

- [models_psa](./models_psa) is the package folder.
- [models_psa.solar_med](models_psa/solar_med.py) is the main module containing the complete model class `SolarMED`.
- [models_psa.solar_field](models_psa/solar_field.py) contains the solar field model code.
- [models_psa.thermal_storage](models_psa/thermal_storage.py) contains the thermal storage model code.
- [models_psa.med](models_psa/med.py) contains the MED model code (will do once updated, right now it's an external package from MATLAB)
- [models_psa.heat_exchanger](models_psa/heat_exchanger.py) contains the heat exchanger model code.
- [models_psa.three_way_valve](models_psa/three_way_valve.py) contains the three-way valve model code.
- [models_psa.validation](models_psa/data_validation.py) contains validation utility functions (within_range_or_min_or_max, etc) and new types definitions (conHotTemperatureType, rangeType, etc)
- [models_psa.power_consumption](models_psa/power_consumption.py) implements models to evaluate different actuators, mainly electricity consumption though maybe these will be refactored to extend the use of the `Actuator` class.
- [models_psa.utils](models_psa/utils) contains different utility functions to process the experimental data.
- [models_psa.metrics](models_psa/metrics) contains different metrics to evaluate the performance of the system (not yet implemented).
- [models_psa.curve_fitting](models_psa/curve_fitting) contains curve fitting functions to calibrate simple fits e.g. fit electricity consumptions and so on.
- [models_psa.calibration](models_psa/calibration) contains the code to perform model parameter calibrations.


## Examples

- Check [test_combined_model](./test_combined_model.ipynb) for an example on how to use the combined model given some test data.
- In [visualize_test](./visualize_test.ipynb) there is an example on how to visualize the data from the system for a given test.
- To evaluate all component and complete system model, and generate a report with the results, check [solarMED_validation_report](./solarMED_validation_report.ipynb).


## Installation

To run the models, a python virtual environment is recommended. In this environment all the python dependencies need to be installed, as well as the MATLAB runtime and MATLAB python package.

### Set up a python virtual environment

Initialize a virtual environment with python 3.11 (requirement of MATLAB runtime) and install the dependencies.

1. Install python 3.11, for example in Fedora:

```bash
sudo dnf install python3.11
```

2. Get directory of python installation (`/usr/bin/python3.X`):

```bash
which python3.X # -> /usr/bin/python3.X
```

3. Create virtual environment with python 3.X:

```bash
virtualenv venv -p /usr/bin/python3.X
```

### Installing dependencies

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Install the dependencies using the `requirements.txt` file in the root of the repository

```bash
pip install -r requirements.txt
```

### Installing MED model dependency

~~To run the combined model **Python 3.9** is required since it's the version that the MATLAB runtime was compiled for.~~
3.11 is supported with the latest compilation of the model.

The MED model is an ANN developed using the MATLAB Neural Network Toolbox. To run it, the MATLAB runtime must be 
installed. The runtime can be downloaded from [here](https://www.mathworks.com/products/compiler/matlab-runtime.html).

1. Create a folder where the runtime will be installed. For example, `$HOME/MATLAB_Runtime`.
2. Use the installer to install the runtime, available in [MED_model_installers](./MED_model_installers).
3. Follow the installer instructions

After that is done, at least in linux, the `LD_LIBRARY_PATH` environment variable needs to be created before being able to use the runtime. To simplfiy the command, an auxiliary `MR` environment variable is created to store the path to the runtime.
This is embedded in the notebooks that make use of the MED model, but it can also be added to the `venv/bin/activate` script to make it available every time the virtual environment is activated.

4. Add the following lines to the end of `venv/bin/activate` script:
```
export MR=$HOME/MATLAB_Runtime
export LD_LIBRARY_PATH=$MR/v911/runtime/glnxa64:$MR/v911/bin/glnxa64:$MR/v911/sys/os/glnxa64:$MR/v911/sys/opengl/lib/glnxa64
```

or within the notebook:
    
```python
os.environ["MR"] = f"{os.environ['HOME']}/MATLAB_Runtime"
MR = os.environ["MR"]
os.environ["LD_LIBRARY_PATH"] = f"{MR}/runtime/glnxa64:{MR}/bin/glnxa64:{MR}/sys/os/glnxa64:{MR}/sys/opengl/lib/glnxa64"
```

5. Install MED model package from `setup.py` file:

```bash
pip install /path/to/matlab/package/directory
```

6. Test that the MED model package is installed correctly by running test script:

```bash
python samples/MED_modelSample1.py
```

## Pending tasks

High priority:
- [ ] Recalibrate thermal storage model, once the experimental data is exported including the necessary variables to estimate qhx_s.
- [x] Find a more robust alternative to obtain the flow from the solar field than inverting the model (implement an internal control loop for the outlet temperature?)
- [ ] Calibrate input signal - flow of qhx_s once the experimental data is exported including the necessary variables to estimate qhx_s.
- [ ] Extend MED model to accept the new operating modes (generating vacuum, starting up, shutting down, idle)
- [ ] Integrate new `SolarMED` states.
- [ ] Add electrical consumption of solar field and thermal storage pump. Pending of physical modifications in the experimental facility

UFSC collaboration towards alternative SolarMED configurations:
- [ ] [JD] Make thermal storage model more flexible by allowing external inputs and extractions to be performed in arbitrary control volumes.
- [ ] [JD] Make solar field model more flexible by allowing an alternative configuration (parallel -> series)
- [ ] [D] Integrate exergy calculations in the component models.
- [ ] [D?JD?] Use MED first-principles model and modify it to make it more flexible by accepting external heat sources in any effect.
- [ ] [JM] Update `SolarMED` so it can make use of the flexible models and stores the exergy evaluation results.

Low priority:
- [ ] Move auxiliary calculations in `SolarMED` (powers, metrics, etc) to the module of the specific component in its own function instead of having them in the `step` method.
- [ ] When `resolution_mode` is `'simple'`, the water physical properties should also be simplified in the models
- [ ] Replace MED model with Python implementation from KWR.

Longer term: 
- [ ] Implement alternative way of exporting experimental data from the plant instead of relying on manually exported txts from LABview.

Maybe more can be found in [Issues](docs/Issues.md) and [Pending tasks](docs/Pending%20tasks.md).