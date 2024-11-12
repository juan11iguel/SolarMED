# Optimal operation of a solar MED plant. Case study: Plataforma Solar de Almería

![](docs/models/attachments/solarMED_optimization-general_diagram.svg)

Repository with source code of a variety of models of a solar field - thermal storage and MED plant combined system located at Plataforma Solar de Almería.
The model of the complete system, called `SolarMED` is contained in the [models_psa package](./solarMED_modeling).

- Documentation and results of the modelling of the different components can be found in the [docs](./docs). For better visualization, using [Obsidian](https://obsidian.md) is recommended.
- The model calibration for the different components is contained in [model_calibrations](./model_calibrations).
- To get started, first check the [installation](#installation) section to set up the environment.
- For examples on how to use it check the [examples](#examples) section.


# TODOs

- [x] Re-factor packaging using uv
- [x] Add .devcontainer
- [ ] Add support for notebook deployment. WIP reverse-proxy giving trouble, works on port
- [x] Model refactor with benchmark implementation
- [x] Integrate new models and last practises to combined model
- [ ] Simulate two consecutive operation days
- [x] In FSMs, add cooldown times for particular state changes
- [x] Add support for partial initialization of FSMs
- [ ] As with FsmParameters and FsmInternalState, use a dataclass for FsmInputs. This simplifies FSM code but also path_explorer from SolarMED-optimization


# Benchmarks

The script [test_model.py](./scripts/test_model.py) is used to benchmark the model performance in different scenarios. The script is run with the following command:

```bash
uv run scripts/test_model.py
```

At the end it prints some performance metrics of the model (error metrics), and the time it took to run the model.
Log here the results of the benchmarks:

|  | Date test | Date evaluated | Scenario | Total time (s) | N iterations | Time/it. (s) | RMSE | MAE |
|--|------|------|----------|-----------|-------|------|---|---|
| 1 | 20230703 | 20240923   | Standard configuration. VM ai.psa.es. Debugging | 54.40 | 660 | 0.0824 | 0.78 | 0.61 |
| 2 | 20230703 | 20240923   | Standard configuration. Asus Rog Flow X13 | 29.89 | 660 | 0.0453 | 0.78 | 0.61 |
| 3 | 20230703 | 20240923   | Standard configuration. VM ai.psa.es. | 28.97 | 660 | 0.0439 | 0.78 | 0.61 |
| 4 | 20230703 | 20240923   | Standard configuration. VM ai.psa.es. Tras llorarle a Joaquín para más potensia | 29.27 | 660 | 0.0443 | 0.78 | 0.61 |
| 5 | 20230703 | 20240924   | Standard configuration. VM ai.psa.es. After some changes (not optimization focused) | 25.90 | 660 | 0.0392 | 0.78 | 0.61 |



> [!NOTE] 
> Comparison between runs where test date or sample rate (=number for iterations) changes, are not valid for total time and error metrics. Time per iteration (`Time/it.`) is a better "universal" execution time metric to compare different runs.

- "Standard configuration" stands for model using both fsms and evaluation component models, prior to applying any optimizations (such as assumming water properties as constant)


## Package structure

- [models_psa](./src/solarmed_modeling) is the package folder.
- [models_psa.solar_med](./src/solarmed_modeling/solar_med.py) is the main module containing the complete model class `SolarMED`.
- [models_psa.solar_field](./src/solarmed_modeling/solar_field.py) contains the solar field model code.
- [models_psa.thermal_storage](./src/solarmed_modeling/thermal_storage.py) contains the thermal storage model code.
- [models_psa.med](./src/solarmed_modeling/med.py) contains the MED model code (will do once updated, right now it's an external package from MATLAB)
- [models_psa.heat_exchanger](./src/solarmed_modeling/heat_exchanger.py) contains the heat exchanger model code.
- [models_psa.three_way_valve](./src/solarmed_modeling/three_way_valve.py) contains the three-way valve model code.
- [models_psa.validation](./src/solarmed_modeling/data_validation.py) contains validation utility functions (within_range_or_min_or_max, etc) and new types definitions (conHotTemperatureType, rangeType, etc)
- [models_psa.power_consumption](./src/solarmed_modeling/power_consumption.py) implements models to evaluate different actuators, mainly electricity consumption though maybe these will be refactored to extend the use of the `Actuator` class.
- [models_psa.utils](./src/solarmed_modeling/utils) contains different utility functions to process the experimental data.
- [models_psa.metrics](./src/solarmed_modeling/metrics) contains different metrics to evaluate the performance of the system (not yet implemented).
- [models_psa.curve_fitting](./src/solarmed_modeling/curve_fitting) contains curve fitting functions to calibrate simple fits e.g. fit electricity consumptions and so on.
- [models_psa.calibration](./src/solarmed_modeling/calibration) contains the code to perform model parameter calibrations.


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

## Docker

A Dockerfile is provided to run the models in a container. The container is based on the `python:3.11` image and installs the necessary dependencies to run the models. A prebuilt image is available in this repository with a different tag for the different branches. To run the container, use the provided `docker-compose.yml` file with the command:

```bash
docker-compose up -d
```

By default the compose file will run a Jupyter server that can be accessed from a browser and this way run the different notebooks. To access it, copy the URL provided in the logs of the container:

```bash
docker logs solarmed-modeling
```

![alt text](docs/attachments/jupyter.png)

Remember to replace the port with the one defined in the compose file as well as the IP address to the one where the notebook is being run from.

The jupyter server will keep the container alive so it can be accessed at any point and directly run the different scripts. However if the Jupyter server overhead is not desired, another option is to uncomment the `command` line in the compose file, and use an alternative entrypoint such as `tail` that will not run anything but keep the container alive.


## Pending tasks

High priority:
- [ ] Recalibrate thermal storage model, once the experimental data is exported including the necessary variables to estimate qhx_s.
- [x] Find a more robust alternative to obtain the flow from the solar field than inverting the model (implement an internal control loop for the outlet temperature?)
- [ ] Calibrate input signal - flow of qhx_s once the experimental data is exported including the necessary variables to estimate qhx_s.
- [x] Extend MED model to accept the new operating modes (generating vacuum, starting up, shutting down, idle)
- [x] Integrate new `SolarMED` states.
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
- [ ] Replace MED model with Python implementation from KWR. Attempted, having trouble importing the model

Longer term: 
- [ ] Implement alternative way of exporting experimental data from the plant instead of relying on manually exported txts from LABview.

Maybe more can be found in [Issues](docs/Issues.md) and [Pending tasks](docs/Pending%20tasks.md).
