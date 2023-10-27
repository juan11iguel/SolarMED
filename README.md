# Optimal operation of a solar MED plant. Case study: Plataforma Solar de Almería

Repository with source code of a variety of models of a solar field - thermal storage and MED plant combined system located at Plataforma Solar de Almería.

## Pending tasks

![Pending tasks](docs/Pending%20tasks.md)

## Installation

To run the combined model **Python 3.9** is required since it's the version that the MATLAB runtime was compiled for.

### Installing MED model dependency

The MED model is an ANN developed using the MATLAB Neural Network Toolbox. To run it, the MATLAB runtime must be 
installed. The runtime can be downloaded from [here](https://www.mathworks.com/products/compiler/matlab-runtime.html).

1. Create a folder where the runtime will be installed. For example, `$HOME/MATLAB_Runtime`.
2. Use the installer to install the runtime, available in [MED_model_installers](./MED_model_installers).
3. Follow the installer instructions, after it's done, at least in linux two environment variables need to be created before being able to use the runtime.

```bash
export MR=$HOME/MATLAB_Runtime
export LD_LIBRARY_PATH=$MR/v911/runtime/glnxa64:$MR/v911/bin/glnxa64:$MR/v911/sys/os/glnxa64:$MR/v911/sys/opengl/lib/glnxa64
```

### Setting up the python environment

Initialize a virtual environment with python 3.9 (requirement of MATLAB runtime) and install the dependencies.

1. Install python 3.9, for example in Fedora:

```bash
sudo dnf install python3.9
```

2. Get directory of python 3.9 installation (`/usr/bin/python3.9`):

```bash
which python3.9 # -> /usr/bin/python3.9
```

3. Create virtual environment with python 3.9:

```bash
virtualenv venv -p /usr/bin/python3.9
```

5. Install MED model package from `setup.py` file:

```bash
cd $MR && python setup.py install --prefix$MR/application/venv
```

6. Test that the MED model package is installed correctly by running test script:

```bash
python samples/MED_modelSample1.py
```

## Examples

Check [test_combined_model](./test_combined_model.ipynb) for examples of how to use the combined model.


