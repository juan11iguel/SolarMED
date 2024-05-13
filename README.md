# SolarMED optimization

![Process diagram](docs/attachments/solarMED_optimization-general_diagram.svg)

Software package that (will) implement(s) several optimization strategies for the `SolarMED` process.

The `SolarMED` process is a solar-powered desalination process that uses a multi-effect distillation (MED) system. 
The model of the complete process and its individual components is implemented in the [solarMED-modeling](https://github.com/juan11iguel/solarMED-modeling) package.


## Files structure

- [docs](./docs) - Documentation files
- [solarMED_optimization](./solarMED_optimization) - Python package with the optimization strategies implementation
- `path_explorer_*.ipynb` - Notebooks with the implementation of the path explorer for the different subsystems and complete process.
- [RS_one_complete_problem.ipynb](./RS_one_complete_problem.ipynb) - Notebook with the implementation of the *one complete problem* Resolution Strategy (RS), see [docs](./docs/Resolution%20strategy.md#one-complete-problem) for more details.
- [RS_n_reduced_problems.ipynb](./RS_n_reduced_problems.ipynb) - Notebook with the implementation of the *n reduced problems* resolution strategy, see [docs](./docs/Resolution%20strategy.md#n-predefined-path-problems) for more details.

There are is (will) also available a Docker image in this repository [packages]() that allows to create a container 
using the [docker-compose](./docker-compose.yml) file. It sets up an environment to replicate the results available
in the [results](./results) folder.


## Reproducing results

A docker image is available in the [packages](./packages) folder. But it can also be manually built by runing the following command:

```bash
docker build . -t solarmed_optimization
```

Then, to run the container, use the [docer-compose](./docker-compose.yml) file:

```bash
docker-compose up
```

Depeding on the `command` that is enabled in the [docker-compose](./docker-compose.yml) file, the container will run a
Jupyter server where the different notebooks can be run to reproduce the results, or it will just stay alive so it can
be remotely accessed [[1]](#additional-information). 



## Additional information

- [1] Develop on a remote Docker host. VSCode documentation. [Link](https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host#_connect-using-docker-contexts).