# This Dockerfile sets the entrypoint to "tail -f /dev/null".
# This is a common practice in Dockerfiles for creating a container that runs indefinitely without executing any commands.
# The purpose of this is to keep the container running so that it can be used as a base image or as a placeholder for other services.

FROM ubuntu:24.04
LABEL authors="Juan Miguel Serrano, Alejandro Clavera"

WORKDIR /app

# Install curl, git, python3, xclip, micro
RUN apt-get update && apt-get install -y
RUN apt-get update && apt-get install -y build-essential curl curl git python3 xclip micro wget

# Install zsh
# Install oh-my-zsh with customized theme
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -p git
RUN echo '[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh' >> .zshrc
RUN wget https://raw.githubusercontent.com/chbroecker/dotfiles/main/zsh/.p10k.zsh -O .p10k.zsh

# Install uv (alternative to pip, https://docs.astral.sh/uv/guides/integration/docker/#available-images)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.cargo/bin/:$PATH"

# Install matlab runtime
ENV MR=/app/MATLAB_Runtime
RUN mkdir MATLAB_Runtime
COPY MED_model_installers /app/MATLAB_Runtime
RUN chmod +rx ${MR}/MED_model_web_python_python_311.install
RUN ${MR}/MED_model_web_python_python_311.install \
    -agreeToLicense yes \
    -destinationFolder ${MR} \
    -mode silent
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
$MR/R2023b/runtime/glnxa64:\
$MR/R2023b/bin/glnxa64:\
$MR/R2023b/sys/os/glnxa64:\
$MR/R2023b/sys/opengl/lib/glnxa64
RUN cd $MR/application/ && python setup.py install

# Install project dependencies
COPY . .
RUN uv sync --frozen

# Set the default command to run when starting the container
# Start jupyter notebook
# CMD [".venv/bin/jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
