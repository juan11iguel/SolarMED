# This Dockerfile sets the entrypoint to "tail -f /dev/null".
# This is a common practice in Dockerfiles for creating a container that runs indefinitely without executing any commands.
# The purpose of this is to keep the container running so that it can be used as a base image or as a placeholder for other services.

FROM python:3.11
LABEL authors="jmserrano"

# Install project dependencies
WORKDIR /app 
COPY requirements.txt /app
RUN pip install -r requirements.txt

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

# Copy the project code into the container
COPY . /app

# ENTRYPOINT ["tail", "-f", "/dev/null"]
