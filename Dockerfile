# This Dockerfile sets the entrypoint to "tail -f /dev/null".
# This is a common practice in Dockerfiles for creating a container that runs indefinitely without executing any commands.
# The purpose of this is to keep the container running so that it can be used as a base image or as a placeholder for other services.

FROM python:3.11
LABEL authors="jmserrano"

# Install project dependencies

WORKDIR /app 
COPY requirements.txt /app 
RUN pip install -r requirements.txt
RUN pip install . # To install the models package in the container

# Copy the project code into the container
COPY . /app

# ENTRYPOINT ["tail", "-f", "/dev/null"]
