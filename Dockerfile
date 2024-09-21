# syntax = devthefuture/dockerfile-x

INCLUDE ./Dockerfile.base

# Copy the rest of the project
COPY . .

# Set the default command to run when starting the container
# Start jupyter notebook
CMD [".venv/bin/jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
