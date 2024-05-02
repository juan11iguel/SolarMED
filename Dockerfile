FROM ubuntu:latest
LABEL authors="jmserrano"

# Should start a container with Python 3.11
# Micro editor
# htop command
# zsh?
# Requirements.txt de este repositorio e instalar todas las dependencias

# Después sea un contenedor que no haga nada pora poder conectarse en remoto con VSCode
# Y asi ejecutar scripts del repositorio que son pesados y no quiero ejecutar en el portátil

ENTRYPOINT ["top", "-b"]