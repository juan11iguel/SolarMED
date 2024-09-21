import os
from typing import Literal
from loguru import logger

def set_matlab_environment(environment: Literal['linux-host', 'container'] = None) -> None:

    if environment is None:
        logger.debug("Not setting MATLAB environment variables (LD_LIBRARY_PATH), presuming they are already set")
        
        return

    if environment == 'linux-host':
        MR = f"{os.environ['HOME']}/MATLAB/R2023b"
        os.environ["MR"] = MR
        os.environ["LD_LIBRARY_PATH"] = \
            f"{MR}/runtime/glnxa64:{MR}/bin/glnxa64:{MR}/sys/os/glnxa64:{MR}/sys/opengl/lib/glnxa64"
    elif environment == 'container':
        MR = "/app/MATLAB_Runtime"
        os.environ["LD_LIBRARY_PATH"] = \
            f"{MR}/runtime/glnxa64:{MR}/bin/glnxa64:{MR}/sys/os/glnxa64:{MR}/sys/opengl/lib/glnxa64"

    else:
        raise NotImplementedError(f"Not supported {environment} environment, options are: 'linux-host', 'container'")
    
    logger.info(f"MATLAB LD_LIBRARY_PATH path set to: {os.getenv('LD_LIBRARY_PATH')} for {environment} environment")