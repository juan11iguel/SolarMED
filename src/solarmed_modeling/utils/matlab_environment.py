import os
from typing import Literal

def set_matlab_environment(operating_system: Literal['linux'] = 'linux') -> None:

    if operating_system == 'linux':
        MR = f"{os.environ['HOME']}/MATLAB/R2023b"
        os.environ["MR"] = MR
        os.environ["LD_LIBRARY_PATH"] = \
            f"{MR}/runtime/glnxa64:{MR}/bin/glnxa64:{MR}/sys/os/glnxa64:{MR}/sys/opengl/lib/glnxa64"

    else:
        raise NotImplementedError(f"Not supported {operating_system} operating system")