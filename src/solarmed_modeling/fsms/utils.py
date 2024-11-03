from enum import Enum
from typing import Literal
from . import MedState, SfTsState, SolarMedState
from .med import MedFsm
from .sfts import SolarFieldWithThermalStorageFsm

SupportedFSMTypes = SolarFieldWithThermalStorageFsm | MedFsm

class SupportedSubsystemsStatesMapping(Enum):
    MED = MedState
    SFTS = SfTsState
class SupportedSubsystemsFsmsMapping(Enum):
    MED = MedFsm
    SFTS = SolarFieldWithThermalStorageFsm
    
SupportedSubsystemsLiteral = Literal['MED', 'SFTS']

class SupportedSystemsStatesMapping(Enum):
    MED = MedState
    SFTS = SfTsState
    SolarMED = SolarMedState
    
SupportedSystemsLiteral = Literal['MED', 'SFTS', 'SolarMED']