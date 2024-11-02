from enum import Enum
from loguru import logger

logger.disable(__name__)

# States definition
class SolarFieldState(Enum):
    IDLE = 0
    ACTIVE = 1

class ThermalStorageState(Enum):
    IDLE = 0
    ACTIVE = 1

class SfTsState(Enum):
    IDLE = 0
    HEATING_UP_SF = 1
    SF_HEATING_TS = 2
    RECIRCULATING_TS = 3

SfTsState_with_value = Enum('SfTsState_with_value', {
    f'{state.name}': i
    for i, state in enumerate(SfTsState)
})

class MedVacuumState(Enum):
    OFF = 0
    LOW = 1
    HIGH = 2

class MedState(Enum):
    OFF = 0
    GENERATING_VACUUM = 1
    IDLE = 2
    STARTING_UP = 3
    SHUTTING_DOWN = 4
    ACTIVE = 5


SolarMedState = Enum('SolarMedState', {
    f'sf_{sf_state.name}_ts_{ts_state.name}_med_{med_state.name}': f'{sf_state.value}{ts_state.value}{med_state.value}'
    for sf_state in SolarFieldState
    for ts_state in ThermalStorageState
    for med_state in MedState
})

SolarMedState_with_value = Enum('SolarMedState_with_value', {
    f'{state.name}': i
    for i, state in enumerate(SolarMedState)
})

SupportedSystemsStatesType = MedState | SolarFieldState | ThermalStorageState | SolarMedState | SfTsState