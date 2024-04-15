from enum import Enum

# States definition
class SolarFieldState(Enum):
    IDLE = 0
    ACTIVE = 1


class ThermalStorageState(Enum):
    IDLE = 0
    ACTIVE = 1


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


# More descriptive manually typed names
class SF_TS_State(Enum):
    IDLE = '00'
    RECIRCULATING_TS = '01'
    HEATING_UP_SF = '10'
    SF_HEATING_TS = '11'


SolarMED_State = Enum('SolarMED_State', {
    f'sf_{sf_state.name}-ts_{ts_state.name}-med_{med_state.name}': f'{sf_state.value}{ts_state.value}{med_state.value}'
    for sf_state in SolarFieldState
    for ts_state in ThermalStorageState
    for med_state in MedState
})


