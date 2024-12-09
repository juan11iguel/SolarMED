from enum import Enum
from typing import Literal
from . import MedState, SfTsState, SolarMedState, SolarFieldState, ThermalStorageState, FsmInputs
from .med import MedFsm, FsmInputs as MedFsmInputs
from .sfts import (SolarFieldWithThermalStorageFsm, get_sfts_state,
                   FsmInputs as SfTsFsmInputs)

SupportedFSMTypes = SolarFieldWithThermalStorageFsm | MedFsm
SupportedSubsystemsLiteral = Literal['MED', 'SFTS']
SupportedSystemsStatesType = MedState | SolarFieldState | ThermalStorageState | SolarMedState | SfTsState    
SupportedSystemsLiteral = Literal['MED', 'SFTS', 'SolarMED']

class SupportedSubsystemsStatesMapping(Enum):
    """ Mapping of system key to corresponding state Enum """
    MED = MedState
    SFTS = SfTsState

class SupportedSubsystemsFsmsMapping(Enum):
    """ Mapping of system key to corresponding FSM class """
    MED = MedFsm
    SFTS = SolarFieldWithThermalStorageFsm

class FsmInputsMapping(Enum):
    """ Mapping of system key to corresponding inputs dataclass """
    MED = MedFsmInputs
    SFTS = SfTsFsmInputs
    
class SupportedSystemsStatesMapping(Enum):
    """ Mapping of system key to corresponding state Enum """
    MED = MedState
    SFTS = SfTsState
    SolarMED = SolarMedState


def convert_to(
    state: str | int | Enum, 
    state_cls: SupportedSystemsStatesType, 
    return_format: Literal["enum", "name", "value"] = "enum"
) -> Enum | str | int:
    
    if isinstance(state, str):
        output = getattr(state_cls, state) 
    elif isinstance(state, int | float):
        output = state_cls(int(state))
    elif isinstance(state, Enum):
        output = state
    else:
        raise ValueError(f"`state` should be either a str or an int, not {type(state)}")
    
    if return_format == "enum":
        return output
    elif return_format == "name":
        return output.name
    elif return_format == "value":
        return output.value
    else:
        raise ValueError(f"`return_format` should be one of enum, str or int. Not {return_format}")
    
    
def get_solarmed_individual_states(
    input_state: int | SolarMedState | str, 
    return_format = Literal["enum", "value", "name"]
) -> tuple[SfTsState | str | int, MedState | str | int]:
    
    if isinstance(input_state, int):
        input_state: SolarMedState = SolarMedState(input_state)
    elif isinstance(input_state, str):
        input_state: SolarMedState = SolarMedState[input_state]
    
    sfts_state = get_sfts_state(sf_state = int(input_state.value[0]), 
                                ts_state = int(input_state.value[1]))
    med_state = MedState( int(input_state.value[2]) )
    
    output = (sfts_state, med_state)
        
    if return_format == "enum":
        return output
    elif return_format == "value":
        return tuple([state.value for state in output])
    elif return_format == "name":
        return tuple([state.name for state in output])