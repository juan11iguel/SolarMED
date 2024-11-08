from typing import Literal
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger
import math

from . import BaseFsm, SfTsState, SolarFieldState, ThermalStorageState
from solarmed_modeling.solar_field import FixedModelParameters as SfFixedModParams
from solarmed_modeling.thermal_storage import FixedModelParameters as TsFixedModParams

@dataclass
class FsmStartupConditions:
    """
    Startup conditions for the Finite State Machines (FSM)
    """ 
    # Solar field
    qsf: float = SfFixedModParams().qsf_min  # Solar field flow rate (m³/h)
    
    # Thermal storage
    qts_src: float = TsFixedModParams().qts_src_min  # Thermal storage flow rate (m³/h)
    
@dataclass
class FsmShutdownConditions:
    """
    Shutdown conditions for the Finite State Machines (FSM)
    """ 
    # Solar field
    qsf: float = 0.0 # Solar field flow rate (m³/h)
    
    # Thermal storage
    qts_src: float = 0.0 # Thermal storage flow rate (m³/h)
    
@dataclass
class FsmParameters:
    recirculating_ts_enabled: bool = True
    recirculating_ts_cooldown_time: int = 9999*3600 # Time to wait before activating state recirculating ts (seconds)
    idle_cooldown_time: int = 3 # Time to wait before activating state idle (seconds)
    
    startup_conditions: FsmStartupConditions = field(default_factory=lambda: FsmStartupConditions())
    shutdown_conditions: FsmShutdownConditions = field(default_factory=lambda: FsmShutdownConditions())

@dataclass
class FsmInternalState:
    """
    Internal state of the Finite State Machine (FSM)
    
    TLDR: Avoid setting the internal state manually.
    
    Notes:
    - No checks are done in the internal state, if an incorrect start_sample value is set,
    it does not matter for example when activating that transitions since it will rewrite the value.
    However, for example if the initial state is set to IDLE, while vacuum_generated is False, it
    migth cause unexpected behavior.
    """
    
    idle_cooldown_done: bool = True
    idle_cooldown_elapsed_samples: int = 0
    recirculating_ts_cooldown_done: bool = False
    recirculating_ts_cooldown_elapsed_samples: int = 0
    
    
def get_sfts_state(sf_state: int | SolarFieldState | str, ts_state: int | ThermalStorageState | str,
                   return_format: Literal["name", "value", "enum"] = "enum") -> SfTsState | str | int:
    
    if isinstance(sf_state, int):
        sf_state = SolarFieldState(sf_state)
    elif isinstance(sf_state, str):
        sf_state = SolarFieldState[sf_state]
    if isinstance(ts_state, int):
        ts_state = ThermalStorageState(ts_state)
    elif isinstance(ts_state, str):
        ts_state = ThermalStorageState[ts_state]
    
    if sf_state.value == SolarFieldState.IDLE.value and ts_state.value == ThermalStorageState.IDLE.value:
        output = SfTsState.IDLE
    elif sf_state.value == SolarFieldState.ACTIVE.value and ts_state.value == ThermalStorageState.IDLE.value:
        output = SfTsState.HEATING_UP_SF
    elif sf_state.value == SolarFieldState.IDLE.value and ts_state.value == ThermalStorageState.ACTIVE.value:
        output = SfTsState.RECIRCULATING_TS
    else:
        output = SfTsState.SF_HEATING_TS
        
    if return_format == "enum":
        return output
    elif return_format == "value":
        return output.value
    elif return_format == "name":
        return output.name
    
def get_sf_ts_individual_states(
    sfts_state: int | SfTsState | str, 
    return_format = Literal["enum", "value", "name"]
) -> tuple[SolarFieldState | str | int, ThermalStorageState | str | int]:
    
    if isinstance(sfts_state, int):
        sfts_state = SfTsState(sfts_state)
    elif isinstance(sfts_state, str):
        sfts_state = SfTsState[sfts_state]
    
    if sfts_state.value == SfTsState.IDLE.value:
        output = ( SolarFieldState.IDLE, ThermalStorageState.IDLE )
    elif sfts_state.value == SfTsState.HEATING_UP_SF.value:
        output = ( SolarFieldState.ACTIVE, ThermalStorageState.IDLE )
    elif sfts_state.value == SfTsState.SF_HEATING_TS.value:
        output = ( SolarFieldState.ACTIVE, ThermalStorageState.ACTIVE )
    elif sfts_state.value == SfTsState.RECIRCULATING_TS.value:
        output = ( SolarFieldState.IDLE, ThermalStorageState.ACTIVE )
    else:
        raise ValueError(f"Invalid SfTsState: {sfts_state}")
        
    if return_format == "enum":
        return output
    elif return_format == "value":
        return tuple([state.value for state in output])
    elif return_format == "name":
        return tuple([state.name for state in output])
    
    
class SolarFieldWithThermalStorageFsm(BaseFsm):

    """
    Finite State Machine for the Solar Field with Thermal Storage (SF-TS) unit.
    """

    params: FsmParameters # to have type hints
    internal_state: FsmInternalState # to have type hints
    _state_type: SfTsState = SfTsState # State type
    _cooldown_callbacks: list[str] = ['is_idle_cooldown_done', 'is_recirculating_ts_cooldown_done']

    def __init__(
            self, 
            sample_time: int, 
            name: str = "SFTS_FSM",
            initial_state: SfTsState = SfTsState.IDLE,
            current_sample: int = 0,
            params: FsmParameters = FsmParameters(),
            internal_state: FsmInternalState = FsmInternalState(),
            
            # Inputs / Decision variables (optional, use to set prior inputs)
            qts_src: float = None, 
            qsf: float = None
    ) -> None:

        # Call parent constructor
        super().__init__(name=name, initial_state=initial_state, sample_time=sample_time,
                         current_sample=current_sample, internal_state=internal_state,
                         params=params)
        
        # Convert duration times to samples
        self.recirculating_ts_cooldown_samples: int = math.ceil(self.params.recirculating_ts_cooldown_time / self.sample_time)
        self.idle_cooldown_samples: int = math.ceil(self.params.idle_cooldown_time / self.sample_time)

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.qts_src: float = qts_src
        self.qsf: float = qsf

        inputs_array: np.ndarray[float] = self.update_inputs_array()
        self.inputs_array_prior: np.ndarray[float] = inputs_array

        # States
        st = self._state_type # alias

        self.machine.add_state(st.IDLE, on_enter=['stop_pumps', 'set_idle_cooldown_start'])
        if self.params.recirculating_ts_enabled:
            self.machine.add_state(st.RECIRCULATING_TS, on_exit=['set_recirculating_ts_cooldown_start'])
        self.machine.add_state(st.HEATING_UP_SF, on_enter=['stop_pump_ts'])
        self.machine.add_state(st.SF_HEATING_TS)

        # Transitions
        if self.params.recirculating_ts_enabled:
            self.machine.add_transition('start_recirculating_ts',  source=st.IDLE, dest=st.RECIRCULATING_TS, 
                                        conditions=['is_pump_ts_on', 'is_recirculating_ts_cooldown_done', 'is_idle_cooldown_done'], unless=['is_pump_sf_on'])
            self.machine.add_transition('stop_recirculating_ts', source=st.RECIRCULATING_TS, dest=st.IDLE, 
                                        unless=['is_pump_ts_on'])
        self.machine.add_transition('start_recirculating_sf', source=st.IDLE, dest=st.HEATING_UP_SF, 
                                    conditions=['is_pump_sf_on', 'is_idle_cooldown_done'])
        self.machine.add_transition('stop_recirculating_sf', source=st.HEATING_UP_SF, dest=st.IDLE, 
                                    unless=['is_pump_sf_on'])
        self.machine.add_transition('start_sf_heating_ts', source=st.HEATING_UP_SF, dest=st.SF_HEATING_TS, 
                                    conditions=['is_pump_ts_on', 'is_pump_sf_on'])
        self.machine.add_transition('stop_sf_heating_ts', source=st.SF_HEATING_TS, dest=st.HEATING_UP_SF, 
                                    conditions=['is_pump_sf_on'], unless=['is_pump_ts_on'])
        self.machine.add_transition('shutdown', source=st.SF_HEATING_TS, dest=st.IDLE, 
                                    unless=['is_pump_sf_on', 'is_pump_ts_on'])

        # Additional
        self.customize_fsm_style()

    def get_inputs(self, format: Literal['array', 'dict'] = 'array') -> np.ndarray[float] | dict:

        super().get_inputs(format=format) # Just to check if the format is valid

        if format == 'array':
            # When the array format is used, all variables necessarily need to be parsed as floats

            return np.array([self.qts_src, self.qsf], dtype=float)

        elif format == 'dict':
            # In the dict format, each variable  can have its own type
            return {
                'qts_src': self.qts_src,
                'qsf': self.qsf,
            }

    # State machine actions - callbacks of states and transitions
    def stop_pump_ts(self, *args) -> None:
        """ Stop the pump for the thermal storage """
        self.qts_src = 0

    def stop_pump_sf(self, *args) -> None:
        """ Stop the pump for the solar field """
        self.qsf = 0

    def stop_pumps(self, *args) -> None:
        """ Stop both pumps """
        logger.info(f"[{self.name}] Stopping pumps")
        self.stop_pump_ts()
        self.stop_pump_sf()
        
    def set_idle_cooldown_start(self, *args) -> None:
        """ Set the idle cooldown start """
        self.internal_state.idle_cooldown_done = False
        self.internal_state.idle_cooldown_elapsed_samples = 0
        logger.info(f"[{self.name}] Idle cooldown started")
        
    def set_recirculating_ts_cooldown_start(self, *args) -> None:
        """ Set the recirculating thermal storage cooldown start """
        self.internal_state.recirculating_ts_cooldown_done = False
        self.internal_state.recirculating_ts_cooldown_elapsed_samples = 0
        logger.info(f"[{self.name}] Recirculating TS cooldown started")

    # State machine transition conditions
    # Solar field
    def is_pump_sf_on(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Check if the pump for the solar field is on """

        if return_valid_inputs:
            return dict(qsf = 1.0)
        elif return_invalid_inputs:
            return dict(qsf = 0.0)

        output: bool = False
        if self.qsf is not None:
            output = self.qsf > 0

        # Prioritize Tsf_out, but if it has no valid value, check qsf
        # if output == False and self.qsf is not None:
        #     output = self.qsf > 0

        return output

    # Thermal storage
    def is_pump_ts_on(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Check if the pump for the thermal storage is on """

        if return_valid_inputs:
            return dict(qts_src = 1.0)
        elif return_invalid_inputs:
            return dict(qts_src = 0.0)

        return self.qts_src > 0 if self.qts_src is not None else False
    
    def is_idle_cooldown_done(self, *args, ) -> bool:
        """ Check if the idle cooldown is done """
        if self.internal_state.idle_cooldown_done:
            return True
        
        cooldown_done, self.internal_state.idle_cooldown_elapsed_samples = self.check_elapsed_samples(
            elapsed_samples=self.internal_state.idle_cooldown_elapsed_samples,
            samples_duration=self.idle_cooldown_samples,
            msg = "idle cooldown"
        )
        
        self.internal_state.idle_cooldown_done = cooldown_done
        
        return cooldown_done
    
    def is_recirculating_ts_cooldown_done(self, *args, ) -> bool:
        """ Check if the recirculating_ts cooldown is done """
        if self.internal_state.recirculating_ts_cooldown_done:
            return True
        
        cooldown_done, self.internal_state.recirculating_ts_cooldown_elapsed_samples = self.check_elapsed_samples(
            elapsed_samples=self.internal_state.recirculating_ts_cooldown_elapsed_samples,
            samples_duration=self.recirculating_ts_cooldown_samples,
            msg = "recirculating_ts cooldown"
        )
        
        self.internal_state.recirculating_ts_cooldown_done = cooldown_done
        
        return cooldown_done


    def step(self, qsf: float, qts_src: float, 
             return_df: bool = False, df: pd.DataFrame = None) -> None | pd.DataFrame:
        
        """ Move the state machine one step forward """

        super().step()

        # Inputs validation (would be done by Pydantic), here just update the values
        self.qsf = qsf
        self.qts_src = qts_src

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.update_inputs_array()

        transition: callable = self.get_next_valid_transition(prior_inputs=self.inputs_array_prior,
                                                    current_inputs=self.inputs_array)
        if transition is not None:
            transition()

        # Save prior inputs
        self.inputs_array_prior = self.inputs_array
        
        # Return updated dataframe
        if return_df:
            return self.to_dataframe(df)

    def to_dataframe(self, df: pd.DataFrame = None, states_format: Literal["enum", "value", "name"] = "enum") -> pd.DataFrame:
        # Return some of the internal variables as a dataframe
        # the state as en Enum?str?, the inputs, the consumptions
        df = super().to_dataframe(df, states_format)
        internal_state_dict = self.get_internal_state_dict()
        
        # There should be some internal method to set the fields. 
        # - Inputs similar to internal states and params should be an `inputs` dataclass
        # - And then just have an attribute with a list of the fields to be included in the dataframe
        # It does not make sense on every child class to be including the convert_to and current samples
        
        data = pd.DataFrame({
            'state': self.convert_to(self.state, state_cls = self._state_type, return_format = states_format),
            'current_sample': self.current_sample,
            'qsf': self.qsf,
            'qts_src': self.qts_src,
            **internal_state_dict
        }, index=[0])

        df = pd.concat([df, data], ignore_index=True)

        return df