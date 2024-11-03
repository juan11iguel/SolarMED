from typing import Literal
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger

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
    startup_conditions: FsmStartupConditions = field(default_factory=lambda: FsmStartupConditions())
    shutdown_conditions: FsmShutdownConditions = field(default_factory=lambda: FsmShutdownConditions())

    
def get_sfts_state(sf_state: int | SolarFieldState, ts_state: int | ThermalStorageState) -> SfTsState:
    
    if isinstance(sf_state, int):
        sf_state = SolarFieldState(sf_state)
    if isinstance(ts_state, int):
        ts_state = ThermalStorageState(ts_state)
    
    if sf_state == SolarFieldState.IDLE and ts_state == ThermalStorageState.IDLE:
        return SfTsState.IDLE
    elif sf_state == SolarFieldState.ACTIVE and ts_state == ThermalStorageState.IDLE:
        return SfTsState.HEATING_UP_SF
    elif sf_state == SolarFieldState.IDLE and ts_state == ThermalStorageState.ACTIVE:
        return SfTsState.RECIRCULATING_TS
    else:
        return SfTsState.SF_HEATING_TS
    
def get_sf_ts_individual_states(sfts_state: int | SfTsState) -> tuple[SolarFieldState, ThermalStorageState]:
    
    if isinstance(sfts_state, int):
        sfts_state = SfTsState(sfts_state)
    
    if sfts_state == SfTsState.IDLE:
        return SolarFieldState.IDLE, ThermalStorageState.IDLE
    
    elif sfts_state == SfTsState.HEATING_UP_SF:
        return SolarFieldState.ACTIVE, ThermalStorageState.IDLE
    
    elif sfts_state == SfTsState.SF_HEATING_TS:
        return SolarFieldState.ACTIVE, ThermalStorageState.ACTIVE
    
    elif sfts_state == SfTsState.RECIRCULATING_TS:
        return SolarFieldState.IDLE, ThermalStorageState.ACTIVE
    
    
class SolarFieldWithThermalStorageFsm(BaseFsm):

    """
    Finite State Machine for the Solar Field with Thermal Storage (SF-TS) unit.
    """

    # sample_rate: int = 1  # seconds
    # current_sample = 0

    # State type
    _state_type: SfTsState = SfTsState


    def __init__(
            self,
            sample_time: int, name: str = "SF-TS_FSM", initial_state: SfTsState = SfTsState.IDLE,

            # Inputs / Decision variables (Optional)
            qts_src: float = None, qsf: float = None
    ) -> None:

        # Call parent constructor
        super().__init__(name, initial_state, sample_time)

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.qts_src: float = qts_src
        self.qsf: float = qsf

        inputs_array: np.ndarray[float] = self.update_inputs_array()
        self.inputs_array_prior: np.ndarray[float] = inputs_array

        # States
        st = self._state_type # alias

        self.machine.add_state(st.IDLE, on_enter=['stop_pumps'])
        self.machine.add_state(st.RECIRCULATING_TS)
        self.machine.add_state(st.HEATING_UP_SF, on_enter=['stop_pump_ts'])
        self.machine.add_state(st.SF_HEATING_TS)

        # Transitions
        self.machine.add_transition('start_recirculating_ts',  source=st.IDLE, dest=st.RECIRCULATING_TS, conditions=['is_pump_ts_on'], unless=['is_pump_sf_on'])
        self.machine.add_transition('stop_recirculating_ts', source=st.RECIRCULATING_TS, dest=st.IDLE, unless=['is_pump_ts_on'])

        self.machine.add_transition('start_recirculating_sf', source=st.IDLE, dest=st.HEATING_UP_SF, conditions=['is_pump_sf_on'])
        self.machine.add_transition('stop_recirculating_sf', source=st.HEATING_UP_SF, dest=st.IDLE, unless=['is_pump_sf_on'])
        self.machine.add_transition('start_sf_heating_ts', source=st.HEATING_UP_SF, dest=st.SF_HEATING_TS, conditions=['is_pump_ts_on', 'is_pump_sf_on'])

        self.machine.add_transition('stop_sf_heating_ts', source=st.SF_HEATING_TS, dest=st.HEATING_UP_SF, conditions=['is_pump_sf_on'], unless=['is_pump_ts_on'])
        self.machine.add_transition('shutdown', source=st.SF_HEATING_TS, dest=st.IDLE, unless=['is_pump_sf_on', 'is_pump_ts_on'])


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

    def update_inputs_array(self) -> np.ndarray[float]:
        self.inputs_array = self.get_inputs(format='array')

        return self.inputs_array

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


    def step(self, qsf: float, qts_src: float) -> None:
        """ Move the state machine one step forward """

        self.current_sample += 1

        # Inputs validation (would be done by Pydantic), here just update the values
        self.qsf = qsf
        self.qts_src = qts_src

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.update_inputs_array()

        transition = self.get_next_valid_transition(prior_inputs=self.inputs_array_prior,
                                                    current_inputs=self.inputs_array)
        if transition is not None:
            transition()

        # Save prior inputs
        self.inputs_array_prior = self.inputs_array

    def to_dataframe(self, df: pd.DataFrame = None) -> pd.DataFrame:
        # Return some of the internal variables as a dataframe
        # the state as en Enum?str?, the inputs, the consumptions

        if df is None:
            df = pd.DataFrame()

        data = pd.DataFrame({
            'state': self.state.name,
            'Tsf_out': self.Tsf_out,
            'qts_src': self.qts_src,
        }, index=[0])

        df = pd.concat([df, data], ignore_index=True)

        return df