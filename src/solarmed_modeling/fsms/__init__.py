from typing import Callable, Literal, Type
from enum import Enum
from pathlib import Path

from loguru import logger
import numpy as np
from transitions.extensions import GraphMachine as Machine


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

# SfTsState_with_value = Enum('SfTsState_with_value', {
#     f'{state.name}': i
#     for i, state in enumerate(SfTsState)
# })

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

# Variant where the values are set as normal integers with increasing value instead of a code
SolarMedState_with_value = Enum('SolarMedState_with_value', {
    f'{state.name}': i
    for i, state in enumerate(SolarMedState)
})

SupportedSystemsStatesType = MedState | SolarFieldState | ThermalStorageState | SolarMedState | SfTsState

def ensure_type(expected_type: Type) -> callable:
    def decorator(func):
        def wrapper(self, value):
            if not isinstance(value, expected_type):
                return getattr(expected_type, value)

            return func(self, value)
        return wrapper

class BaseFsm:

    """
    Base class for Finite State Machines (FSM)

    Some conventions:
    - Every input that needs to be provided in the `step` method (to move the state machine one step forward), needs to
    be gettable from the get_inputs method, either as a numpy array or a dictionary. 
    
    For consistency during comparison to check whether values have changed or not, every input needs to be convertable 
    to a **float**.
    """

    current_sample: int = 0
    warn_different_inputs_but_no_state_change: bool = False

    _state_type: Enum = None  # To be defined in the child class, type of the FSM state, should be an Enum like class

    def __init__(self, name: str, initial_state: str | Enum, sample_time: int):

        self.name = name
        self.sample_time = sample_time

        if isinstance(initial_state, str):
            initial_state = getattr(self._state_type, initial_state)

        # State machine initialization
        self.machine: Machine = Machine(
            model=self, initial=initial_state, auto_transitions=False, show_conditions=True, show_state_attributes=True,
            ignore_invalid_triggers=False, queued=True, send_event=True, # finalize_event=''
            before_state_change='inform_exit_state', after_state_change='inform_enter_state',
        )

    def get_state(self) -> Enum:
        return self.state if isinstance(self.state, self._state_type) else getattr(self._state_type, self.state)

    # @property
    # def state(self):
    #     return self._state
    #
    # @state.setter
    # @ensure_type(type(cls.))
    # def state(self, value):
    #     self._state = value

    def get_inputs(self, format: Literal['array', 'dict'] = 'array') -> None:
        """
        This base method can be used to perform validation of format, 
        but the logic to get the inputs should be implemented in the child class.
        
        Example implementation:

        def get_inputs(self, format: Literal['array', 'dict'] = 'array'):
        
        super().get_inputs(format=format) # Just to check if the format is valid

        if format == 'array':
            # When the array format is used, all variables necessarily need to be parsed as floats

            med_vac_float = float(str(self.med_vacuum_state.value)) if self.med_vacuum_state is not None else None
            return np.array([self.qmed_s, self.qmed_f, self.Tmed_s_in, self.Tmed_c_out, med_vac_float], dtype=float)

        elif format == 'dict':
            # In the dict format, each variable  can have its own type

            return {
                'qmed_s': self.qmed_s,
                'qmed_f': self.qmed_f,
                'Tmed_s_in': self.Tmed_s_in,
                'Tmed_c_out': self.Tmed_c_out,
                'med_vacuum_state': self.med_vacuum_state,
            }
        :return:
        """
        if format not in ["array", "dict"]:
            raise ValueError(f"Format should be either 'array' or 'dict', not {format}")

    def update_inputs_array(self) -> np.ndarray[float]:

        # self.inputs_array = self.get_inputs(format='array')
        # return self.inputs_array

        raise NotImplementedError("This method should be implemented in the child class")


    def customize_fsm_style(self) -> None:
        # Custom styling of state machine graph
        self.machine.machine_attributes['ratio'] = '0.2'
        # self.machine.machine_attributes['rankdir'] = 'TB'
        self.machine.style_attributes['node']['transient'] = {'fillcolor': '#FBD385'}
        self.machine.style_attributes['node']['steady'] = {'fillcolor': '#E0E8F1'}

        # # customize node styling
        # model_ = list(self.machine.model_graphs.keys())[0]  # lavin
        # for s in [MedState.GENERATING_VACUUM, MedState.STARTING_UP, MedState.SHUTTING_DOWN]:
        #     self.machine.model_graphs[model_].set_node_style(s, 'transient')
        # for s in [MedState.OFF, MedState.IDLE, MedState.ACTIVE]:
        #     self.machine.model_graphs[model_].set_node_style(s, 'steady')

    # State machine actions - callbacks of states and transitions
    def inform_wasteful_operation(self, *args) -> None:
        """ This is supposed to be implemented by the child class"""
        pass

    def inform_enter_state(self, *args) -> None:
        event = args[0]

        # Inform of not invalid but wasteful operations
        self.inform_wasteful_operation(event)

        logger.debug(f"[{self.name} - sample {self.current_sample}] Entered state {event.state.name}")

    def inform_exit_state(self, *args) -> None:
        event = args[0]
        logger.debug(f"[{self.name} - sample {self.current_sample}] Left state {event.state.name}")

    def get_next_valid_transition(self, prior_inputs: np.ndarray, current_inputs: np.ndarray) -> Callable | None:
        # Check every transition possible from the current state
        # There could be several
        candidate_transitions = self.machine.get_triggers(self.state)
        # However, when checking if the transition is valid, only one should be valid
        transition_trigger_id = None
        for candidate in candidate_transitions:
            check_id = f'may_{candidate}'
            check_transition = getattr(self, check_id)
            if check_transition():
                if transition_trigger_id is not None:
                    raise ValueError("WDYM More than one transition is valid")

                transition_trigger_id = candidate

        if transition_trigger_id is None:
            if not np.array_equal(prior_inputs, current_inputs):
                # Inputs changed, yet no valid transition, raise error
                # raise tr.MachineError(f"No valid transition for given inputs {current_inputs} from state {self.state}")

                # Does it really need to raise an error? Maybe just log it
                if self.warn_different_inputs_but_no_state_change:
                    logger.warning(f"[{self.name}] Inputs changed from prior iteration, yet no valid transition. Inputs: {current_inputs}")
            else:
                logger.debug(f"[{self.name}] No transition found and inputs are the same as in prior iteration, staying in the same state")

            return None

        return getattr(self, transition_trigger_id)

    # Auxiliary methods (to calculate associated costs depending on the state)
    def generate_graph(self, fmt :Literal['png', 'svg'] = 'svg', output_path: Path = None) -> str:
        if output_path is None:
            return self.machine.get_graph().draw(None, format=fmt, prog='dot')
        else:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'bw') as f:
                return self.machine.get_graph().draw(f, format=fmt, prog='dot')


# class SolarMED(BaseModel):
#     """
#     This class is a simplified copy for the one in solarMED_modeling.
#     It is used to test the FSMs integration without dealing with the whole model complexity before adding changes to the
#     main model class.
#     It's signature might be outdated from the current version in the main package.
    
#     It should act like a wrapper around the two individual finite state machines (fsm), and depending on the inputs 
#     given to the step method, call the correct events in the individual fsms. It should also provide utility methods 
#     like getting the current state of the system, information like the number of complete cycles, etc.
#     """
    
#     # Limits
#     # Important to define first, so that they are available for validation
#     ## Flows. Need to be defined separately to validate using `within_range_or_zero_or_max`
#     lims_mts_src: rangeType = Field((0.95, 20), title="mts,src limits", json_schema_extra={"units": "m3/h"},
#                                     description="Thermal storage heat source flow rate range (m³/h)", repr=False)
#     ## Solar field, por comprobar!!
#     lims_msf: rangeType = Field((4.7, 14), title="msf limits", json_schema_extra={"units": "m3/h"},
#                                 description="Solar field flow rate range (m³/h)", repr=False)
#     lims_qmed_s: rangeType = Field((30, 48), title="mmed,s limits", json_schema_extra={"units": "m3/h"},
#                                    description="MED hot water flow rate range (m³/h)", repr=False)
#     lims_qmed_f: rangeType = Field((5, 9), title="mmed,f limits", json_schema_extra={"units": "m3/h"},
#                                    description="MED feedwater flow rate range (m³/h)", repr=False)
#     lims_mmed_c: rangeType = Field((8, 21), title="mmed,c limits", json_schema_extra={"units": "m3/h"},
#                                    description="MED condenser flow rate range (m³/h)", repr=False)

#     # Tmed_s_in, límite dinámico
#     lims_Tmed_s_in: rangeType = Field((60, 75), title="Tmed,s,in limits", json_schema_extra={"units": "C"},
#                                       # TODO: Upper limit should be greater if new model was trained
#                                       description="MED hot water inlet temperature range (ºC)", repr=False)
#     lims_Tsf_out: rangeType = Field((65, 120), title="Tsf,out setpoint limits", json_schema_extra={"units": "C"},
#                                     description="Solar field outlet temperature setpoint range (ºC)", repr=False)
#     ## Common
#     lims_T_hot: rangeType = Field((0, 120), title="Thot* limits", json_schema_extra={"units": "C"},
#                                   description="Solar field and thermal storage temperature range (ºC)", repr=False)

#     # Parameters
#     ## General parameters
#     sample_time: int = Field(60, description="Sample rate (seg)", title="sample rate",
#                                json_schema_extra={"units": "s"})
#     # curve_fits_path: Path = Field(Path('data/curve_fits.json'), description="Path to the file with the curve fits", repr=False)

#     ## MED
#     # Chapuza: Por favor, asegurarse de que aquí se definen en el mimso orden que se usan después al asociarle un caudal
#     # mmed_b, qmed_f, mmed_d, mmed_c, qmed_s
#     med_actuators: list[Actuator] | list[str] = Field(["med_brine_pump", "med_feed_pump",
#                                                        "med_distillate_pump", "med_cooling_pump",
#                                                        "med_heatsource_pump"],
#                                                       description="Actuators to estimate electricity consumption for the MED",
#                                                       title="MED actuators", repr=False)

#     ## Thermal storage
#     ts_actuators: list[Actuator] | list[str] = Field(["ts_src_pump"], title="Thermal storage actuators", repr=False,
#                                                      description="Actuators to estimate electricity consumption for the thermal storage")

#     UAts_h: list[PositiveFloat] = Field([0.0069818, 0.00584034, 0.03041486], title="UAts,h",
#                                         json_schema_extra={"units": "W/K"},
#                                         description="Heat losses to the environment from the hot tank (W/K)",
#                                         repr=False)
#     UAts_c: list[PositiveFloat] = Field([0.01396848, 0.0001, 0.02286885], title="UAts,c",
#                                         json_schema_extra={"units": "W/K"},
#                                         description="Heat losses to the environment from the cold tank (W/K)",
#                                         repr=False)
#     Vts_h: list[PositiveFloat] = Field([5.94771006, 4.87661781, 2.19737023], title="Vts,h",
#                                        json_schema_extra={"units": "m3"},
#                                        description="Volume of each control volume of the hot tank (m³)", repr=False)
#     Vts_c: list[PositiveFloat] = Field([5.33410037, 7.56470594, 0.90547187], title="Vts,c",
#                                        json_schema_extra={"units": "m3"},
#                                        description="Volume of each control volume of the cold tank (m³)", repr=False)

#     ## Solar field
#     sf_actuators: list[Actuator] | list[str] = Field(["sf_pump"], title="Solar field actuators", repr=False,
#                                                      description="Actuators to estimate electricity consumption for the solar field")

#     beta_sf: float = Field(4.36396e-02, title="βsf", json_schema_extra={"units": "m"}, repr=False,
#                            description="Solar field. Gain coefficient", gt=0, le=1)
#     H_sf: float = Field(13.676448551722462, title="Hsf", json_schema_extra={"units": "W/m2"}, repr=False,
#                         description="Solar field. Losses to the environment", ge=0, le=20)
#     gamma_sf: float = Field(0.1, title="γsf", json_schema_extra={"units": "-"}, repr=False,
#                             description="Solar field. Artificial parameters to account for flow variations within the "
#                                         "whole solar field", ge=0, le=1)
#     filter_sf: float = Field(0.1, title="filter_sf", json_schema_extra={"units": "-"}, repr=False,
#                              description="Solar field. Weighted average filter coefficient to smooth the flow rate",
#                              ge=0, le=1)

#     nt_sf: int = Field(1, title="nt,sf", repr=False,
#                        description="Solar field. Number of tubes in parallel per collector. Defaults to 1.", ge=0)
#     np_sf: int = Field(7 * 5, title="np,sf", repr=False,
#                        description="Solar field. Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments.",
#                        ge=0)
#     ns_sf: int = Field(2, title="ns,sf", repr=False,
#                        description="Solar field. Number of loops in series", ge=0)
#     Lt_sf: float = Field(1.15 * 20, title="Ltsf", repr=False,
#                          json_schema_extra={"units": "m"}, description="Solar field. Collector tube length", gt=0)
#     Acs_sf: float = Field(7.85e-5, title="Acs,sf", repr=False, json_schema_extra={"units": "m2"},
#                           description="Solar field. Flat plate collector tube cross-section area", gt=0)
#     Kp_sf: float = Field(-0.1, title="Kp,sf", repr=False,
#                          description="Solar field. Proportional gain for the local PID controller", le=0)
#     Ki_sf: float = Field(-0.01, title="Ki,sf", repr=False,
#                          description="Solar field. Integral gain for the local PID controller", le=0)

#     ## Heat exchanger
#     UA_hx: float = Field(13536.596, title="UA,hx", json_schema_extra={"units": "W/K"}, repr=False,
#                          description="Heat exchanger. Heat transfer coefficient", gt=0)
#     H_hx: float = Field(0, title="Hhx", json_schema_extra={"units": "W/m2"}, repr=False,
#                         description="Heat exchanger. Losses to the environment")

#     # Variables (states, outputs, decision variables, inputs, etc.)
#     # Environment
#     wmed_f: float = Field(35, title="wmed,f", json_schema_extra={"units": "g/kg"},
#                           description="Environment. Seawater / MED feedwater salinity (g/kg)", gt=0)
#     Tamb: float = Field(None, title="Tamb", json_schema_extra={"units": "C"},
#                         description="Environment. Ambient temperature (ºC)", ge=-15, le=50)
#     I: float = Field(None, title="I", json_schema_extra={"units": "W/m2"},
#                      description="Environment. Solar irradiance (W/m2)", ge=0, le=2000)
#     Tmed_c_in: float = Field(None, title="Tmed,c,in", json_schema_extra={"units": "C"},
#                              description="Environment. Seawater temperature (ºC)", ge=10, le=28)

#     # Thermal storage
#     mts_src_sp: float = Field(None, title="mts,src*", json_schema_extra={"units": "m3/h"},
#                               description="Decision variable. Thermal storage recharge flow rate (m³/h)")

#     mts_src: float = Field(None, title="mts,src", json_schema_extra={"units": "m3/h"},
#                            description="Output. Thermal storage recharge flow rate (m³/h)")
#     mts_dis: float = Field(None, title="mts,dis", json_schema_extra={"units": "m3/h"},
#                            description="Output. Thermal storage discharge flow rate (m³/h)")
#     Tts_h_in: conHotTemperatureType = Field(None, title="Tts,h,in", json_schema_extra={"units": "C"},
#                                             description="Output. Thermal storage heat source inlet temperature, to top of hot tank == Thx_s_out (ºC)")
#     Tts_c_in: conHotTemperatureType = Field(None, title="Tts,c,in", json_schema_extra={"units": "C"},
#                                             description="Output. Thermal storage load discharge inlet temperature, to bottom of cold tank == Tmed_s_out (ºC)")
#     Tts_h_out: conHotTemperatureType = Field(None, title="Tts,h,out", json_schema_extra={"units": "C"},
#                                              description="Output. Thermal storage heat source outlet temperature, from top of hot tank == Tts_h_t (ºC)")
#     Tts_h: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,h",
#                                                                                    json_schema_extra={"units": "C"},
#                                                                                    description="Output. Temperature profile in the hot tank (ºC)")
#     Tts_c: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,c",
#                                                                                    json_schema_extra={"units": "C"},
#                                                                                    description="Output. Temperature profile in the cold tank (ºC)")
#     Pts_src: float = Field(None, title="Pth,ts,in", json_schema_extra={"units": "kWth"},
#                            description="Output. Thermal storage inlet power (kWth)")
#     Pts_dis: float = Field(None, title="Pth,ts,dis", json_schema_extra={"units": "kWth"},
#                            description="Output. Thermal storage outlet power (kWth)")
#     Jts: float = Field(None, title="Jts,e", json_schema_extra={"units": "kWe"},
#                        description="Output. Thermal storage electrical power consumption (kWe)")

#     # Solar field
#     Tsf_out_sp: conHotTemperatureType = Field(None, title="Tsf,out*", json_schema_extra={"units": "C"},
#                                               description="Decision variable. Solar field outlet temperature (ºC)")

#     Tsf_out: conHotTemperatureType = Field(None, title="Tsf,out", json_schema_extra={"units": "C"},
#                                            description="Output. Solar field outlet temperature (ºC)")
#     Tsf_in: conHotTemperatureType = Field(None, title="Tsf,in", json_schema_extra={"units": "C"},
#                                           description="Output. Solar field inlet temperature (ºC)")
#     Tsf_in_ant: np.ndarray[conHotTemperatureType] = Field(..., title="Tsf,in_ant", json_schema_extra={"units": "C"},
#                                                           description="Solar field inlet temperature in the previous Nsf_max steps (ºC)")
#     msf_ant: np.ndarray[float] = Field(..., repr=False, exclude=False,
#                                        description='Solar field flow rate in the previous Nsf_max steps', )
#     Tsf_out_ant: conHotTemperatureType = Field(None, title="Tsf,out,ant", json_schema_extra={"units": "C"},
#                                                description="Output. Solar field prior outlet temperature (ºC)")
#     msf: float = Field(None, title="msf", json_schema_extra={"units": "m3/h"},
#                        description="Output. Solar field flow rate (m³/h)", alias="qsf")
#     SEC_sf: float = Field(None, title="SEC_sf", json_schema_extra={"units": "kWhe/kWth"},
#                           description="Output. Solar field conversion efficiency (kWhe/kWth)")
#     Jsf: float = Field(None, title="Jsf,e", json_schema_extra={"units": "kW"},
#                        description="Output. Solar field electrical power consumption (kWe)")
#     Psf: float = Field(None, title="Pth_sf", json_schema_extra={"units": "kWth"},
#                        description="Output. Solar field thermal power generated (kWth)")

#     # MED
#     qmed_s_sp: float = Field(None, title="mmed,s*", json_schema_extra={"units": "m3/h"},
#                              description="Decision variable. MED hot water flow rate (m³/h)")
#     qmed_f_sp: float = Field(None, title="mmed,f*", json_schema_extra={"units": "m3/h"},
#                              description="Decision variable. MED feedwater flow rate (m³/h)")
#     # Here absolute limits are defined, but upper limit depends on Tts_h_t
#     Tmed_s_in_sp: float = Field(None, title="Tmed,s,in*", json_schema_extra={"units": "C"},
#                                 description="Decision variable. MED hot water inlet temperature (ºC)")
#     Tmed_c_out_sp: float = Field(None, title="Tmed,c,out*", json_schema_extra={"units": "C"},
#                                  description="Decision variable. MED condenser outlet temperature (ºC)")

#     qmed_s: float = Field(None, title="mmed,s", json_schema_extra={"units": "m3/h"},
#                           description="Output. MED hot water flow rate (m³/h)")
#     qmed_f: float = Field(None, title="mmed,f", json_schema_extra={"units": "m3/h"},
#                           description="Output. MED feedwater flow rate (m³/h)")
#     Tmed_s_in: float = Field(None, title="Tmed,s,in", json_schema_extra={"units": "C"},
#                              description="Output. MED hot water inlet temperature (ºC)")
#     Tmed_c_out: float = Field(None, title="Tmed,c,out", json_schema_extra={"units": "C"},
#                               description="Output. MED condenser outlet temperature (ºC)", ge=0)
#     mmed_c: float = Field(None, title="mmed,c", json_schema_extra={"units": "m3/h"},
#                           description="Output. MED condenser flow rate (m³/h)")
#     Tmed_s_out: float = Field(None, title="Tmed,s,out", json_schema_extra={"units": "C"},
#                               description="Output. MED heat source outlet temperature (ºC)")
#     mmed_d: float = Field(None, title="mmed,d", json_schema_extra={"units": "m3/h"},
#                           description="Output. MED distillate flow rate (m³/h)")
#     mmed_b: float = Field(None, title="mmed,b", json_schema_extra={"units": "m3/h"},
#                           description="Output. MED brine flow rate (m³/h)")
#     Jmed: float = Field(None, title="Jmed", json_schema_extra={"units": "kWe"},
#                         description="Output. MED electrical power consumption (kW)")
#     Pmed: float = Field(None, title="Pmed", json_schema_extra={"units": "kWth"},
#                         description="Output. MED thermal power consumption ~= Pth_ts_out (kW)")
#     STEC_med: float = Field(None, title="STEC_med", json_schema_extra={"units": "kWhe/m3"},
#                             description="Output. MED specific thermal energy consumption (kWhe/m³)")
#     SEEC_med: float = Field(None, title="SEEC_med", json_schema_extra={"units": "kWhth/m3"},
#                             description="Output. MED specific electrical energy consumption (kWhth/m³)")

#     # Heat exchanger
#     # Basically copies of existing variables, but with different names, no bounds checking
#     Thx_p_in: conHotTemperatureType = Field(None, title="Thx,p,in", json_schema_extra={"units": "C"},
#                                             description="Output. Heat exchanger primary circuit (hot side) inlet temperature == Tsf_out (ºC)")
#     Thx_p_out: conHotTemperatureType = Field(None, title="Thx,p,out", json_schema_extra={"units": "C"},
#                                              description="Output. Heat exchanger primary circuit (hot side) outlet temperature == Tsf_in (ºC)")
#     Thx_s_in: conHotTemperatureType = Field(None, title="Thx,s,in", json_schema_extra={"units": "C"},
#                                             description="Output. Heat exchanger secondary circuit (cold side) inlet temperature == Tts_c_out(ºC)")
#     Thx_s_out: conHotTemperatureType = Field(None, title="Thx,s,out", json_schema_extra={"units": "C"},
#                                              description="Output. Heat exchanger secondary circuit (cold side) outlet temperature == Tts_t_in (ºC)")
#     mhx_p: float = Field(None, title="mhx,p", json_schema_extra={"units": "m3/h"},
#                          description="Output. Heat exchanger primary circuit (hot side) flow rate == msf (m³/h)")
#     mhx_s: float = Field(None, title="mhx,s", json_schema_extra={"units": "m3/h"},
#                          description="Output. Heat exchanger secondary circuit (cold side) flow rate == mts_src (m³/h)")
#     Phx_p: float = Field(None, title="Pth,hx,p", json_schema_extra={"units": "kWth"},
#                          description="Output. Heat exchanger primary circuit (hot side) power == Pth_sf (kWth)")
#     Phx_s: float = Field(None, title="Pth,hx,s", json_schema_extra={"units": "kWth"},
#                          description="Output. Heat exchanger secondary circuit (cold side) power == Pth_ts_in (kWth)")
#     epsilon_hx: float = Field(None, title="εhx", json_schema_extra={"units": "-"},
#                               description="Output. Heat exchanger effectiveness (-)")

#     # Three-way valve
#     # Same case as with heat exchanger
#     R3wv: float = Field(None, title="R3wv", json_schema_extra={"units": "-"},
#                         description="Output. Three-way valve mix ratio (-)")
#     m3wv_src: float = Field(None, title="m3wv,src", json_schema_extra={"units": "m3/h"},
#                             description="Output. Three-way valve source flow rate == mts,dis (m³/h)")
#     m3wv_dis: float = Field(None, title="m3wv,dis", json_schema_extra={"units": "m3/h"},
#                             description="Output. Three-way valve discharge flow rate == mmed,s (m³/h)")
#     T3wv_src: conHotTemperatureType = Field(None, title="T3wv,src", json_schema_extra={"units": "C"},
#                                             description="Output. Three-way valve source temperature == Tts,h,t (ºC)")
#     T3wv_dis_in: conHotTemperatureType = Field(None, title="T3wv,dis,in", json_schema_extra={"units": "C"},
#                                                description="Output. Three-way valve discharge inlet temperature == Tmed,s,in (ºC)")
#     T3wv_dis_out: conHotTemperatureType = Field(None, title="T3wv,dis,out", json_schema_extra={"units": "C"},
#                                                 description="Output. Three-way valve discharge outlet temperature == Tmed,s,out (ºC)")


#     # New variables for FSM
#     # states: list[Enum] = [state for state in SolarMED_State]
#     vacuum_duration_time: int = Field(5*60, title="MED vacuum,duration", json_schema_extra={"units": "s"},
#                                       description="Time to generate vacuum (seconds)")
#     brine_emptying_time: int = Field(3*60, title="MED brine,emptying,duration", json_schema_extra={"units": "s"},
#                                         description="Time to extract brine from MED plant (seconds)")
#     startup_duration_time: int = Field(1*60, title="MED startup,duration", json_schema_extra={"units": "s"},
#                                        description="Time to start up the MED plant (seconds)")

#     med_vacuum_state: MedVacuumState = Field(MedVacuumState.OFF, title="MEDvacuum,state", json_schema_extra={"units": "-"},
#                                                 description="Input. MED vacuum system state")
#     med_state: MedState = Field(MedState.OFF, title="MED,state", json_schema_extra={"units": "-"},
#                                 description="Input/Output. MED state. It can be used to define the MED initial state, after it's always an output")
#     sf_state: SolarFieldState = Field(SolarFieldState.IDLE, title="SF,state", json_schema_extra={"units": "-"},
#                                      description="Input/Output. Solar field state. It can be used to define the Solar Field initial state, after it's always an output")
#     ts_state: ThermalStorageState = Field(ThermalStorageState.IDLE, title="TS,state", json_schema_extra={"units": "-"},
#                                             description="Input/Output. Thermal storage state. It can be used to define the Thermal Storage initial state, after it's always an output")
#     sf_ts_state: SfTsState = Field(SfTsState.IDLE, title="SF_TS,state", json_schema_extra={"units": "-"},
#                                         description="Output. Solar field with thermal storage state")
#     current_state: SolarMedState = Field(None, title="state", json_schema_extra={"units": "-"},
#                                             description="Output. Current state of the SolarMED system")

#     _med_fsm: MedFsm = PrivateAttr(None)
#     _sf_ts_state: SfTsState = PrivateAttr(None)
#     _created_at: datetime = PrivateAttr(default_factory=datetime.datetime.now)

#     model_config = ConfigDict(
#         validate_assignment=True,  # So that fields are validated, not only when created, but every time they are set
#         arbitrary_types_allowed=True
#         # numpy.ndarray[typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=110)])]]
#     )

#     def model_post_init(self, ctx):

#         initial_sf_ts = SfTsState(str(self.sf_state.value) + str(self.ts_state.value))
#         self.current_state = SolarMedState(initial_sf_ts.value + str(self.med_state.value))

#         self._sf_ts_fsm: SolarFieldWithThermalStorageFsm = SolarFieldWithThermalStorageFsm(
#             name='SolarFieldWithThermalStorage_FSM', initial_state=initial_sf_ts, sample_time=self.sample_time)
#         self._med_fsm: MedFsm = MedFsm(
#             name='MED_FSM', initial_state=self.med_state,
#             sample_time=self.sample_time, vacuum_duration_time=self.vacuum_duration_time,
#             brine_emptying_time=self.brine_emptying_time, startup_duration_time=self.startup_duration_time
#         )

#     def step(
#             self,
#             # Thermal storage decision variables
#             mts_src: float,
#             # Solar field decision variables
#             Tsf_out: float,
#             # MED decision variables
#             qmed_s, qmed_f, Tmed_s_in, Tmed_c_out, med_vacuum_state: MedVacuumState | int,
#             # Environment variables
#             Tmed_c_in: float, Tamb: float, I: float, wmed_f: float = None,
#             # Optional
#             msf: float = None,
#             # Optional, to provide the solar field flow rate when starting up (Tsf_out takes priority)
#     ):
#         # Validation of inputs
#         # In this mockup class just copy the inputs into class variables
#         self.mts_src = mts_src
#         self.Tsf_out = Tsf_out
#         self.qmed_s = qmed_s
#         self.qmed_f = qmed_f
#         self.Tmed_s_in = Tmed_s_in
#         self.Tmed_c_out = Tmed_c_out
#         self.med_vacuum_state = med_vacuum_state

#         # After the validation, variables are either zero or within the limits (>0),
#         # based on this, the step method in the individual state machines are called

#         self._sf_ts_fsm.step(Tsf_out=Tsf_out, qts_src=mts_src)
#         self._med_fsm.step(qmed_s=qmed_s, qmed_f=qmed_f, Tmed_s_in=Tmed_s_in, Tmed_c_out=Tmed_c_out,
#                           med_vacuum_state=med_vacuum_state)

#         self.update_current_state()
#         logger.info(f"SolarMED current state: {self.current_state}")
        

#     def get_state(self, mode: Literal["default", "human_readable"] = 'default') -> SolarMedState:
#         # state_code = self.generate_state_code(self._sf_ts_fsm.state, self._med_fsm.state)

#         state_code = str(self.sf_state.value) + str(self.ts_state.value) + str(self.med_state.value)

#         if mode == 'human_readable':
#             state_str = SolarMedState(state_code).name
#             # Replace _ by space and make everything minusculas
#             state_str =  state_str.replace('_', ' ').lower()
#             # Replace ts to TS, sf to SF and med to MED
#             state_str = state_str.replace('ts', 'TS').replace('sf', 'SF').replace('med', 'MED')

#             return state_str

#         else:
#             return SolarMedState(state_code)

#     def update_internal_states(self) -> None:
#         self.med_state = self._med_fsm.get_state()
#         self.sf_ts_state: SfTsState = self._sf_ts_fsm.get_state()
#         self.sf_state = SolarFieldState(int(self.sf_ts_state.value[0]))
#         self.ts_state = ThermalStorageState(int(self.sf_ts_state.value[1]))

#     def update_current_state(self) -> None:
#         self.update_internal_states()
#         self.current_state = self.get_state()


#     def to_dataframe(self, df: pd.DataFrame = None):
#         # Return some of the internal variables as a dataframe
#         # the state as en Enum?str?, the inputs, the consumptions

#         if df is None:
#             df = pd.DataFrame()

#         data = pd.DataFrame({
#             'state': self.current_state.name,
#             'state_title': self.get_state(mode='human_readable'),

#             'state_med': self.med_state,
#             'qmed_s': self.qmed_s,
#             'qmed_f': self.qmed_f,
#             'Tmed_s_in': self.Tmed_s_in,
#             'Tmed_c_out': self.Tmed_c_out,

#             'state_sf': self.sf_state,
#             'state_ts': self.ts_state,
#             'state_sf_ts': self.sf_ts_state, #SF_TS_State(str(self.sf_state.value) + str(self.ts_state.value)),
#             'Tsf_out': self.Tsf_out,
#             'qts_src': self.mts_src,
#         }, index=[0])

#         df = pd.concat([df, data], ignore_index=True)

#         return df