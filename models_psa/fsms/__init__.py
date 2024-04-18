from typing import Literal
from collections.abc import Callable
from pathlib import Path
import pandas as pd
import numpy as np
import datetime

import transitions as tr
from loguru import logger
from transitions.extensions import GraphMachine as Machine
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PrivateAttr

from models_psa import (
    MedVacuumState,
    MedState,
    SF_TS_State,
    SolarMED_State,
    SolarFieldState,
    ThermalStorageState,
)
from models_psa.data_validation import rangeType, conHotTemperatureType
from models_psa.power_consumption import Actuator, SupportedActuators



class Base_FSM:

    """
    Base class for Finite State Machines (FSM)
    """

    sample_rate: int = 1  # seconds
    current_sample = 0

    def __init__(self, name, initial_state):

        self.name = name

        # State machine initialization
        self.machine = Machine(
            model=self, initial=initial_state, auto_transitions=False, show_conditions=True, show_state_attributes=True,
            ignore_invalid_triggers=False, queued=True, send_event=True, # finalize_event=''
            before_state_change='inform_exit_state', after_state_change='inform_enter_state',
        )

    def customize_fsm_style(self):
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
    def inform_wasteful_operation(self, *args):
        """ This is supposed to be implemented by the child class"""
        pass

    def inform_enter_state(self, *args):
        event = args[0]

        # Inform of not invalid but wasteful operations
        self.inform_wasteful_operation(event)

        logger.debug(f"Entered state {event.state.name}")

    def inform_exit_state(self, *args):
        event = args[0]
        logger.debug(f"Left state {event.state.name}")

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
                logger.warning(f"No valid transition for given inputs {current_inputs} from state {self.state}")
            else:
                logger.info("Inputs are the same from prior iteration, no transition needed")

            return None

        return getattr(self, transition_trigger_id)

    # Auxiliary methods (to calculate associated costs depending on the state)
    def generate_graph(self, fmt :Literal['png', 'svg'] = 'svg', output_path :Path = None):
        if output_path is not None:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'bw') as f:
                return self.machine.get_graph().draw(f, format=fmt, prog='dot')
        else:
            return self.machine.get_graph().draw(None, format=fmt, prog='dot')

class SolarFieldWithThermalStorage_FSM(Base_FSM):

    """
    Finite State Machine for the Solar Field with Thermal Storage (SF-TS) unit.
    """

    # sample_rate: int = 1  # seconds
    # current_sample = 0

    # Inputs / Decision variables
    Tsf_out: float = 0
    qts_src: float = 0


    def __init__(self, name :str = "SF-TS_FSM", initial_state: SF_TS_State = SF_TS_State.IDLE):

        # Call parent constructor
        super().__init__(name, initial_state)

        st = SF_TS_State # alias

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.inputs_array = np.array([self.Tsf_out, self.qts_src])

        # Initialize prior values
        self.Tsf_out_prior = self.Tsf_out
        self.qts_src_prior = self.qts_src
        self.inputs_array_prior = self.inputs_array

        # States
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


    # State machine actions - callbacks of states and transitions
    def stop_pump_ts(self, *args):
        self.qts_src = 0

    def stop_pump_sf(self, *args):
        self.Tsf_out = 0

    def stop_pumps(self, *args):
        logger.info("Stopping pumps")
        self.stop_pump_ts()
        self.stop_pump_sf()

    # State machine transition conditions
    # Solar field
    def is_pump_sf_on(self, *args):
        return self.Tsf_out > 0

    # Thermal storage
    def is_pump_ts_on(self, *args):
        return self.qts_src > 0


    def step(self, Tsf_out: float, qts_src: float):

        self.current_sample += 1

        # Inputs validation (would be done by Pydantic), here just update the values
        self.Tsf_out = Tsf_out
        self.qts_src = qts_src

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.inputs_array = np.array([self.Tsf_out, self.qts_src])

        transition = self.get_next_valid_transition(prior_inputs=self.inputs_array_prior,
                                                    current_inputs=self.inputs_array)
        if transition is not None:
            transition()

        # Save prior inputs
        self.Tsf_out_prior = Tsf_out
        self.qts_src_prior = qts_src
        self.inputs_array_prior = np.array([self.Tsf_out_prior, self.qts_src_prior])

    def to_dataframe(self, df: pd.DataFrame = None):
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

class MedFSM(Base_FSM):

    """
    Finite State Machine for the Multi-Effect Distillation (MED) unit.
    """

    # sample_rate: int = 1  # seconds
    # current_sample = 0

    # Vacuum
    generating_vacuum: bool = False
    vacuum_generated: bool = False
    vacuum_duration_time: int = 5  # seconds
    vacuum_started_sample = None

    # Shutdown
    brine_empty = True
    brine_emptying_time: float = 1  # seconds
    brine_emptying_started_sample = None

    # Startup
    startup_done: bool = False
    startup_duration_time: int = 2  # seconds
    startup_started_sample = None

    # Inputs / Decision variables
    mmed_s: float = 0
    mmed_f: float = 0
    Tmed_s_in: float = 0
    Tmed_c_out: float = 0
    med_vacuum_state: MedVacuumState = MedVacuumState.OFF

    def __init__(self, name: str = "MED_FSM", initial_state: MedState = MedState.OFF):

        # Call parent constructor
        super().__init__(name, initial_state)

        self.vacuum_duration_samples = int(self.vacuum_duration_time / self.sample_rate)
        self.brine_emptying_samples = int(self.brine_emptying_time / self.sample_rate)
        self.startup_duration_samples = int(self.startup_duration_time / self.sample_rate)

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.inputs_array = np.array([self.mmed_s, self.mmed_f, self.Tmed_s_in, self.Tmed_c_out])
        self.inputs_array_prior = self.inputs_array

        if initial_state in [MedState.ACTIVE, MedState.STARTING_UP]:
            self.brine_empty = False

        # States
        # OFF = 0
        # IDLE = 1
        # ACTIVE = 2
        st = MedState

        self.machine.add_state(st.OFF, on_enter=['reset_fsm'])
        self.machine.add_state(st.GENERATING_VACUUM)
        self.machine.add_state(st.IDLE)
        self.machine.add_state(st.STARTING_UP, on_enter=['set_brine_non_empty'])
        self.machine.add_state(st.ACTIVE, on_exit=['reset_startup'])
        self.machine.add_state(st.SHUTTING_DOWN, on_exit=['set_brine_empty'])

        # Transitions
        # {'trigger': 'generate_vacuum', 'source': 'OFF', 'dest': 'IDLE'},
        # {'trigger': 'start_up', 'source': 'IDLE', 'dest': 'ACTIVE'},
        # {'trigger': 'suspend', 'source': 'ACTIVE', 'dest': 'IDLE'},
        # {'trigger': 'complete_shut_off', 'source': 'ACTIVE', 'dest': 'OFF'},
        # {'trigger': 'partial_shut_off', 'source': 'IDLE', 'dest': 'OFF'},

        # Vacuum
        self.machine.add_transition('start_generating_vacuum', source=st.OFF, dest=st.GENERATING_VACUUM,
                                    conditions=['is_high_vacuum'], after='set_vacuum_start')
        self.machine.add_transition('finish_generating_vacuum', source=st.GENERATING_VACUUM, dest=st.IDLE,
                                    conditions=['is_vacuum_done'], unless=['is_off_vacuum'], after='set_vacuum_done')
        self.machine.add_transition('cancel_generating_vacuum', source=st.GENERATING_VACUUM, dest=st.OFF,
                                    conditions=['is_off_vacuum'])

        # Start-up
        self.machine.add_transition('start_startup', source=st.IDLE, dest=st.STARTING_UP,
                                    conditions=['are_inputs_active'], after='set_startup_start')
        self.machine.add_transition('finish_startup', source=st.STARTING_UP, dest=st.ACTIVE,
                                    conditions=['are_inputs_active', 'is_startup_done'], after='set_startup_done')

        # Shutdown
        self.machine.add_transition('start_shutdown', source=st.ACTIVE, dest=st.SHUTTING_DOWN,
                                    unless=['are_inputs_active'], after='set_brine_emptying_start')
        self.machine.add_transition('finish_shutdown', source=[st.SHUTTING_DOWN, st.IDLE], dest=st.OFF,
                                    conditions=['is_off_vacuum',
                                                'is_brine_empty'])  # Since destination is OFF already resets FSM
        self.machine.add_transition('finish_suspend', source=st.SHUTTING_DOWN, dest=st.IDLE,
                                    conditions=['is_low_vacuum', 'is_brine_empty'], after='set_brine_empty')

        self.customize_fsm_style()

    def customize_fsm_style(self):
        # # Custom styling of state machine graph
        # self.machine.machine_attributes['rankdir'] = 'TB'
        # self.machine.style_attributes['node']['transient'] = {'fillcolor': '#FBD385'}
        # self.machine.style_attributes['node']['steady'] = {'fillcolor': '#E0E8F1'}
        super(self.__class__, self).customize_fsm_style()
        self.machine.machine_attributes['ratio'] = '0.05'

        # customize node styling
        model_ = list(self.machine.model_graphs.keys())[0]  # lavin
        for s in [MedState.GENERATING_VACUUM, MedState.STARTING_UP, MedState.SHUTTING_DOWN]:
            self.machine.model_graphs[model_].set_node_style(s, 'transient')
        for s in [MedState.OFF, MedState.IDLE, MedState.ACTIVE]:
            self.machine.model_graphs[model_].set_node_style(s, 'steady')

    # State machine actions - callbacks of states and transitions

    def inform_wasteful_operation(self, *args):

        """ This is supposed to be called from parent class"""

        event = args[0]

        # Inform of not invalid but wasteful operations
        if self.vacuum_generated == True and self.is_high_vacuum():
            logger.warning("Vacuum already generated, keeping vacuum at high value is wasteful")
        if event.state == MedState.OFF and self.is_low_vacuum():
            logger.warning("MED vacuum state is OFF, vacuum should be off or high to start generating vacuum")
        if event.state in [MedState.SHUTTING_DOWN, MedState.IDLE, MedState.OFF] and self.are_inputs_valid():
            logger.warning("MED is not operating, there is no point in having its inputs active")

    def reset_fsm(self, *args):
        logger.info(f"Resetting {self.name} FSM")
        self.set_vacuum_reset()
        # self.set_brine_empty()
        # self.reset_startup()
        # Anything else?

    # Vacuum
    def set_vacuum_start(self, *args):
        if self.generating_vacuum:
            logger.warning("Already generating vacuum, no need to start again")
            return

        # Else
        self.vacuum_started_sample = self.current_sample
        logger.info(f"Started generating vacuum, it will take {self.vacuum_duration_samples} samples to complete")

    def set_vacuum_reset(self, *args):
        self.vacuum_started_sample = None
        logger.info("Cancelled vacuum generation")

    def set_vacuum_done(self, *args):
        self.vacuum_generated = True
        logger.info("Vacuum generated")

    # Shutdown
    def set_brine_emptying_start(self, *args):
        if self.brine_empty:
            logger.warning("Brine is already empty, no need to start emptying again")
            return

        self.brine_emptying_started_sample = self.current_sample
        logger.info(f"Started emptying brine, it will take {self.brine_emptying_samples} samples to complete")

    def set_brine_empty(self, *args):
        self.brine_empty = True
        self.brine_emptying_started_sample = None
        logger.info("Brine emptied")

    def set_brine_non_empty(self, *args):
        self.brine_empty = False
        logger.info("Brine non-empty")

    # Startup
    def set_startup_start(self, *args):
        if self.startup_done:
            logger.warning("Startup already done, no need to start again")
            return

        self.startup_started_sample = self.current_sample
        logger.info(f"Started starting up, it will take {self.startup_duration_samples} samples to complete")

    def set_startup_done(self, *args):
        self.startup_done = True
        logger.info("Startup done")

    def reset_startup(self, *args):
        self.startup_done = False
        self.startup_started_sample = None
        logger.info("Startup reset")

    # State machine transition conditions
    # Vacuum
    def is_high_vacuum(self, *args):  # , raise_error: bool = False):
        return self.med_vacuum_state == MedVacuumState.HIGH

    def is_low_vacuum(self, *args):
        return self.med_vacuum_state == MedVacuumState.LOW

    def is_off_vacuum(self, *args):
        return self.med_vacuum_state == MedVacuumState.OFF

    def is_vacuum_done(self, *args):
        if self.vacuum_generated:
            return True

        if self.current_sample - self.vacuum_started_sample >= self.vacuum_duration_samples:
            return True
        else:
            logger.info(
                f"Still generating vacuum, {self.current_sample - self.vacuum_started_sample}/{self.vacuum_duration_samples} samples completed")
            return False

    # Startup
    def is_startup_done(self, *args):
        if self.startup_done:
            return True

        if self.current_sample - self.startup_started_sample >= self.startup_duration_samples:
            return True
        else:
            logger.info(
                f"Still starting up, {self.current_sample - self.startup_started_sample}/{self.startup_duration_samples} samples completed")
            return False

    # Shutdown
    def is_brine_empty(self, *args):
        if self.brine_empty:
            return True

        if self.current_sample - self.brine_emptying_started_sample >= self.brine_emptying_samples:
            return True
        else:
            logger.info(
                f"Still emptying brine, {self.current_sample - self.brine_emptying_started_sample}/{self.brine_emptying_samples} samples completed")
            return False

    def are_inputs_valid(self, *args):
        # Just check if the inputs are greater than zero, not the vacuum
        return np.all(self.inputs_array > 0)

    def are_inputs_active(self, *args):  # inputs: list[float] | np.ndarray = np.array([0])
        # Checks if all variables required for MED to active are valid (greater than zero and vacuum active)
        # inputs = np.array( event.kwargs.get('inputs', 0) )

        return self.are_inputs_valid() and self.med_vacuum_state != MedVacuumState.OFF

    def step(self, mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float,
             med_vacuum_state: int | MedVacuumState):

        self.current_sample += 1

        # Inputs validation (would be done by Pydantic), here just update the values
        self.mmed_s = mmed_s
        self.mmed_f = mmed_f
        self.Tmed_s_in = Tmed_s_in
        self.Tmed_c_out = Tmed_c_out

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.inputs_array = np.array([self.mmed_s, self.mmed_f, self.Tmed_s_in, self.Tmed_c_out])

        self.med_vacuum_state = MedVacuumState(med_vacuum_state)

        transition = self.get_next_valid_transition(prior_inputs=self.inputs_array_prior,
                                                    current_inputs=self.inputs_array)

        if transition is not None:
            transition()

        # Save prior inputs
        self.mmed_s_prior = self.mmed_s
        self.mmed_f_prior = self.mmed_f
        self.Tmed_s_in_prior = self.Tmed_s_in
        self.Tmed_c_out_prior = self.Tmed_c_out
        self.inputs_array_prior = np.array([self.mmed_s_prior, self.mmed_f_prior, self.Tmed_s_in_prior, self.Tmed_c_out_prior])


    def to_dataframe(self, df: pd.DataFrame = None):
        # Return some of the internal variables as a dataframe
        # the state as en Enum?str?, the inputs, the consumptions

        if df is None:
            df = pd.DataFrame()

        data = pd.DataFrame({
            'state': self.state.name,
            'mmed_s': self.mmed_s,
            'mmed_f': self.mmed_f,
            'Tmed_s_in': self.Tmed_s_in,
            'Tmed_c_out': self.Tmed_c_out,
        }, index=[0])

        df = pd.concat([df, data], ignore_index=True)

        return df


class SolarMED(BaseModel):
    """

    WIP: Integrate with Pydanctic
    This class is a template for the one in models_psa.
    It should act like a wrapper around the two individual finite state machines (fsm), and depending on the inputs given to the step method, call the correct events in the individual fsms. It should also provide utility methods like getting the current state of the system, information like the number of complete cycles, etc.

    """
    # Limits
    # Important to define first, so that they are available for validation
    ## Flows. Need to be defined separately to validate using `within_range_or_zero_or_max`
    lims_mts_src: rangeType = Field((0.95, 20), title="mts,src limits", json_schema_extra={"units": "m3/h"},
                                    description="Thermal storage heat source flow rate range (m³/h)", repr=False)
    ## Solar field, por comprobar!!
    lims_msf: rangeType = Field((4.7, 14), title="msf limits", json_schema_extra={"units": "m3/h"},
                                description="Solar field flow rate range (m³/h)", repr=False)
    lims_mmed_s: rangeType = Field((30, 48), title="mmed,s limits", json_schema_extra={"units": "m3/h"},
                                   description="MED hot water flow rate range (m³/h)", repr=False)
    lims_mmed_f: rangeType = Field((5, 9), title="mmed,f limits", json_schema_extra={"units": "m3/h"},
                                   description="MED feedwater flow rate range (m³/h)", repr=False)
    lims_mmed_c: rangeType = Field((8, 21), title="mmed,c limits", json_schema_extra={"units": "m3/h"},
                                   description="MED condenser flow rate range (m³/h)", repr=False)

    # Tmed_s_in, límite dinámico
    lims_Tmed_s_in: rangeType = Field((60, 75), title="Tmed,s,in limits", json_schema_extra={"units": "C"},
                                      # TODO: Upper limit should be greater if new model was trained
                                      description="MED hot water inlet temperature range (ºC)", repr=False)
    lims_Tsf_out: rangeType = Field((65, 120), title="Tsf,out setpoint limits", json_schema_extra={"units": "C"},
                                    description="Solar field outlet temperature setpoint range (ºC)", repr=False)
    ## Common
    lims_T_hot: rangeType = Field((0, 120), title="Thot* limits", json_schema_extra={"units": "C"},
                                  description="Solar field and thermal storage temperature range (ºC)", repr=False)

    # Parameters
    ## General parameters
    sample_time: float = Field(60, description="Sample rate (seg)", title="sample rate",
                               json_schema_extra={"units": "s"})
    # curve_fits_path: Path = Field(Path('data/curve_fits.json'), description="Path to the file with the curve fits", repr=False)

    ## MED
    # Chapuza: Por favor, asegurarse de que aquí se definen en el mimso orden que se usan después al asociarle un caudal
    # mmed_b, mmed_f, mmed_d, mmed_c, mmed_s
    med_actuators: list[Actuator] | list[str] = Field(["med_brine_pump", "med_feed_pump",
                                                       "med_distillate_pump", "med_cooling_pump",
                                                       "med_heatsource_pump"],
                                                      description="Actuators to estimate electricity consumption for the MED",
                                                      title="MED actuators", repr=False)

    ## Thermal storage
    ts_actuators: list[Actuator] | list[str] = Field(["ts_src_pump"], title="Thermal storage actuators", repr=False,
                                                     description="Actuators to estimate electricity consumption for the thermal storage")

    UAts_h: list[PositiveFloat] = Field([0.0069818, 0.00584034, 0.03041486], title="UAts,h",
                                        json_schema_extra={"units": "W/K"},
                                        description="Heat losses to the environment from the hot tank (W/K)",
                                        repr=False)
    UAts_c: list[PositiveFloat] = Field([0.01396848, 0.0001, 0.02286885], title="UAts,c",
                                        json_schema_extra={"units": "W/K"},
                                        description="Heat losses to the environment from the cold tank (W/K)",
                                        repr=False)
    Vts_h: list[PositiveFloat] = Field([5.94771006, 4.87661781, 2.19737023], title="Vts,h",
                                       json_schema_extra={"units": "m3"},
                                       description="Volume of each control volume of the hot tank (m³)", repr=False)
    Vts_c: list[PositiveFloat] = Field([5.33410037, 7.56470594, 0.90547187], title="Vts,c",
                                       json_schema_extra={"units": "m3"},
                                       description="Volume of each control volume of the cold tank (m³)", repr=False)

    ## Solar field
    sf_actuators: list[Actuator] | list[str] = Field(["sf_pump"], title="Solar field actuators", repr=False,
                                                     description="Actuators to estimate electricity consumption for the solar field")

    beta_sf: float = Field(4.36396e-02, title="βsf", json_schema_extra={"units": "m"}, repr=False,
                           description="Solar field. Gain coefficient", gt=0, le=1)
    H_sf: float = Field(13.676448551722462, title="Hsf", json_schema_extra={"units": "W/m2"}, repr=False,
                        description="Solar field. Losses to the environment", ge=0, le=20)
    gamma_sf: float = Field(0.1, title="γsf", json_schema_extra={"units": "-"}, repr=False,
                            description="Solar field. Artificial parameters to account for flow variations within the "
                                        "whole solar field", ge=0, le=1)
    filter_sf: float = Field(0.1, title="filter_sf", json_schema_extra={"units": "-"}, repr=False,
                             description="Solar field. Weighted average filter coefficient to smooth the flow rate",
                             ge=0, le=1)

    nt_sf: int = Field(1, title="nt,sf", repr=False,
                       description="Solar field. Number of tubes in parallel per collector. Defaults to 1.", ge=0)
    np_sf: int = Field(7 * 5, title="np,sf", repr=False,
                       description="Solar field. Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments.",
                       ge=0)
    ns_sf: int = Field(2, title="ns,sf", repr=False,
                       description="Solar field. Number of loops in series", ge=0)
    Lt_sf: float = Field(1.15 * 20, title="Ltsf", repr=False,
                         json_schema_extra={"units": "m"}, description="Solar field. Collector tube length", gt=0)
    Acs_sf: float = Field(7.85e-5, title="Acs,sf", repr=False, json_schema_extra={"units": "m2"},
                          description="Solar field. Flat plate collector tube cross-section area", gt=0)
    Kp_sf: float = Field(-0.1, title="Kp,sf", repr=False,
                         description="Solar field. Proportional gain for the local PID controller", le=0)
    Ki_sf: float = Field(-0.01, title="Ki,sf", repr=False,
                         description="Solar field. Integral gain for the local PID controller", le=0)

    ## Heat exchanger
    UA_hx: float = Field(13536.596, title="UA,hx", json_schema_extra={"units": "W/K"}, repr=False,
                         description="Heat exchanger. Heat transfer coefficient", gt=0)
    H_hx: float = Field(0, title="Hhx", json_schema_extra={"units": "W/m2"}, repr=False,
                        description="Heat exchanger. Losses to the environment")

    # Variables (states, outputs, decision variables, inputs, etc.)
    # Environment
    wmed_f: float = Field(35, title="wmed,f", json_schema_extra={"units": "g/kg"},
                          description="Environment. Seawater / MED feedwater salinity (g/kg)", gt=0)
    Tamb: float = Field(None, title="Tamb", json_schema_extra={"units": "C"},
                        description="Environment. Ambient temperature (ºC)", ge=-15, le=50)
    I: float = Field(None, title="I", json_schema_extra={"units": "W/m2"},
                     description="Environment. Solar irradiance (W/m2)", ge=0, le=2000)
    Tmed_c_in: float = Field(None, title="Tmed,c,in", json_schema_extra={"units": "C"},
                             description="Environment. Seawater temperature (ºC)", ge=10, le=28)

    # Thermal storage
    mts_src_sp: float = Field(None, title="mts,src*", json_schema_extra={"units": "m3/h"},
                              description="Decision variable. Thermal storage recharge flow rate (m³/h)")

    mts_src: float = Field(None, title="mts,src", json_schema_extra={"units": "m3/h"},
                           description="Output. Thermal storage recharge flow rate (m³/h)")
    mts_dis: float = Field(None, title="mts,dis", json_schema_extra={"units": "m3/h"},
                           description="Output. Thermal storage discharge flow rate (m³/h)")
    Tts_h_in: conHotTemperatureType = Field(None, title="Tts,h,in", json_schema_extra={"units": "C"},
                                            description="Output. Thermal storage heat source inlet temperature, to top of hot tank == Thx_s_out (ºC)")
    Tts_c_in: conHotTemperatureType = Field(None, title="Tts,c,in", json_schema_extra={"units": "C"},
                                            description="Output. Thermal storage load discharge inlet temperature, to bottom of cold tank == Tmed_s_out (ºC)")
    Tts_h_out: conHotTemperatureType = Field(None, title="Tts,h,out", json_schema_extra={"units": "C"},
                                             description="Output. Thermal storage heat source outlet temperature, from top of hot tank == Tts_h_t (ºC)")
    Tts_h: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,h",
                                                                                   json_schema_extra={"units": "C"},
                                                                                   description="Output. Temperature profile in the hot tank (ºC)")
    Tts_c: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,c",
                                                                                   json_schema_extra={"units": "C"},
                                                                                   description="Output. Temperature profile in the cold tank (ºC)")
    Pts_src: float = Field(None, title="Pth,ts,in", json_schema_extra={"units": "kWth"},
                           description="Output. Thermal storage inlet power (kWth)")
    Pts_dis: float = Field(None, title="Pth,ts,dis", json_schema_extra={"units": "kWth"},
                           description="Output. Thermal storage outlet power (kWth)")
    Jts: float = Field(None, title="Jts,e", json_schema_extra={"units": "kWe"},
                       description="Output. Thermal storage electrical power consumption (kWe)")

    # Solar field
    Tsf_out_sp: conHotTemperatureType = Field(None, title="Tsf,out*", json_schema_extra={"units": "C"},
                                              description="Decision variable. Solar field outlet temperature (ºC)")

    Tsf_out: conHotTemperatureType = Field(None, title="Tsf,out", json_schema_extra={"units": "C"},
                                           description="Output. Solar field outlet temperature (ºC)")
    Tsf_in: conHotTemperatureType = Field(None, title="Tsf,in", json_schema_extra={"units": "C"},
                                          description="Output. Solar field inlet temperature (ºC)")
    Tsf_in_ant: np.ndarray[conHotTemperatureType] = Field(..., title="Tsf,in_ant", json_schema_extra={"units": "C"},
                                                          description="Solar field inlet temperature in the previous Nsf_max steps (ºC)")
    msf_ant: np.ndarray[float] = Field(..., repr=False, exclude=False,
                                       description='Solar field flow rate in the previous Nsf_max steps', )
    Tsf_out_ant: conHotTemperatureType = Field(None, title="Tsf,out,ant", json_schema_extra={"units": "C"},
                                               description="Output. Solar field prior outlet temperature (ºC)")
    msf: float = Field(None, title="msf", json_schema_extra={"units": "m3/h"},
                       description="Output. Solar field flow rate (m³/h)", alias="qsf")
    SEC_sf: float = Field(None, title="SEC_sf", json_schema_extra={"units": "kWhe/kWth"},
                          description="Output. Solar field conversion efficiency (kWhe/kWth)")
    Jsf: float = Field(None, title="Jsf,e", json_schema_extra={"units": "kW"},
                       description="Output. Solar field electrical power consumption (kWe)")
    Psf: float = Field(None, title="Pth_sf", json_schema_extra={"units": "kWth"},
                       description="Output. Solar field thermal power generated (kWth)")

    # MED
    mmed_s_sp: float = Field(None, title="mmed,s*", json_schema_extra={"units": "m3/h"},
                             description="Decision variable. MED hot water flow rate (m³/h)")
    mmed_f_sp: float = Field(None, title="mmed,f*", json_schema_extra={"units": "m3/h"},
                             description="Decision variable. MED feedwater flow rate (m³/h)")
    # Here absolute limits are defined, but upper limit depends on Tts_h_t
    Tmed_s_in_sp: float = Field(None, title="Tmed,s,in*", json_schema_extra={"units": "C"},
                                description="Decision variable. MED hot water inlet temperature (ºC)")
    Tmed_c_out_sp: float = Field(None, title="Tmed,c,out*", json_schema_extra={"units": "C"},
                                 description="Decision variable. MED condenser outlet temperature (ºC)")

    mmed_s: float = Field(None, title="mmed,s", json_schema_extra={"units": "m3/h"},
                          description="Output. MED hot water flow rate (m³/h)")
    mmed_f: float = Field(None, title="mmed,f", json_schema_extra={"units": "m3/h"},
                          description="Output. MED feedwater flow rate (m³/h)")
    Tmed_s_in: float = Field(None, title="Tmed,s,in", json_schema_extra={"units": "C"},
                             description="Output. MED hot water inlet temperature (ºC)")
    Tmed_c_out: float = Field(None, title="Tmed,c,out", json_schema_extra={"units": "C"},
                              description="Output. MED condenser outlet temperature (ºC)", ge=0)
    mmed_c: float = Field(None, title="mmed,c", json_schema_extra={"units": "m3/h"},
                          description="Output. MED condenser flow rate (m³/h)")
    Tmed_s_out: float = Field(None, title="Tmed,s,out", json_schema_extra={"units": "C"},
                              description="Output. MED heat source outlet temperature (ºC)")
    mmed_d: float = Field(None, title="mmed,d", json_schema_extra={"units": "m3/h"},
                          description="Output. MED distillate flow rate (m³/h)")
    mmed_b: float = Field(None, title="mmed,b", json_schema_extra={"units": "m3/h"},
                          description="Output. MED brine flow rate (m³/h)")
    Jmed: float = Field(None, title="Jmed", json_schema_extra={"units": "kWe"},
                        description="Output. MED electrical power consumption (kW)")
    Pmed: float = Field(None, title="Pmed", json_schema_extra={"units": "kWth"},
                        description="Output. MED thermal power consumption ~= Pth_ts_out (kW)")
    STEC_med: float = Field(None, title="STEC_med", json_schema_extra={"units": "kWhe/m3"},
                            description="Output. MED specific thermal energy consumption (kWhe/m³)")
    SEEC_med: float = Field(None, title="SEEC_med", json_schema_extra={"units": "kWhth/m3"},
                            description="Output. MED specific electrical energy consumption (kWhth/m³)")

    # Heat exchanger
    # Basically copies of existing variables, but with different names, no bounds checking
    Thx_p_in: conHotTemperatureType = Field(None, title="Thx,p,in", json_schema_extra={"units": "C"},
                                            description="Output. Heat exchanger primary circuit (hot side) inlet temperature == Tsf_out (ºC)")
    Thx_p_out: conHotTemperatureType = Field(None, title="Thx,p,out", json_schema_extra={"units": "C"},
                                             description="Output. Heat exchanger primary circuit (hot side) outlet temperature == Tsf_in (ºC)")
    Thx_s_in: conHotTemperatureType = Field(None, title="Thx,s,in", json_schema_extra={"units": "C"},
                                            description="Output. Heat exchanger secondary circuit (cold side) inlet temperature == Tts_c_out(ºC)")
    Thx_s_out: conHotTemperatureType = Field(None, title="Thx,s,out", json_schema_extra={"units": "C"},
                                             description="Output. Heat exchanger secondary circuit (cold side) outlet temperature == Tts_t_in (ºC)")
    mhx_p: float = Field(None, title="mhx,p", json_schema_extra={"units": "m3/h"},
                         description="Output. Heat exchanger primary circuit (hot side) flow rate == msf (m³/h)")
    mhx_s: float = Field(None, title="mhx,s", json_schema_extra={"units": "m3/h"},
                         description="Output. Heat exchanger secondary circuit (cold side) flow rate == mts_src (m³/h)")
    Phx_p: float = Field(None, title="Pth,hx,p", json_schema_extra={"units": "kWth"},
                         description="Output. Heat exchanger primary circuit (hot side) power == Pth_sf (kWth)")
    Phx_s: float = Field(None, title="Pth,hx,s", json_schema_extra={"units": "kWth"},
                         description="Output. Heat exchanger secondary circuit (cold side) power == Pth_ts_in (kWth)")
    epsilon_hx: float = Field(None, title="εhx", json_schema_extra={"units": "-"},
                              description="Output. Heat exchanger effectiveness (-)")

    # Three-way valve
    # Same case as with heat exchanger
    R3wv: float = Field(None, title="R3wv", json_schema_extra={"units": "-"},
                        description="Output. Three-way valve mix ratio (-)")
    m3wv_src: float = Field(None, title="m3wv,src", json_schema_extra={"units": "m3/h"},
                            description="Output. Three-way valve source flow rate == mts,dis (m³/h)")
    m3wv_dis: float = Field(None, title="m3wv,dis", json_schema_extra={"units": "m3/h"},
                            description="Output. Three-way valve discharge flow rate == mmed,s (m³/h)")
    T3wv_src: conHotTemperatureType = Field(None, title="T3wv,src", json_schema_extra={"units": "C"},
                                            description="Output. Three-way valve source temperature == Tts,h,t (ºC)")
    T3wv_dis_in: conHotTemperatureType = Field(None, title="T3wv,dis,in", json_schema_extra={"units": "C"},
                                               description="Output. Three-way valve discharge inlet temperature == Tmed,s,in (ºC)")
    T3wv_dis_out: conHotTemperatureType = Field(None, title="T3wv,dis,out", json_schema_extra={"units": "C"},
                                                description="Output. Three-way valve discharge outlet temperature == Tmed,s,out (ºC)")


    # New variables for FSM
    # states: list[Enum] = [state for state in SolarMED_State]
    med_vacuum_state: MedVacuumState = Field(MedVacuumState.OFF, title="MEDvacuum,state", json_schema_extra={"units": "-"},
                                                description="Input. MED vacuum system state")
    med_state: MedState = Field(MedState.OFF, title="MED,state", json_schema_extra={"units": "-"},
                                description="Input/Output. MED state. It can be used to define the MED initial state, after it's always an output")
    sf_state: SolarFieldState = Field(SolarFieldState.IDLE, title="SF,state", json_schema_extra={"units": "-"},
                                     description="Input/Output. Solar field state. It can be used to define the Solar Field initial state, after it's always an output")
    ts_state: ThermalStorageState = Field(ThermalStorageState.IDLE, title="TS,state", json_schema_extra={"units": "-"},
                                            description="Input/Output. Thermal storage state. It can be used to define the Thermal Storage initial state, after it's always an output")
    current_state: SolarMED_State = Field(None, title="state", json_schema_extra={"units": "-"},
                                            description="Output. Current state of the SolarMED system")

    _med_fsm: MedFSM = PrivateAttr(None)
    _sf_ts_state: SF_TS_State = PrivateAttr(None)
    _created_at: datetime = PrivateAttr(default_factory=datetime.datetime.now)

    model_config = ConfigDict(
        validate_assignment=True,  # So that fields are validated, not only when created, but every time they are set
        arbitrary_types_allowed=True
        # numpy.ndarray[typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=110)])]]
    )

    def model_post_init(self, ctx):

        initial_sf_ts = SF_TS_State(str(self.sf_state.value) + str(self.ts_state.value))
        self.current_state = SolarMED_State(initial_sf_ts.value + str(self.med_state.value))

        self._sf_ts_fsm: SolarFieldWithThermalStorage_FSM = SolarFieldWithThermalStorage_FSM(
            name='SolarFieldWithThermalStorage_FSM', initial_state=initial_sf_ts)
        self._med_fsm: MedFSM = MedFSM(name='MED_FSM', initial_state=self.med_state)

    def step(
            self,
            # Thermal storage decision variables
            mts_src: float,
            # Solar field decision variables
            Tsf_out: float,
            # MED decision variables
            mmed_s, mmed_f, Tmed_s_in, Tmed_c_out, med_vacuum_state: MedVacuumState | int,
            # Environment variables
            Tmed_c_in: float, Tamb: float, I: float, wmed_f: float = None,
            # Optional
            msf: float = None,
            # Optional, to provide the solar field flow rate when starting up (Tsf_out takes priority)
    ):
        # Validation of inputs
        # In this mockup class just copy the inputs into class variables
        self.mts_src = mts_src
        self.Tsf_out = Tsf_out
        self.mmed_s = mmed_s
        self.mmed_f = mmed_f
        self.Tmed_s_in = Tmed_s_in
        self.Tmed_c_out = Tmed_c_out
        self.med_vacuum_state = med_vacuum_state

        # After the validation, variables are either zero or within the limits (>0),
        # based on this, the step method in the individual state machine are called

        self._sf_ts_fsm.step(Tsf_out=Tsf_out, qts_src=mts_src)
        self._med_fsm.step(mmed_s=mmed_s, mmed_f=mmed_f, Tmed_s_in=Tmed_s_in, Tmed_c_out=Tmed_c_out,
                          med_vacuum_state=med_vacuum_state)

        self.update_current_state()
        logger.info(f"SolarMED current state: {self.current_state}")

    # def generate_state_code(self, sf_ts_state: SF_TS_State, med_state: MedState):
    #     # Make sure our states are of the correct type
    #     if not isinstance(sf_ts_state, SF_TS_State):
    #         sf_ts_state = getattr(SF_TS_State, str(sf_ts_state))
    #     if not isinstance(med_state, MedState):
    #         med_state = getattr(MedState, str(med_state))
    #
    #     return f"{sf_ts_state.value}{med_state.value}"

    def get_current_state(self, mode: Literal['default', 'human_readable'] = 'defualt') -> SolarMED_State:
        # state_code = self.generate_state_code(self._sf_ts_fsm.state, self._med_fsm.state)

        state_code = str(self.sf_state.value) + str(self.ts_state.value) + str(self.med_state.value)

        if mode == 'human_readable':
            state_str = SolarMED_State(state_code).name
            # Replace _ by space and make everything minusculas
            state_str =  state_str.replace('_', ' ').lower()
            # Replace ts to TS, sf to SF and med to MED
            state_str = state_str.replace('ts', 'TS').replace('sf', 'SF').replace('med', 'MED')

            return state_str

        else:
            return SolarMED_State(state_code)

    def update_internal_states(self) -> None:
        self.med_state = self._med_fsm.state

        sf_ts_state: SF_TS_State = self._sf_ts_fsm.machine.get_state(self._sf_ts_fsm.state).value # Precioso
        self.sf_state = SolarFieldState(int(sf_ts_state.value[0]))
        self.ts_state = ThermalStorageState(int(sf_ts_state.value[1]))

    def update_current_state(self) -> None:
        self.update_internal_states()
        self.current_state = self.get_current_state()


    def to_dataframe(self, df: pd.DataFrame = None):
        # Return some of the internal variables as a dataframe
        # the state as en Enum?str?, the inputs, the consumptions

        if df is None:
            df = pd.DataFrame()

        data = pd.DataFrame({
            'state': self.current_state.name,
            'state_title': self.get_current_state(mode='human_readable'),

            'state_med': self.med_state,
            'mmed_s': self.mmed_s,
            'mmed_f': self.mmed_f,
            'Tmed_s_in': self.Tmed_s_in,
            'Tmed_c_out': self.Tmed_c_out,

            'state_sf': self.sf_state,
            'state_ts': self.ts_state,
            'state_sf_ts': SF_TS_State(str(self.sf_state.value) + str(self.ts_state.value)),
            'Tsf_out': self.Tsf_out,
            'qts_src': self.mts_src,
        }, index=[0])

        df = pd.concat([df, data], ignore_index=True)

        return df
