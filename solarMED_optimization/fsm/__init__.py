from typing import Literal
from collections.abc import Callable
from pathlib import Path
import pandas as pd
import numpy as np
import transitions as tr

from transitions.extensions import GraphMachine as Machine
from loguru import logger

from solarMED_optimization import MedVacuumState, MedState, SF_TS_State, SolarMED_State


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
        # self.machine.machine_attributes['ratio'] = '0.3'
        # self.machine.machine_attributes['rankdir'] = 'TB'
        # self.machine.style_attributes['node']['transient'] = {'fillcolor': '#FBD385'}
        # self.machine.style_attributes['node']['steady'] = {'fillcolor': '#E0E8F1'}
        super(self.__class__, self).customize_fsm_style()

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


class SolarMED():
    """

    This class is a template for the one in models_psa.
    It should act like a wrapper around the two individual finite state machines (fsm), and depending on the inputs given to the step method, call the correct events in the individual fsms. It should also provide utility methods like getting the current state of the system, information like the number of complete cycles, etc.

    """

    # states: list[Enum] = [state for state in SolarMED_State]
    current_state: SolarMED_State = None

    def __init__(self):
        self.sf_ts_fsm: SolarFieldWithThermalStorage_FSM = SolarFieldWithThermalStorage_FSM(
            name='SolarFieldWithThermalStorage_FSM', initial_state=SF_TS_State.IDLE)
        self.med_fsm: MedFSM = MedFSM(name='MED_FSM', initial_state=MedState.OFF)

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

        self.sf_ts_fsm.step(Tsf_out=Tsf_out, qts_src=mts_src)
        self.med_fsm.step(mmed_s=mmed_s, mmed_f=mmed_f, Tmed_s_in=Tmed_s_in, Tmed_c_out=Tmed_c_out,
                          med_vacuum_state=med_vacuum_state)

        self.update_current_state()
        logger.info(f"SolarMED current state: {self.current_state}")

    def generate_state_code(self, sf_ts_state: SF_TS_State, med_state: MedState):
        # Make sure our states are of the correct type
        if not isinstance(sf_ts_state, SF_TS_State):
            sf_ts_state = getattr(SF_TS_State, str(sf_ts_state))
        if not isinstance(med_state, MedState):
            med_state = getattr(MedState, str(med_state))

        return f"{sf_ts_state.value}{med_state.value}"

    def get_current_state(self, mode: Literal['default', 'human_readable'] = 'defualt') -> SolarMED_State:
        state_code = self.generate_state_code(self.sf_ts_fsm.state, self.med_fsm.state)

        if mode == 'human_readable':
            state_str = SolarMED_State(state_code).name
            # Replace _ by space and make everything minusculas
            state_str =  state_str.replace('_', ' ').lower()
            # Replace ts to TS, sf to SF and med to MED
            state_str = state_str.replace('ts', 'TS').replace('sf', 'SF').replace('med', 'MED')

            return state_str

        else:
            return SolarMED_State(state_code)

    def update_current_state(self) -> None:
        self.current_state = self.get_current_state()

    def to_dataframe(self, df: pd.DataFrame = None):
        # Return some of the internal variables as a dataframe
        # the state as en Enum?str?, the inputs, the consumptions

        if df is None:
            df = pd.DataFrame()

        data = pd.DataFrame({
            'state': self.current_state.name,
            'state_title': self.get_current_state(mode='human_readable'),

            'state_med': self.med_fsm.state if isinstance(self.med_fsm.state, MedState) else getattr(MedState, self.med_fsm.state),
            'mmed_s': self.mmed_s,
            'mmed_f': self.mmed_f,
            'Tmed_s_in': self.Tmed_s_in,
            'Tmed_c_out': self.Tmed_c_out,

            'state_sf_ts': self.sf_ts_fsm.state if isinstance(self.sf_ts_fsm.state, SF_TS_State) else getattr(SF_TS_State, self.sf_ts_fsm.state),
            'Tsf_out': self.Tsf_out,
            'qts_src': self.mts_src,
        }, index=[0])

        df = pd.concat([df, data], ignore_index=True)

        return df
