from typing import Literal
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger

from . import BaseFsm, MedState, MedVacuumState
from solarmed_modeling.med import FixedModelParameters


@dataclass
class FsmStartupConditions:
    """
    Startup conditions for the Finite State Machines (FSM)
    """ 
    # MED
    qmed_s: float = FixedModelParameters().qmed_s_min # Heat source flow rate (m³/h)
    qmed_f: float = FixedModelParameters().qmed_f_min # Feed water flow rate (m³/h)
    qmed_b: float = FixedModelParameters().qmed_f_min  # Brine extraction flow rate (m³/h)
    qmed_c: float = 0.0  # Cooling water flow rate (m³/h)
    Tmed_s_in: float = FixedModelParameters().Tmed_s_min # Heat source inlet temperature (ºC)
    

@dataclass
class FsmShutdownConditions:
    """
    Shutdown conditions for the Finite State Machines (FSM)
    """ 
    # MED
    qmed_s: float = 0.0 # Heat source flow rate (m³/h)
    qmed_f: float = 0.0  # Feed water flow rate (m³/h)
    qmed_b: float = 5.0  # Brine extraction flow rate (m³/h)
    qmed_c: float = 0.0  # Cooling water flow rate (m³/h)
    Tmed_s_in: float = 0.0 # Heat source inlet temperature (ºC)
    

@dataclass
class FsmParameters:
    vacuum_duration_time: int = 30 * 60  # Time to generate vacuum in the MED system (seconds)
    brine_emptying_time: int = 60 * 60  # Time to extract brine from MED plant (seconds)
    startup_duration_time: int = 30 * 60  # Time to start up the MED plant (seconds)
    # deactivate_cooldown
    # activate_cooldown
    
    startup_conditions: FsmStartupConditions = field(default_factory=lambda: FsmStartupConditions())
    shutdown_conditions: FsmShutdownConditions = field(default_factory=lambda: FsmShutdownConditions())

@dataclass
class FsmInitialStates:
    vacuum_generated: bool = False
    vacuum_started_sample: int = 0
    brine_empty: bool = True
    brine_emptying_started_sample: bool = 0
    startup_done: bool = False
    startup_started_sample: int = 0

class MedFsm(BaseFsm):

    """
    Finite State Machine for the Multi-Effect Distillation (MED) unit.
    """

    # sample_rate: int = 1  # seconds
    # current_sample = 0
    # fsm_params: FsmParameters = FsmParameters(),
    # fsm_initial_states: FsmInitialStates = FsmInitialStates(),

    # Vacuum
    generating_vacuum: bool = False
    vacuum_generated: bool = False
    vacuum_started_sample: int = 0

    # Shutdown
    brine_empty: bool = True
    brine_emptying_started_sample: bool = 0

    # Startup
    startup_done: bool = False
    startup_started_sample: int = 0

    # State type
    _state_type: MedState = MedState


    def __init__(
            self, sample_time: int,
            # After refactor should be
            # fsm_params: FsmParameters = FsmParameters(),
            # fsm_initial_states: FsmInitialStates = FsmInitialStates(),
            vacuum_duration_time: int,  # seconds
            brine_emptying_time: int,  # seconds
            startup_duration_time: int,  # seconds
            name: str = "MED_FSM",
            initial_state: MedState = MedState.OFF,

            # Inputs / Decision variables (Optional)
            qmed_s: float = None,
            qmed_f: float = None,
            Tmed_s_in: float = None,
            Tmed_c_out: float = None,
            med_vacuum_state: MedVacuumState = None,
     ) -> None:

        # Call parent constructor
        super().__init__(name, initial_state, sample_time)

        self.vacuum_duration_time = vacuum_duration_time
        self.brine_emptying_time = brine_emptying_time
        self.startup_duration_time = startup_duration_time

        self.vacuum_duration_samples = int(self.vacuum_duration_time / self.sample_time)
        self.brine_emptying_samples = int(self.brine_emptying_time / self.sample_time)
        self.startup_duration_samples = int(self.startup_duration_time / self.sample_time)

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.qmed_s = qmed_s
        self.qmed_f = qmed_f
        self.Tmed_s_in = Tmed_s_in
        self.Tmed_c_out = Tmed_c_out
        self.med_vacuum_state = med_vacuum_state

        inputs_array = self.update_inputs_array()
        self.inputs_array_prior = inputs_array

        if initial_state in [MedState.ACTIVE, MedState.STARTING_UP]:
            self.brine_empty = False

        # States
        # OFF = 0
        # IDLE = 1
        # ACTIVE = 2
        st = self._state_type # alias

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
        # self.machine.add_transition('cancel_generating_vacuum', source=st.GENERATING_VACUUM, dest=st.OFF,
        #                             conditions=['is_off_vacuum']) # Removed to reduce the FSM posibilities

        # Start-up
        self.machine.add_transition('start_startup', source=st.IDLE, dest=st.STARTING_UP,
                                    conditions=['are_inputs_active'], after='set_startup_start')
        self.machine.add_transition('finish_startup', source=st.STARTING_UP, dest=st.ACTIVE,
                                    conditions=['are_inputs_active', 'is_startup_done'], after='set_startup_done')

        # Shutdown
        self.machine.add_transition('start_shutdown', source=st.ACTIVE, dest=st.SHUTTING_DOWN,
                                    unless=['are_inputs_active'], after='set_brine_emptying_start')
        self.machine.add_transition('finish_shutdown', source=[st.SHUTTING_DOWN, st.IDLE], dest=st.OFF,
                                    conditions=['is_off_vacuum', 'is_brine_empty'])  # Since destination is OFF already resets FSM
        self.machine.add_transition('finish_suspend', source=st.SHUTTING_DOWN, dest=st.IDLE,
                                    conditions=['is_brine_empty'], unless=['is_off_vacuum'], after='set_brine_empty')

        self.customize_fsm_style()

    def customize_fsm_style(self) -> None:
        """ Custom styling of state machine graph """
        
        super(self.__class__, self).customize_fsm_style()
        
        self.machine.machine_attributes['ratio'] = '0.05'
        # customize node styling
        model_ = list(self.machine.model_graphs.keys())[0]  # lavin
        for s in [MedState.GENERATING_VACUUM, MedState.STARTING_UP, MedState.SHUTTING_DOWN]:
            self.machine.model_graphs[model_].set_node_style(s, 'transient')
        for s in [MedState.OFF, MedState.IDLE, MedState.ACTIVE]:
            self.machine.model_graphs[model_].set_node_style(s, 'steady')

    # State machine actions - callbacks of states and transitions

    def inform_wasteful_operation(self, *args) -> None:
        """
        Inform of not invalid but wasteful operations 
        This is supposed to be called from parent class
        """

        event = args[0]

        # Inform of not invalid but wasteful operations
        if self.vacuum_generated and self.is_high_vacuum():
            logger.warning(f"[{self.name}] Vacuum already generated, keeping vacuum at high value is wasteful")
        if event.state == MedState.OFF and self.is_low_vacuum():
            logger.warning(f"[{self.name}] MED vacuum state is OFF, vacuum should be off or high to start generating vacuum")
        if event.state in [MedState.SHUTTING_DOWN, MedState.IDLE, MedState.OFF] and self.are_inputs_valid():
            logger.warning(f"[{self.name}] MED is not operating, there is no point in having its inputs active")

    def reset_fsm(self, *args) -> None:
        logger.info(f"[{self.name}] Resetting FSM")
        self.set_vacuum_reset()
        # self.set_brine_empty()
        # self.reset_startup()
        # Anything else?

    # Vacuum
    def set_vacuum_start(self, *args) -> None:
        """ Set the start of the vacuum generation """
        
        if self.generating_vacuum:
            logger.warning(f"[{self.name}] Already generating vacuum, no need to start again")
            return

        # Else
        self.vacuum_started_sample = self.current_sample
        logger.info(f"[{self.name}] Started generating vacuum, it will take {self.vacuum_duration_samples} samples to complete")

    def set_vacuum_reset(self, *args) -> None:
        """ Reset the vacuum generation """
        self.vacuum_started_sample = None
        logger.info(f"[{self.name}] Cancelled vacuum generation")

    def set_vacuum_done(self, *args):
        self.vacuum_generated = True
        logger.info(f"[{self.name}] Vacuum generated")

    # Shutdown
    def set_brine_emptying_start(self, *args) -> None:
        """ Start emptying the brine """
        
        if self.brine_empty:
            logger.warning(f"[{self.name}] Brine is already empty, no need to start emptying again")
            return

        self.brine_emptying_started_sample = self.current_sample
        logger.info(f"[{self.name}] Started emptying brine, it will take {self.brine_emptying_samples} samples to complete")

    def set_brine_empty(self, *args) -> None:
        """ Set the brine as empty """
        
        self.brine_empty = True
        self.brine_emptying_started_sample = None
        logger.info(f"[{self.name}] Brine emptied")

    def set_brine_non_empty(self, *args) -> None:
        """ Set the brine as non-empty """
        
        self.brine_empty = False
        logger.info(f"[{self.name}] Brine non-empty")

    # Startup
    def set_startup_start(self, *args) -> None:
        """ Start the MED plant """
        
        if self.startup_done:
            logger.warning(f"[{self.name}] Startup already done, no need to start again")
            return

        self.startup_started_sample = self.current_sample
        logger.info(f"[{self.name}] Started starting up, it will take {self.startup_duration_samples} samples to complete")

    def set_startup_done(self, *args) -> None:
        """ Set the MED plant as started up """
        
        self.startup_done = True
        logger.info(f"[{self.name}] Startup done")

    def reset_startup(self, *args):
        self.startup_done = False
        self.startup_started_sample = None
        logger.info(f"[{self.name}] Startup reset")

    # State machine transition conditions
    # Vacuum
    def is_high_vacuum(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:  # , raise_error: bool = False):
        """ Check if the vacuum is high """
        
        if return_valid_inputs:
            return dict(med_vacuum_state = MedVacuumState.HIGH)
        elif return_invalid_inputs:
            return dict(med_vacuum_state = MedVacuumState.OFF) # Could also be LOW
        
        return self.med_vacuum_state == MedVacuumState.HIGH
        

    def is_low_vacuum(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Check if the vacuum is low """
        
        if return_valid_inputs:
            return dict(med_vacuum_state = MedVacuumState.LOW)
        elif return_invalid_inputs:
            return dict(med_vacuum_state = MedVacuumState.OFF) # Should not be HIGH! since it will still return a valid transition in some cases
            
        return self.med_vacuum_state == MedVacuumState.LOW
                    


    def is_off_vacuum(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Check if the vacuum is off """

        if return_valid_inputs:
            return dict(med_vacuum_state = MedVacuumState.OFF)
        elif return_invalid_inputs:
            return dict(med_vacuum_state = MedVacuumState.HIGH) # Could also be LOW
        
        return self.med_vacuum_state == MedVacuumState.OFF

    def is_vacuum_done(self, *args) -> bool:
        """ Check if the vacuum generation is done """
        
        if self.vacuum_generated:
            return True

        if self.current_sample - self.vacuum_started_sample >= self.vacuum_duration_samples:
            return True
        else:
            logger.info(
                f"[{self.name}] Still generating vacuum, {self.current_sample - self.vacuum_started_sample}/{self.vacuum_duration_samples} samples completed")
            return False

    def is_startup_done(self, *args) -> bool:
        """ Check if the MED plant has started up """
        if self.startup_done:
            return True

        if self.current_sample - self.startup_started_sample >= self.startup_duration_samples:
            return True
        else:
            logger.info(
                f"[{self.name}] Still starting up, {self.current_sample - self.startup_started_sample}/{self.startup_duration_samples} samples completed")
            return False

    def is_brine_empty(self, *args) -> bool:
        """ Check if the brine is empty """
        
        if self.brine_empty:
            return True

        if self.current_sample - self.brine_emptying_started_sample >= self.brine_emptying_samples:
            return True
        else:
            logger.info(
                f"[{self.name}] Still emptying brine, {self.current_sample - self.brine_emptying_started_sample}/{self.brine_emptying_samples} samples completed")
            return False

    def are_inputs_active(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Checks if all variables required for MED to active are valid (greater than zero and vacuum active) """

        if return_valid_inputs:
            inputs = self.are_inputs_valid(return_valid_inputs=True)
            inputs['med_vacuum_state'] = MedVacuumState.LOW

            return inputs

        elif return_invalid_inputs:
            inputs = self.are_inputs_valid(return_invalid_inputs = True)
            inputs['med_vacuum_state'] = MedVacuumState.OFF

            return inputs

        return self.are_inputs_valid() and self.med_vacuum_state != MedVacuumState.OFF

    def are_inputs_valid(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Just check if the inputs are greater than zero, not the vacuum """
        
        if return_valid_inputs:
            return dict(
                qmed_s = 1.0,
                qmed_f = 1.0,
                Tmed_s_in = 1.0,
                Tmed_c_out = 1.0,
            )
        elif return_invalid_inputs:
            return dict( # Just one would be enough
                qmed_s = 0.0,
                qmed_f = 0.0,
                Tmed_s_in = 0.0,
                Tmed_c_out = 0.0,
            )
        
        return np.all(self.inputs_array > 0)


    def get_inputs(self, format: Literal['array', 'dict'] = 'array') -> np.ndarray[float] | dict:
        
        super().get_inputs(format=format) # Just to check if the format is valid

        if format == 'array':
            """ When the array format is used, all variables necessarily need to be parsed as floats """

            med_vac_float = float(str(self.med_vacuum_state.value)) if self.med_vacuum_state is not None else None
            return np.array([self.qmed_s, self.qmed_f, self.Tmed_s_in, self.Tmed_c_out, med_vac_float], dtype=float)

        elif format == 'dict':
            """ In the dict format, each variable  can have its own type """

            return {
                'qmed_s': self.qmed_s,
                'qmed_f': self.qmed_f,
                'Tmed_s_in': self.Tmed_s_in,
                'Tmed_c_out': self.Tmed_c_out,
                'med_vacuum_state': self.med_vacuum_state,
            }

    def update_inputs_array(self) -> np.ndarray[float]:
        """ Update the inputs array """
        self.inputs_array = self.get_inputs(format='array')

        return self.inputs_array

    def step(self, qmed_s: float, qmed_f: float, Tmed_s_in: float, Tmed_c_out: float,
             med_vacuum_state: int | MedVacuumState) -> None:
        """ Move the state machine one step forward """

        self.current_sample += 1

        # Inputs validation (would be done by Pydantic), here just update the values
        self.qmed_s = qmed_s
        self.qmed_f = qmed_f
        self.Tmed_s_in = Tmed_s_in
        self.Tmed_c_out = Tmed_c_out
        self.med_vacuum_state = MedVacuumState(med_vacuum_state) if med_vacuum_state is not None else None

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.update_inputs_array()

        transition = self.get_next_valid_transition(prior_inputs=self.inputs_array_prior,
                                                    current_inputs=self.inputs_array)

        if transition is not None:
            transition()

        # Save prior inputs

        # self.qmed_s_prior = self.qmed_s
        # self.qmed_f_prior = self.qmed_f
        # self.Tmed_s_in_prior = self.Tmed_s_in
        # self.Tmed_c_out_prior = self.Tmed_c_out
        self.inputs_array_prior = self.inputs_array


    def to_dataframe(self, df: pd.DataFrame = None) -> pd.DataFrame:
        # Return some of the internal variables as a dataframe
        # the state as en Enum?str?, the inputs, the consumptions

        if df is None:
            df = pd.DataFrame()

        data = pd.DataFrame({
            'state': self.state.name,
            'qmed_s': self.qmed_s,
            'qmed_f': self.qmed_f,
            'Tmed_s_in': self.Tmed_s_in,
            'Tmed_c_out': self.Tmed_c_out,
        }, index=[0])

        df = pd.concat([df, data], ignore_index=True)

        return df

