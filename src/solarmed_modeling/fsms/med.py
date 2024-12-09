from enum import Enum
from typing import Literal, Type
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import math
from loguru import logger

from . import BaseFsm, MedState, MedVacuumState, FsmInputs as BaseFsmInputs
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
    off_cooldown_time: int = 12 * 3600 # Time to wait before activating the MED plant again after shutting it off (12 hours, in seconds)
    active_cooldown_time: int = 2 * 3600 # Time to wait before activating the MED plant again after shutting it off / suspending it
    
    startup_conditions: FsmStartupConditions = field(default_factory=lambda: FsmStartupConditions())
    shutdown_conditions: FsmShutdownConditions = field(default_factory=lambda: FsmShutdownConditions())

@dataclass
class FsmInternalState:
    """
    TLDR: Avoid setting the internal state manually.
    
    Notes:
    - No checks are done in the internal state, if an incorrect start_sample value is set,
    it does not matter for example when activating that transitions since it will rewrite the value.
    However, for example if the initial state is set to IDLE, while vacuum_generated is False, it
    migth cause unexpected behavior.
    
    """
    # Vacuum
    vacuum_generated: bool = False
    vacuum_elapsed_samples: int = 0
    
    # Shutdown
    brine_empty: bool = True
    brine_emptying_elapsed_samples: int = 0
    
    # Startup
    startup_done: bool = False
    startup_elapsed_samples: int = 0
    
    # Active
    active_cooldown_done: bool = True
    active_cooldown_elapsed_samples: int = 0
    
    # Off
    off_cooldown_done: bool = True
    off_cooldown_elapsed_samples: int = 0

@dataclass
class FsmInputs(BaseFsmInputs):
    """
    Inputs / Decision variables
    """
    med_active: bool
    med_vacuum_state: MedVacuumState
    
    # def __post_init__(self):
    #     super().__post_init__()
        
    #     # Make sure med_vacuum_state is of the correct type
    #     self.med_vacuum_state: MedVacuumState = self._convert_to(self.med_vacuum_state,
    #                                                             state_cls=MedVacuumState,
    #                                                             return_format="enum")
        
    # def dump_as_array(self):
        
        # return super().dump_as_array()
        # med_vacuum_state = self.med_vacuum_state.value if self.med_vacuum_state is not None else None
        # return np.array([float(x) if x is not None else 0.0 for x in [self.med_active, med_vacuum_state]], dtype=float)


class MedFsm(BaseFsm):

    """
    Finite State Machine for the Multi-Effect Distillation (MED) unit.
    """

    params: FsmParameters # to have type hints
    internal_state: FsmInternalState # to have type hints
    _state_type: MedState = MedState  # State type
    _inputs_cls: Type = FsmInputs # Inputs class
    _cooldown_callbacks: list[str] = ['is_active_cooldown_done', 'is_off_cooldown_done']
    _counter_callbacks: list[str] = ['is_vacuum_done', 'is_startup_done', 'is_brine_empty']

    def __init__(
            self, 
            sample_time: int, 
            name: str = "MED_FSM",
            initial_state: MedState = MedState.OFF,
            current_sample: int = 0,
            params: FsmParameters = FsmParameters(),
            internal_state: FsmInternalState = FsmInternalState(),

            # Inputs / Decision variables (optional, use to set prior inputs)
            # qmed_s: float = None,
            # qmed_f: float = None,
            # Tmed_s_in: float = None,
            # Tmed_c_out: float = None,
            # med_active: bool = None,
            # med_vacuum_state: MedVacuumState = None,
            inputs: FsmInputs = None,
    ) -> None:
        
        # Call parent constructor
        super().__init__(name=name, initial_state=initial_state, sample_time=sample_time,
                         current_sample=current_sample, internal_state=internal_state,
                         params=params, inputs=inputs)

        # Convert duration times to samples
        self.vacuum_duration_samples = math.ceil(self.params.vacuum_duration_time / self.sample_time)
        self.brine_emptying_samples = math.ceil(self.params.brine_emptying_time / self.sample_time)
        self.startup_duration_samples = math.ceil(self.params.startup_duration_time / self.sample_time)
        self.active_cooldown_duration_samples = math.ceil(self.params.active_cooldown_time / self.sample_time)
        self.off_cooldown_duration_samples = math.ceil(self.params.off_cooldown_time / self.sample_time)

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        # self.qmed_s = qmed_s
        # self.qmed_f = qmed_f
        # self.Tmed_s_in = Tmed_s_in
        # self.Tmed_c_out = Tmed_c_out
        # if self.inputs is None:
        #     self.inputs = FsmInputs(
        #         med_active = False,
        #         med_vacuum_state=MedVacuumState.OFF
        #     )
            # if med_vacuum_state is not None:
            #     
            # else:
            #     self.inputs.med_vacuum_state = 

        # inputs_array = self.update_inputs_array()
        # self.inputs_array_prior = inputs_array

        if initial_state in [MedState.ACTIVE, MedState.STARTING_UP]:
            self.internal_state.brine_empty = False

        # States
        st = self._state_type # alias

        # Not enterily sure if set_active_cooldown_done and set_off_cooldown_done
        # really bring something, if we want to have some internal variable to keep track of the 
        # internal state, we can just set it in the condition callback
        self.machine.add_state(st.OFF, on_enter=["set_off_cooldown_start"]) # , on_exit=["set_off_cooldown_done"])#, on_enter=['reset_fsm'])
        self.machine.add_state(st.GENERATING_VACUUM)
        self.machine.add_state(st.IDLE,) # on_enter=["set_idle_cooldown"])
        self.machine.add_state(st.STARTING_UP, on_enter=['set_brine_non_empty'])
        self.machine.add_state(st.ACTIVE, on_exit=["set_active_cooldown_start"])# , on_enter=["set_active_cooldown_done"])#, on_exit=['reset_startup'])
        self.machine.add_state(st.SHUTTING_DOWN) #, on_exit=['set_brine_empty'])
        
        ## State inputs sets
        self.states_inputs_set: dict[str|int, FsmInputs] = {
            "OFF": FsmInputs(med_active=False, med_vacuum_state=0),
            "GENERATING_VACUUM": FsmInputs(med_active=False, med_vacuum_state=2),
            "IDLE": FsmInputs(med_active=False, med_vacuum_state=1),
            "STARTING_UP": FsmInputs(med_active=True, med_vacuum_state=1),
            "SHUTTING_DOWN": FsmInputs(med_active=False, med_vacuum_state=1),
            "ACTIVE": FsmInputs(med_active=True, med_vacuum_state=1),
        }
        self.default_inputs: FsmInputs = self.states_inputs_set["OFF"]

        # Transitions
        ## Vacuum
        self.machine.add_transition('start_generating_vacuum', source=st.OFF, dest=st.GENERATING_VACUUM,
                                    conditions=['is_high_vacuum', 'is_off_cooldown_done'], after=['set_vacuum_start'])
        self.machine.add_transition('finish_generating_vacuum', source=st.GENERATING_VACUUM, dest=st.IDLE,
                                    conditions=['is_vacuum_done'], unless=['are_inputs_active'])#, after='set_vacuum_done')
        # To avoid staying more than strictly needed in this transitionary state
        self.machine.add_transition('generating_vacuum_interrupted', source=st.GENERATING_VACUUM, dest=st.OFF,
                                    unless=['is_high_vacuum', 'is_vacuum_done'])
        # self.machine.add_transition('cancel_generating_vacuum', source=st.GENERATING_VACUUM, dest=st.OFF,
        #                             conditions=['is_off_vacuum']) # Removed to reduce the FSM posibilities
        ## Start-up
        self.machine.add_transition('start_startup', source=[st.IDLE, st.GENERATING_VACUUM], dest=st.STARTING_UP,
                                    conditions=['is_vacuum_done', 'are_inputs_active', 'is_active_cooldown_done'], after='set_startup_start')
        self.machine.add_transition('finish_startup', source=st.STARTING_UP, dest=st.ACTIVE,
                                    conditions=['are_inputs_active', 'is_startup_done'])
        # To avoid staying more than strictly needed in this transitionary state
        self.machine.add_transition('startup_interrupted', source=st.STARTING_UP, dest=st.IDLE,
                                    unless=['are_inputs_active'])#, after='set_startup_done')
        ## Shutdown
        self.machine.add_transition('start_shutdown', source=st.ACTIVE, dest=st.SHUTTING_DOWN,
                                    conditions=["is_off_vacuum"],
                                    unless=['are_inputs_active'], after='set_brine_emptying_start')
        self.machine.add_transition('start_suspend', source=st.ACTIVE, dest=st.SHUTTING_DOWN,
                                    conditions=["is_low_vacuum"],
                                    unless=['are_inputs_active'], after='set_brine_emptying_start')
        self.machine.add_transition('finish_shutdown', source=[st.SHUTTING_DOWN, st.IDLE], dest=st.OFF,
                                    conditions=['is_off_vacuum', 'is_brine_empty'])  # Since destination is OFF already resets FSM
        self.machine.add_transition('finish_suspend', source=st.SHUTTING_DOWN, dest=st.IDLE,
                                    conditions=['is_low_vacuum', 'is_brine_empty'])#, unless=['is_off_vacuum'])# , after='set_brine_empty')

        # Validate inputs or set default values 
        self.validate_or_set_inputs()
        
        # Additional
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
    ## General
    def inform_wasteful_operation(self, *args) -> None:
        """
        Inform of not invalid but wasteful operations 
        This is supposed to be called from parent class
        """

        event = args[0]

        # Inform of not invalid but wasteful operations
        if self.internal_state.vacuum_generated and self.is_high_vacuum():
            logger.warning(f"[{self.name}] Vacuum already generated, keeping vacuum at high value is wasteful")
        if event.state == MedState.OFF and self.is_low_vacuum():
            logger.warning(f"[{self.name}] MED vacuum state is OFF, vacuum should be off or high to start generating vacuum")
        if event.state in [MedState.SHUTTING_DOWN, MedState.IDLE, MedState.OFF] and self.are_inputs_valid():
            logger.warning(f"[{self.name}] MED is not operating, there is no point in having its inputs active")
    
    ## Active
    def set_active_cooldown_start(self, *args) -> None:
        """ Set the cooldown for the active state """
        self.internal_state.active_cooldown_elapsed_samples = 0
        self.internal_state.active_cooldown_done = False
        logger.info(f"[{self.name}] active cooldown started")

    # def set_active_cooldown_done(self, *args):
    #     self.internal_state.active_cooldown_done = True
    #     logger.info(f"[{self.name}] Active cooldown done")
        
    ## Off
    def set_off_cooldown_start(self, *args) -> None:
        """ Set the cooldown for the off state """
        self.internal_state.off_cooldown_elapsed_samples = 0
        self.internal_state.off_cooldown_done = False
        logger.info(f"[{self.name}] off cooldown started")
        
    # def set_off_cooldown_done(self, *args):
    #     self.internal_state.off_cooldown_done = True
    #     logger.info(f"[{self.name}] Off cooldown done")
        
    ## Vacuum
    def set_vacuum_start(self, *args) -> None:
        """ Set the start of the vacuum generation """
        
        # if self.internal_state.generating_vacuum:
        #     logger.warning(f"[{self.name}] Already generating vacuum, no need to start again")
        #     return

        # Else
        self.internal_state.vacuum_generated = False
        self.internal_state.vacuum_elapsed_samples = 0 # 0 samples, it will be incremented in the first check in this same step
        logger.info(f"[{self.name}] Started generating vacuum. {self.internal_state.vacuum_elapsed_samples}/{self.vacuum_duration_samples} samples to complete")
    
    # def set_vacuum_reset(self, *args) -> None:
    #     """ Reset the vacuum generation """
    #     self.internal_state.vacuum_elapsed_samples = 0
    #     self.internal_state.vacuum_generated = False
    #     logger.info(f"[{self.name}] Cancelled vacuum generation")

    # def set_vacuum_done(self, *args):
    #     self.internal_state.vacuum_generated = True
    #     # self.internal_state.vacuum_elapsed_samples = self.vacuum_duration_samples
    #     logger.info(f"[{self.name}] Vacuum generated")

    ## Shutdown
    def set_brine_emptying_start(self, *args) -> None:
        """ Start emptying the brine """
        
        # if self.internal_state.brine_empty:
        #     logger.warning(f"[{self.name}] Brine is already empty, no need to start emptying again")
        #     return

        self.internal_state.brine_empty = False
        self.internal_state.brine_emptying_elapsed_samples = 0
        logger.info(f"[{self.name}] Started emptying brine. {self.internal_state.brine_emptying_elapsed_samples}/{self.brine_emptying_samples} samples to complete")

    # def set_brine_empty(self, *args) -> None:
    #     """ Set the brine as empty """
        
    #     self.internal_state.brine_empty = True
    #     self.internal_state.brine_emptying_elapsed_samples = 0
    #     logger.info(f"[{self.name}] Brine emptied")

    def set_brine_non_empty(self, *args) -> None:
        """ Set the brine as non-empty """
        
        self.internal_state.brine_empty = False
        self.internal_state.brine_emptying_elapsed_samples = 0
        logger.info(f"[{self.name}] Brine non-empty")

    ## Startup
    def set_startup_start(self, *args) -> None:
        """ Start the MED plant """
        
        # if self.internal_state.startup_done:
        #     logger.warning(f"[{self.name}] Startup already done, no need to start again")
        #     return
        
        self.internal_state.startup_done = False
        self.internal_state.startup_elapsed_samples = 0
        logger.info(f"[{self.name}] Started starting up. {self.internal_state.startup_elapsed_samples}/{self.startup_duration_samples} samples to complete")

    # def set_startup_done(self, *args) -> None:
    #     """ Set the MED plant as started up """
        
    #     self.internal_state.startup_done = True
    #     # self.internal_state.startup_elapsed_samples = self.startup_duration_samples
    #     logger.info(f"[{self.name}] Startup done")

    # def reset_startup(self, *args):
    #     self.internal_state.startup_done = False
    #     self.internal_state.startup_elapsed_samples = 0
    #     logger.info(f"[{self.name}] Startup reset")

    # State machine transition conditions
    ## Active
    def is_active_cooldown_done(self, *args) -> bool:
        """ Check if the active cooldown is done """
        
        if self.internal_state.active_cooldown_done:
            return True
        
        cooldown_done, self.internal_state.active_cooldown_elapsed_samples = self.check_elapsed_samples(
            elapsed_samples = self.internal_state.active_cooldown_elapsed_samples, 
            samples_duration = self.active_cooldown_duration_samples, 
            msg = "active cooldown",
        )
        
        self.internal_state.active_cooldown_done = cooldown_done
        
        return cooldown_done
    
    ## Off
    def is_off_cooldown_done(self, *args) -> bool:
        """ Check if the off cooldown is done """
        
        if self.internal_state.off_cooldown_done:
            return True
        
        cooldown_done, self.internal_state.off_cooldown_elapsed_samples = self.check_elapsed_samples(
            elapsed_samples = self.internal_state.off_cooldown_elapsed_samples, 
            samples_duration = self.off_cooldown_duration_samples, 
            msg = "off cooldown",
        )
        
        # Alternative instead of using an specific callback
        # Simplifies machine code
        self.internal_state.off_cooldown_done = cooldown_done
        
        return cooldown_done
    
    ## Vacuum
    def is_high_vacuum(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:  # , raise_error: bool = False):
        """ Check if the vacuum is high """
        
        if return_valid_inputs:
            return dict(med_vacuum_state = MedVacuumState.HIGH)
        elif return_invalid_inputs:
            return dict(med_vacuum_state = MedVacuumState.OFF) # Could also be LOW
        
        return self.inputs.med_vacuum_state == MedVacuumState.HIGH
        
    def is_low_vacuum(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Check if the vacuum is low """
        
        if return_valid_inputs:
            return dict(med_vacuum_state = MedVacuumState.LOW)
        elif return_invalid_inputs:
            # Should not be HIGH! since it will still return a valid transition in some cases
            return dict(med_vacuum_state = MedVacuumState.OFF)
            
        return self.inputs.med_vacuum_state == MedVacuumState.LOW
                    
    def is_off_vacuum(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Check if the vacuum is off """

        if return_valid_inputs:
            return dict(med_vacuum_state = MedVacuumState.OFF)
        elif return_invalid_inputs:
            return dict(med_vacuum_state = MedVacuumState.HIGH) # Could also be LOW
        
        return self.inputs.med_vacuum_state == MedVacuumState.OFF

    def is_vacuum_done(self, *args) -> bool:
        """ Check if the vacuum generation is done """
        
        if self.internal_state.vacuum_generated:
            return True

        vacuum_done, self.internal_state.vacuum_elapsed_samples = self.check_elapsed_samples(
            elapsed_samples = self.internal_state.vacuum_elapsed_samples, 
            samples_duration = self.vacuum_duration_samples, 
            msg = "generating vacuum"
        )
        
        self.internal_state.vacuum_generated = vacuum_done
        
        return vacuum_done
    
    ## Startup
    def is_startup_done(self, *args) -> bool:
        """ Check if the MED plant has started up """
        
        if self.internal_state.startup_done:
            return True
        
        startup_done, self.internal_state.startup_elapsed_samples = self.check_elapsed_samples(
            elapsed_samples = self.internal_state.startup_elapsed_samples, 
            samples_duration = self.startup_duration_samples, 
            msg = "starting up"
        )
        
        self.internal_state.startup_done = startup_done
        
        return startup_done

    ## Shutdown
    def is_brine_empty(self, *args) -> bool:
        """ Check if the brine is empty """
        
        if self.internal_state.brine_empty:
            return True
        
        brine_empty, self.internal_state.brine_emptying_elapsed_samples = self.check_elapsed_samples(
            elapsed_samples = self.internal_state.brine_emptying_elapsed_samples, 
            samples_duration = self.brine_emptying_samples, 
            msg = "emptying brine"
        )
        
        self.internal_state.brine_empty = brine_empty
        
        return brine_empty

    ## General
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

        return self.are_inputs_valid() and self.inputs.med_vacuum_state != MedVacuumState.OFF

    def are_inputs_valid(self, *args, return_valid_inputs: bool = False, return_invalid_inputs: bool = False) -> bool | dict:
        """ Just check if the inputs are greater than zero, not the vacuum """
        
        if return_valid_inputs:
            return dict(
                # qmed_s = 1.0,
                # qmed_f = 1.0,
                # Tmed_s_in = 1.0,
                # Tmed_c_out = 1.0,
                med_active = True
            )
        elif return_invalid_inputs:
            return dict( # Just one would be enough
                # qmed_s = 0.0,
                # qmed_f = 0.0,
                # Tmed_s_in = 0.0,
                # Tmed_c_out = 0.0,
                med_active = False
            )
        
        return np.all(self.inputs_array > 0)

