import copy
import itertools
from dataclasses import asdict, dataclass, fields
import inspect
from typing import Callable, Literal, Type
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

from loguru import logger
import numpy as np
import pandas as pd
from transitions.extensions import GraphMachine as Machine

import logging
""" Disable
WARNING	transitions.core:core.py:_checked_assignment()- Skip binding of 'is_SF_HEATING_TS' to model due to model override policy.
WARNING	transitions.core:core.py:_checked_assignment()- Skip binding of 'is_IDLE' to model due to model override policy.
WARNING	transitions.core:core.py:_checked_assignment()- Skip binding of 'is_SF_HEATING_TS' to model due to model override policy.
"""
logging.getLogger("transitions").setLevel(logging.ERROR)

CombinationType = dict[str, bool | int]

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


# def ensure_type(expected_type: Type) -> callable:
#     def decorator(func):
#         def wrapper(self, value):
#             if not isinstance(value, expected_type):
#                 return getattr(expected_type, value)

#             return func(self, value)
#         return wrapper

@dataclass
class FsmParameters:
    """
    Parameters for the finite state machine
    """
    ...
    
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
    ...
    
@dataclass
class FsmInputs:
    """Inputs to the Finite State Machine (FSM)
    Placeholder class used for type hinting and validation of null values
    """
    ...
    
    def __post_init__(self, ):        
        assert all([value is not None for value in list(asdict(self).values())]), "Input value(s) cannot be None"
        
        # Make sure inputs are of the correct type
        from solarmed_modeling.fsms.utils import convert_to # Avoid circular import  
        self._convert_to = convert_to
        
        for field in fields(self):
            
            if issubclass(field.type, Enum):
                value = self._convert_to(getattr(self, field.name),
                                         state_cls=field.type,
                                         return_format="enum")
            else:
                value = field.type( getattr(self, field.name) )
            
            setattr(self, field.name, value)
        
    def dump_as_dict(self, format: Literal["enum", "value", "name"] = "enum") -> dict:
        """Dump the inputs as a dictionary

        Args:
            format (Literal['enum', 'value', 'name'], optional): Format of the output values. Defaults to "enum".

        Returns:
            dict: Dictionary representation of the inputs with the specified format.
        """
        
        output = asdict(self)
        
        for field in fields(self):
            field_value = output[field.name]
            if isinstance(field_value, Enum):
                output[field.name] = self._convert_to(field_value, field.type, return_format=format)
    
        return output
    
    def dump_as_array(self, ) -> np.ndarray[float]:
        """Dump the inputs as a numpy array

        Returns:
            np.ndarray[float]: Float array representation of the inputs
        """
        
        output: dict[str, float | int] = self.dump_as_dict(format="value")
        
        return np.array(list(output.values()), dtype=float)

def generate_combinations(inputs_cls: FsmInputs) -> CombinationType:
    """
    Generate all possible combinations of input values.
    
    Args:
    - inputs_cls: A class with fields representing the input variables. Their domains will
    be inferred from the field types (bool and discrete integer values from Enums).
    
    Returns:
    - A list of dictionaries, where each dictionary represents a unique combination of input values.
    """
    # keys, domains = zip(*asdict(inputs_cls).items())
    keys = []
    domains = []
    for field in fields(inputs_cls):
        keys.append(field.name)
        domains.append(
            [True, False] if field.type is bool else 
                [member.value for member in field.type] # Enum
        )
        
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*domains)
    ]
    return combinations


def evaluate_conditions(combination: CombinationType, conditions: list[CombinationType]) -> bool:
    """
    Check if a combination satisfies any of the given conditions.
    
    Args:
    - combination: A dictionary representing a specific input combination.
    - conditions: A list of conditions, where each condition is a list of dictionaries.
    
    Returns:
    - True if the combination satisfies any condition, False otherwise.
    """
    for condition in conditions:
        if all(all(combination.get(var, None) == val for var, val in clause.items()) for clause in condition):
            return True
    return False


def filter_invalid_combinations(inputs_cls: FsmInputs, conditions: list[CombinationType]) -> list[CombinationType]:
    """
    Generate the table of all input combinations and filter out those that satisfy conditions.
    
    Args:
    - variables: A dictionary mapping variable names to their possible values.
    - conditions: A list of conditions, where each condition is a list of dictionaries.
    
    Returns:
    - A list of dictionaries representing the invalid combinations.
    """
    # Step 1: Generate all possible combinations
    combinations = generate_combinations(inputs_cls)

    # Step 2: Filter out combinations that satisfy any condition
    invalid_combinations = [
        combo for combo in combinations
        if not evaluate_conditions(combo, conditions)
    ]
    
    return invalid_combinations

class BaseFsm(ABC):

    """
    Base class for Finite State Machines (FSM)

    Some conventions:
    - Every input that needs to be provided in the `step` method (to move the state machine one step forward), needs to
    be gettable from the get_inputs method, either as a numpy array or a dictionary. 
    
    - For consistency during comparison to check whether values have changed or not, every input needs to be convertable 
    to a **float**.
    
    - Any conditional transition that implements some counter (usually named *_elapsed_samples), should check
    for `last_checked_sample` to be different to `current_sample` before increasing its counter. This is to avoid
    increasing the counter more than once per step since these methods could be called multiple times in a single step.
    
    - If for a transition, there are two or more conditions with counters, they need to use different last_checked_sample, otherwise
    only the first condition increase its number of elapsed samples.
    
    # About validation or setting default input values to remain at a given state
    # ---------------------------------------------------------------------------
    # In order to ensure that only one set of inputs is valid to stay in the same state,
    # we can create dummy transitions that start and go to the same state when "invalids"
    # set or input are given, or that when multiple combinations are obtained, have a
    # policy to choose one, such as the taking the lowest values
    # Another option is to just set is to hardcode the set of inputs for each state and
    # machine in the child class
    # For now, go with the last one
    """
    # Here we could include all attributes even though set in __init__ just for 
    # clarity and type hinting

    states_inputs_set: dict[str|int, FsmInputs] = None # Dictionary with the expected inputs to remain on each state, needs to be defined in init of every child class
    default_inputs: FsmInputs = None # FsmInputs instance of default input values, used to fill undefined input values when exploring transitions
    warn_different_inputs_but_no_state_change: bool = False
    inputs: FsmInputs = None
    _last_checked_sample: dict[str, int] = None  #  Automatically initialized in the constructor given the _cooldown_callbacks and _counter_callbacks lists
    _state_type: Enum = None  # To be defined in the child class, type of the FSM state, should be an Enum like class
    _inputs_cls: Type[FsmInputs] = None  # To be defined in the child class, class of the inputs
    _cooldown_callbacks: list[str] = None  # To be defined in the child class, list of methods to call at the begining of a new step (prepare_event) and registered in the last_checked_sample dict
    _counter_callbacks: list[str] = None  # To be defined in the child class, list of methods to register in the last_checked_sample dict
    
    def __init__(self, sample_time: int, name: str, initial_state: str | Enum, 
                 params: FsmParameters, internal_state: FsmInternalState, 
                 inputs: FsmInputs = None,
                 current_sample: int = 0,
                 ) -> None:
        
        from solarmed_modeling.fsms.utils import convert_to # Avoid circular import
        self.convert_to = convert_to
        
        self.name = name
        self.sample_time = sample_time
        self.current_sample: int = current_sample # Counter to keep track of the current sample
        
        # In condition functions, counters will be increased only if `last_checked_sample` is different to `current_sample`
        # By default we only have one checker, it's up to the child class to add more if it includes
        # multiple conditions with counters for the same transition
        checker_ids: list[str] = []
        if self._cooldown_callbacks is not None:
            checker_ids += self._cooldown_callbacks
        if self._counter_callbacks is not None:
            checker_ids += self._counter_callbacks
        
        self._last_checked_sample: dict[str, int] = {checker_id: self.current_sample for checker_id in checker_ids}
        
        self.params: FsmParameters = copy.deepcopy(params) # Suputamadrepython
        self.internal_state: FsmInternalState = copy.deepcopy(internal_state) # Suputamadrepython
        self.initial_internal_state = copy.deepcopy(internal_state) # mutable?
        initial_state = convert_to(initial_state, self._state_type, return_format='enum')
        self.initial_state = initial_state
        self.inputs = inputs
        
        # State machine initialization
        self.machine: Machine = Machine(
            model=self, initial=initial_state, auto_transitions=False, show_conditions=True,
            show_state_attributes=True, ignore_invalid_triggers=False, queued=True, 
            send_event=True, # finalize_event=''
            before_state_change='inform_exit_state', after_state_change='inform_enter_state',
            prepare_event=self._cooldown_callbacks
        )
        
    def validate_or_set_inputs(self, ) -> None:
        
        # Check if the inputs are compatible with the current state or set them to the expected ones
        expected_inputs = self.get_inputs_for_current_state()
        if self.inputs is not None:
            # Check inputs are compatible with current state
            assert self.inputs == expected_inputs, f"Inputs {self.inputs} are not compatible with current state {self.state.name}. Expected inputs: {expected_inputs}"
        else:
            # Set inputs to the expected ones
            self.inputs = expected_inputs
        # self.inputs: FsmInputs = self.inputs

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        inputs_array: np.ndarray[float] = self.update_inputs_array()
        self.inputs_array_prior: np.ndarray[float] = inputs_array

    @property
    def state(self) -> Enum:
        return self._state
    
    @state.setter
    def state(self, value: int | Enum | str) -> None:
        # if not isinstance(value, self._state_type):
        #     raise ValueError("State must a {self._state_type} instance")
        self._state = self.convert_to(value, self._state_type, return_format='enum')

    def make_hissing_noises(self, event) -> None:
        logger.debug(f"[{self.name}] Hiss hiss")

    def get_state(self) -> Enum:
        return self.state

    def get_inputs(self, format: Literal['array', 'dict'] = 'array') -> np.ndarray[float] | dict:
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
        
        if format == 'array':
            # When the array format is used, all variables necessarily need to be parsed as floats
            # return np.array(list(asdict(self.inputs).values()), dtype=float)
            return self.inputs.dump_as_array()

        elif format == 'dict':
            # In the dict format, each variable  can have its own type
            return self.inputs.dump_as_dict()

    def update_inputs_array(self) -> np.ndarray[float]:
        """ Update the inputs array """
        self.inputs_array = self.get_inputs(format='array')

        return self.inputs_array


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
        ...

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
                    raise ValueError(f"WDYM More than one transition is valid: {transition_trigger_id}, {candidate}")

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
    
    def check_elapsed_samples(self, elapsed_samples: int, samples_duration: int, msg: str = None) -> tuple[bool, int]:
        # Elapsed samples is increased only if the current sample is different 
        # to the last checked sample, and only once per step
        
        caller_fn_id: str = inspect.currentframe().f_back.f_code.co_name
        assert caller_fn_id in self._last_checked_sample.keys(), f"Function {caller_fn_id} not registered in last_checked_sample. They need to be defined in one off `_cooldown_callbacks` or `_counter_callbacks` class attributes. Check the non-existing documenation for more info."
        
        first_check: bool = self._last_checked_sample[caller_fn_id] != self.current_sample
        if first_check:
            elapsed_samples += 1
            self._last_checked_sample[caller_fn_id] = self.current_sample
        
        if elapsed_samples >= samples_duration:
            output = True
            log_msg = f"[{self.name}] {msg if msg else 'transition'} done"
        else:                
            output = False
            log_msg = f"[{self.name}] Still {msg if msg else 'performing transition'}, {elapsed_samples}/{samples_duration} to complete"
        
        if first_check:
            logger.info(log_msg)
                
        return output, elapsed_samples
    
    def step(self, inputs: FsmInputs, return_df: bool = False, df: pd.DataFrame = None) -> None | pd.DataFrame:
        """Move the state machine one step forward
        Common FSM logic here that could be extended in child classes. 
        
        Method steps:
            1. Update input attributes and call self.update_inputs_array()
            2. Get next valid transition
            3. Trigger the transition
            4. Update prior inputs attribute
            5. Update and return the dataframe if needed
            
        """
        self.current_sample += 1

        # Does not work because it will use dataclass defined here, not the one defined in the child module
        # self.inputs: FsmInputs = inputs if isinstance(inputs, FsmInputs) else FsmInputs(**inputs)
        self.inputs: FsmInputs = inputs

        # Store inputs in an array, needs to be updated every time the inputs change (step)
        self.update_inputs_array()

        transition = self.get_next_valid_transition(prior_inputs=self.inputs_array_prior,
                                                    current_inputs=self.inputs_array)

        if transition is not None:
            transition()

        # Save prior inputs
        self.inputs_array_prior = self.inputs_array
        
        # Return updated dataframe
        if return_df:
            return self.to_dataframe(df)
        
    
    def reset_fsm(self, *args, **kwargs) -> None:
        """ Resets FSM to initial state and initial internal state """
        self.current_sample = 0
        self.state = self.initial_state
        self.internal_state = self.initial_internal_state
        # super().reset_fsm(*args, **kwargs) # Resets current_sample counter
        
        logger.info(f"[{self.name}] Resetting FSM")
    
    def to_dataframe(self, df: pd.DataFrame = None, states_format: Literal["enum", "value", "name"] = "enum") -> pd.DataFrame:

        if df is None:
            df = pd.DataFrame()
            
        assert states_format in ["enum", "value", "name"], "states_format should be either 'enum', 'value' or 'name'"
        
        # There should be some internal method to set the fields. 
        # - Inputs similar to internal states and params should be an `inputs` dataclass
        # - And then just have an attribute with a list of the fields to be included in the dataframe        
        
        data = pd.DataFrame({
            'state': self.convert_to(self.state, state_cls = self._state_type, return_format = states_format),
            'current_sample': self.current_sample,
            # 'qsf': self.qsf,
            # 'qts_src': self.qts_src,
            **self.inputs.dump_as_dict(format=states_format),
            **self.get_internal_state_dict()
        }, index=[0])

        df = pd.concat([df, data], ignore_index=True)

        return df

    def get_internal_state_dict(self) -> dict:
        # What to include would be better defined in a parameter,
        # which is updated starting from a default parent class value
        # Should we export almost all the variables that are not internal like the machine?
        internal_state_dict = asdict(self.internal_state)
        for key, value in internal_state_dict.items():
            if not isinstance(value, (int, float, bool, str)):
                internal_state_dict.pop(key)
                
        return internal_state_dict
        

    # Auxiliary methods (to calculate associated costs depending on the state)
    def generate_graph(self, fmt :Literal['png', 'svg'] = 'svg', output_path: Path = None) -> str:
        if output_path is None:
            return self.machine.get_graph().draw(None, format=fmt, prog='dot')
        else:
            output_path = Path(output_path)
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'bw') as f:
                return self.machine.get_graph().draw(f, format=fmt, prog='dot')
            
    def get_inputs_for_current_state(self, return_stay_in_state_combinations: bool = False) -> FsmInputs:
        
        # stay_in_st_inputs: list[list[dict]] = []
        # for trigger in self.machine.get_triggers(self.state):
        #     condition_objs = self.machine.get_transitions(trigger=trigger)[0].conditions
        #     # inv_inps: dict[str, int | bool] = {} # 
        #     inv_inps: list[dict] = []
        #     for condition_obj in condition_objs:
        #         if "cooldown" in condition_obj.func:
        #             # Skip cooldown conditions
        #             continue
                
        #         condition = getattr(self, condition_obj.func)

        #         if not condition_obj.target:
        #             inv_inp = condition(return_invalid_inputs=True)
        #         else:
        #             inv_inp = condition(return_valid_inputs=True)
        #         # inv_inps.update(inv_inp)
        #         inv_inps.append(inv_inp)
                
        #     print(f"Transition {trigger}, invalid inputs: {inv_inps}")   
        #     stay_in_st_inputs.append(inv_inps)
        
        # stay_in_st_combos = filter_invalid_combinations(inputs_cls=self._inputs_cls, conditions=stay_in_st_inputs)
        # # assert len(stay_in_st_combos) == 1, f"More than one valid input combination for state {self.state.name}. Wrong definition of FSM transitions: {stay_in_st_combos}"
        # # if len(stay_in_st_combos) > 1:
        # #     logger.warning(f"More than one valid input combination for state {self.state.name}. Wrong definition of FSM transitions: {stay_in_st_combos}")
        
        # # return stay_in_st_inputs
        # if return_stay_in_state_combinations:
        #     return stay_in_st_combos
        # return self._inputs_cls(**stay_in_st_combos[0])
        
        return self.states_inputs_set[self.state.name]
    
    @abstractmethod
    def reset_cooldowns(self, ) -> None:
        ...