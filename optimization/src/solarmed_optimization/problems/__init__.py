from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from typing import Type
from pathlib import Path

from solarmed_modeling.solar_med import SolarMED
from solarmed_modeling.fsms import SolarMedState

from solarmed_optimization import (RealDecVarsBoxBounds, 
                                   DecisionVariablesUpdates,
                                   EnvironmentVariables,
                                   InitialDecVarsValues,
                                   DecisionVariables,
                                   IntegerDecisionVariables,
                                   RealDecisionVariablesUpdatePeriod,
                                   RealDecisionVariablesUpdateTimes,
                                   FsmData,)

class BaseNlpProblem(ABC):
    """Interface for NLP problems."""
    sample_time_mod: int # Model sample time
    sample_time_ts: int  # Thermal storage sample time
    env_vars: EnvironmentVariables # Environment variables
    initial_values: InitialDecVarsValues
    int_dec_vars: IntegerDecisionVariables
    int_dec_vars_pre_resample: IntegerDecisionVariables
    real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod
    episode_range: tuple[datetime, datetime] # Episode range
    dec_var_ids: list[str] # All decision variables ids
    dec_var_dtypes: list[Type]  # Decision variable data types
    dec_var_int_ids: list[str] # Logical / integer decision variables ids
    dec_var_real_ids: list[str]  # Real decision variables ids
    model_dict: dict # SolarMED model dumped instance
    initial_state: SolarMedState # System initial state
    real_dec_vars_times: RealDecisionVariablesUpdateTimes
    size_dec_vector: int # Size of the decision vector
    dec_var_updates: DecisionVariablesUpdates # Decision variables updates
    real_dec_vars_box_bounds: RealDecVarsBoxBounds
    box_bounds_lower: list[np.ndarray[float]] # Lower bounds for the decision variables (in list of arrays format).
    box_bounds_upper: list[np.ndarray[float]] # Upper bounds for the decision variables (in list of arrays format).
    x_evaluated: list[list[float]] # Decision variables vector evaluated (i.e. sent to the fitness function)
    fitness_history: list[float] # Fitness record of decision variables sent to the fitness function
    operation_span: tuple[datetime, datetime] # Operation start and end datetimes
    
    
    @abstractmethod
    def __init__(self, int_dec_vars: IntegerDecisionVariables, 
              	 initial_dec_vars_values: InitialDecVarsValues, 
                 env_vars: EnvironmentVariables, 
                 real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod,
                 model: SolarMED,
                 sample_time_ts: int
                ) -> None:
        pass
    
    @abstractmethod
    def __post_init__(self, ) -> None:
        pass
    
    @abstractmethod
    def get_bounds(self, ) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_extra_info(self) -> str:
        pass
    
    @abstractmethod
    def get_nic(self) -> int:
        pass
    
    @abstractmethod
    def get_nix(self) -> int:
        pass
    
    @abstractmethod
    def decision_vector_to_decision_variables(self, x: np.ndarray[float], resample: bool = True) -> DecisionVariables:
        pass
    
    @abstractmethod
    def fitness(self, x: np.ndarray[float],  store_x: bool = True, debug_mode: bool = False) -> list[float]:
        pass
    
    
class BaseMinlpProblem(ABC):
    """Interface for MINLP problems."""
    # model: SolarMED  # SolarMED model instance
    optim_window_size: int  # Optimization window size in seconds
    env_vars: EnvironmentVariables # Environment variables
    dec_var_updates: DecisionVariablesUpdates # Decision variables updates
    fsm_med_data: FsmData # FSM data for the MED system
    fsm_sfts_data: FsmData # FSM data for the SFTS systems
    use_inequality_contraints: bool # Whether to use inequality constraints or not
    
    # Computed attributes, actually setting default values makes no difference
    # since __init__ dataclass method is being overriden
    # x: np.ndarray[float] = None # Decision variables values vector
    size_dec_vector: int # Size of the decision vector
    real_dec_vars_box_bounds: RealDecVarsBoxBounds
    initial_state: SolarMedState # System initial state
    dec_var_ids: list[str] # All decision variables ids
    dec_var_int_ids: list[str] # Logical / integer decision variables ids
    dec_var_real_ids: list[str]  # Real decision variables ids
    dec_var_dtypes: list[Type]  # Decision variable data types
    ni: int # Number of logical / integer decision variables
    nr: int # Number of real decision variables
    model_dict: dict # SolarMED model dumped instance
    n_evals_mod_in_hor_window: int # Number of model evaluations per optimization window
    n_evals_mod_in_opt_step: int # Number of model evaluations per optimization step
    sample_time_mod: int # Model sample time
    sample_time_opt: int # Optimization sample time
    box_bounds_lower: list[np.ndarray[float | int]] # Lower bounds for the decision variables (in list of arrays format). Updated every time `get_bounds` is called
    box_bounds_upper: list[np.ndarray[float | int]] # Upper bounds for the decision variables (in list of arrays format). Updated every time `get_bounds` is called
    integer_dec_vars_mapping: dict[str, np.ndarray[list[int]]] # Mapping from integer decision variables to FSMs inputs
    x_evaluated: list[list[float | int]] # Decision variables vector evaluated (i.e. sent to the fitness function)
    fitness_history: list[float] # Fitness record of decision variables sent to the fitness function
    
    @abstractmethod
    def __init__(self, 
                 model: SolarMED,
                 sample_time_opt: int,
                 optim_window_time: int,
                 dec_var_updates: DecisionVariablesUpdates,
                 env_vars: EnvironmentVariables,
                 fsm_valid_sequences: dict[ str, list[list] ],
                 fsm_data_path: Path = Path("../results"),
                 use_inequality_contraints: bool = True
                ) -> None:
        pass
    
    @abstractmethod
    def __post_init__(self, ) -> None:
        pass
    
    @abstractmethod
    def get_bounds(self, ) -> tuple[np.ndarray[float | int], np.ndarray[float | int]]:
        pass
    
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_extra_info(self) -> str:
        pass
    
    @abstractmethod
    def get_nic(self) -> int:
        pass
    
    @abstractmethod
    def get_nix(self) -> int:
        pass
    
    @abstractmethod
    def fitness(self, x: np.ndarray[float | int], store_x: bool = True) -> list[float]:
        pass