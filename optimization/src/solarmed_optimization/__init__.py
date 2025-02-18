import math
from typing import get_args, NamedTuple, Literal
from dataclasses import dataclass, fields, field, is_dataclass, asdict
from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd

from solarmed_modeling.solar_med import (SolarMED, 
                                         EnvironmentParameters,
                                         ModelParameters,
                                         FixedModelParameters,
                                         FsmParameters,
                                         FsmInternalState,
                                         InitialStates)
from solarmed_modeling.fsms import MedState, SfTsState, MedVacuumState
from solarmed_modeling.fsms.med import (FsmParameters as MedFsmParams,
                                        FsmInputs as MedFsmInputs)
from solarmed_modeling.fsms.sfts import (FsmParameters as SftsFsmParams,
                                         FsmInputs as SfTsFsmInputs)


class MedMode(Enum):
    """ Possible decisions for MED operation modes.
    Given this, the FSM inputs are deterministic """
    OFF = 0
    # IDLE = 1
    ACTIVE = 1
    
med_fsm_inputs_table: dict[tuple[MedMode, MedState], MedFsmInputs] = {
    # med_mode = OFF
    (MedMode.OFF, MedState.OFF):               MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
    (MedMode.OFF, MedState.GENERATING_VACUUM): MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
    (MedMode.OFF, MedState.IDLE):              MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
    (MedMode.OFF, MedState.STARTING_UP):       MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
    (MedMode.OFF, MedState.SHUTTING_DOWN):     MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
    (MedMode.OFF, MedState.ACTIVE):            MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
    
    # med_mode = IDLE
    # (MedMode.IDLE, MedState.OFF):               MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
    # (MedMode.IDLE, MedState.GENERATING_VACUUM): MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
    # (MedMode.IDLE, MedState.IDLE):              MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
    # (MedMode.IDLE, MedState.STARTING_UP):       MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
    # (MedMode.IDLE, MedState.SHUTTING_DOWN):     MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
    # (MedMode.IDLE, MedState.ACTIVE):            MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
    
    # med_mode = ACTIVE
    (MedMode.ACTIVE, MedState.OFF):               MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
    (MedMode.ACTIVE, MedState.GENERATING_VACUUM): MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
    (MedMode.ACTIVE, MedState.IDLE):              MedFsmInputs(med_active=True,  med_vacuum_state=MedVacuumState.LOW),
    (MedMode.ACTIVE, MedState.STARTING_UP):       MedFsmInputs(med_active=True,  med_vacuum_state=MedVacuumState.LOW),
    (MedMode.ACTIVE, MedState.SHUTTING_DOWN):     MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
    (MedMode.ACTIVE, MedState.ACTIVE):            MedFsmInputs(med_active=True,  med_vacuum_state=MedVacuumState.LOW),
}

class SfTsMode(Enum):
    """ Possible decisions for Solar field and Thermal storage modes.
    Given this, the associated FSM inputs are deterministic """
    OFF = 0
    # SF_HEATING_UP = 1
    ACTIVE = 1
    
sfts_fsm_inputs_table: dict[tuple[SfTsMode, SfTsState], SfTsFsmInputs] = {
    # sfts_mode = OFF
    (SfTsMode.OFF, SfTsState.IDLE): SfTsFsmInputs(sf_active=False, ts_active=False),
    (SfTsMode.OFF, SfTsState.HEATING_UP_SF): SfTsFsmInputs(sf_active=False, ts_active=False),
    (SfTsMode.OFF, SfTsState.SF_HEATING_TS): SfTsFsmInputs(sf_active=False, ts_active=False),
    
    # sfts_mode = ACTIVE
    (SfTsMode.ACTIVE, SfTsState.IDLE): SfTsFsmInputs(sf_active=True, ts_active=True),
    (SfTsMode.ACTIVE, SfTsState.HEATING_UP_SF): SfTsFsmInputs(sf_active=True, ts_active=True),
    (SfTsMode.ACTIVE, SfTsState.SF_HEATING_TS): SfTsFsmInputs(sf_active=True, ts_active=True),
}

class SubsystemId(Enum):
    SFTS = "sfts"
    MED = "med"
    
class SubsystemDecVarId(Enum):
    SFTS = SfTsMode
    MED = MedMode


def dump_in_span(vars_dict: dict, span: tuple[int, int] | tuple[datetime, datetime], return_format: Literal["values", "series"] = "values") -> dict:
        """
        Dump variables within a given span.
        
        Args:
            vars_dict: A dictionary containing the variables to dump.
            span: A tuple representing the range (indices or datetimes).
            return_format: Format of the returned values ("values" or "series").
        
        Returns:
            A new dictionary containing the filtered data.
        """
        if isinstance(span[0], datetime):
            # Ensure all attributes are pd.Series for datetime filtering
            for name, value in vars_dict.items():
                if not isinstance(value, pd.Series) and value is not None:
                    raise TypeError(f"All attributes must be pd.Series for datetime indexing: {name} is {type(value)}")
            
            # Extract the range
            dt_start, dt_end = span
            span_vars_dict = {
                name: value[(value.index >= dt_start) & (value.index < dt_end)]
                for name, value in vars_dict.items() if value is not None
            }
        else:
            # Assume numeric indices
            idx_start, idx_end = span
            span_vars_dict = {
                name: value[idx_start:idx_end]
                if isinstance(value, (pd.Series, np.ndarray)) else value
                for name, value in vars_dict.items()
            }

        if return_format == "values":
            span_vars_dict = {name: value.values if isinstance(value, pd.Series) else value for name, value in span_vars_dict.items()}

        return span_vars_dict
    
@dataclass
class EnvironmentVariables:
    """
    Simple class to make sure that the required environment variables are passed
    
    All the variables should be 1D arrays with as many elements as the horizon of the optimization problem
    """
    # n_horizon: int
    
    Tmed_c_in: float | np.ndarray[float] | pd.Series  # Seawater temperature
    Tamb: float | np.ndarray[float] | pd.Series # Ambient temperature
    I: float | np.ndarray[float] | pd.Series# Solar radiation
    
    cost_w: float | np.ndarray[float] | pd.Series # Cost of water, €/m³ 
    cost_e: float | np.ndarray[float] | pd.Series # Cost of electricity, €/kWhe
    
    wmed_f: float | np.ndarray[float] | pd.Series = None # Seawater salinity
    
    # def __post_init__(self):
    #     # Validate that all the environment variables have the same length
    #     assert all(
    #         len(getattr(self, var_id)) == len(self.Tmed_c_in) 
    #         for var_id in ["Tamb", "I", "cost_w", "cost_e"]
    #     ), "All variables must have the same length (optim_window_size // sample_time_mod)"

    def dump_at_index(self, idx: int, return_dict: bool = False) -> "EnvironmentVariables":
        """
        Dump instance at a given index.

        Parameters:
        - idx: Integer index to extract.

        Returns:
        - A dictionary.
        """
        dump =  {name: np.asarray(value)[idx] for name, value in asdict(self).items() if value is not None}
        
        return dump if return_dict else EnvironmentVariables(**dump)
    
    def dump_in_span(self, span: tuple[int, int] | tuple[datetime, datetime], return_format: Literal["values", "series"] = "values") -> 'EnvironmentVariables':
        """ Dump environment variables within a given span """
        
        vars_dict = dump_in_span(vars_dict=asdict(self), span=span, return_format=return_format)
        return EnvironmentVariables(**vars_dict)
    
    def resample(self, *args, **kwargs) -> "EnvironmentVariables":
        """ Return a new resampled environment variables instance """
        
        output = {}
        for name, value in asdict(self).items():
            if value is None:
                continue
            elif not isinstance(value, pd.Series):
                raise TypeError(f"All attributes must be pd.Series for datetime indexing. Got {type(value)} instead.")
            
            target_freq = int(float(args[0][:-1]))
            current_freq = value.index.freq.n
            
            value = value.resample(*args, **kwargs)
            if  target_freq > current_freq: # Downsample
                value = value.mean()
            else: # Upsample
                value = value.interpolate()
            output[name] = value
            
        return EnvironmentVariables(**output)
    
@dataclass
class DecisionVariables:
    """
    Simple class to make sure that the required decision variables are passed
    to the model instance with the correct type
    """
    # Real
    qsf: float | np.ndarray[float] | pd.Series #  Solar field flow -> Actual optimization output will be the outlet temperature (`Tsf,out`) after evaluating the inverse solar field model.
    qts_src: float | np.ndarray[float] | pd.Series #  Thermal storage recharge flow.
    qmed_s: float | np.ndarray[float] | pd.Series #  MED heat source flow.
    qmed_f: float | np.ndarray[float] | pd.Series #  MED feed flow.
    Tmed_s_in: float | np.ndarray[float] | pd.Series #  MED heat source inlet temperature.
    Tmed_c_out: float | np.ndarray[float] | pd.Series #  MED condenser outlet temperature.
    # Logical / integers
    sfts_mode: int | np.ndarray[int] | pd.Series #  Solar field and thermal storage mode (off, active)
    # sf_active: bool | np.ndarray[bool]#  Solar field state (off, active)
    # ts_active: bool | np.ndarray[bool]#  Thermal storage state (off, active)
    # med_active: bool | np.ndarray[bool]#  MED heat source state (off, active)
    # med_vac_state: int | np.ndarray[int] | pd.Series #  MED vacuum system state (off, low, high)
    med_mode: int | np.ndarray[int] | pd.Series #  MED operation mode (off, active)
    
    # def __post_init__(self) -> None:
    #     # Ensure attributes are of correct type
    #     for fld in fields(self):
    #         value: np.ndarray | bool | float | int | pd.Series = getattr(self, fld.name)
    #         _type = get_args(fld.type)[0]
    #         if isinstance(value, np.ndarray):
    #             setattr(self, fld.name, value.astype(_type))
    #         elif isinstance(value, pd.Series):
    #             setattr(self, fld.name, value.values.astype(_type))
    #         else:
    #             setattr(self, fld.name, _type(value))
    
    def dump_at_index(self, idx: int, return_dict: bool = False) -> "DecisionVariables":
        """
        Dump instance at a given index.

        Parameters:
        - idx: Integer index to extract.

        Returns:
        - A dictionary.
        """
        dump =  {name: np.asarray(value)[idx] for name, value in asdict(self).items() if value is not None}
        
        return dump if return_dict else DecisionVariables(**dump)
    
    def dump_in_span(self, span: tuple[int, int] | tuple[datetime, datetime], return_format: Literal["values", "series"] = "values") -> 'DecisionVariables':
        """ Dump decision variables within a given span """
        
        vars_dict = dump_in_span(vars_dict=asdict(self), span=span, return_format=return_format)
        return DecisionVariables(**vars_dict)
        
                
def dump_at_index_dec_vars(dec_vars: DecisionVariables, idx: int, return_dict: bool = False) -> DecisionVariables | dict:
    """ Dump decision variables at a given index """
    
    import warnings
    
    warnings.warn("This function is deprecated. Use DecisionVariables.dump_at_index instead.", DeprecationWarning)
    
    # Precioso, equivale a: {field.name: field.type( field.value[idx] )}
    dump = {field.name: get_args(field.type)[0]( getattr(dec_vars, field.name)[idx] ) for field in fields(dec_vars)}
    
    if return_dict:
        return dump
    return DecisionVariables(**dump)
    
@dataclass
class IntegerDecisionVariables:
    sfts_mode: SfTsMode | pd.Series
    med_mode: MedMode | pd.Series
    
@dataclass
class DecisionVariablesUpdates:
    """ Number of decision variable updates in the optimization window """
    
    # sf_active: int # 
    # ts_active: int # 
    sfts_mode: int #
    med_mode: int #
    # med_active: int # 
    # med_vac_state: int # 
    qsf: int # 
    qts_src: int # 
    qmed_s: int # 
    qmed_f: int # 
    Tmed_s_in: int # 
    Tmed_c_out: int # 
    
    # def __post_init__(self):
        # Validate that SfTs FSM related decision varaiables have the same number of updates
        # assert self.sf_active == self.ts_active, "Solar field and thermal storage logical variables should have the same number of updates"
        
        # Validate that MED FSM related decision variables have the same number of updates
        # assert self.med_active == self.med_vac_state, "MED logical variables should have the same number of updates"
        
        # TODO: Would be good to validate that the number of updates is within:
        # 1 <= n_uptes <= optim_window_size / sample_time_mod (=n_evals_mod)

# Check errors during development
# assert [field.name for field in fields(DecisionVariables)] == [field.name for field in fields(DecisionVariablesUpdates)], \
#     "Attributes of DecisionVariables should exactly match attributes in DecisionVariableUpdates"

class OptimToFsmsVarIdsMapping(NamedTuple):
    # sf_active: tuple = ("sf_active", )
    # ts_active: tuple = ("ts_active", )
    sfts_mode: tuple = ("sf_active", "ts_active")
    med_mode: tuple  = ("med_active", "med_vacuum_state")
    
# class FsmstoOptimVarIdsMapping:
#     sf_active = "sf_active" 
#     ts_active = "ts_active"
#     med_mode = ""
    
class OptimVarIdstoModelVarIdsMapping(Enum):
    """
    Mapping between optimization decision variable ids and model variable ids.
    Using an Enum allows for bi-directional lookups compared to a dictionary
    
    Maybe we could include here only variables that differ, and by default assume
    that the variable ids are the same in the optimization and the model
    
    Structure:
    optim_var_id = model_var_id
    
    Examples:
    # Convert from optim id to model id
    print(f"optim_id: qsf -> model id: {OptimVarIdstoModelVarIdsMapping.qsf.value}")

    # Convert from model id to optim id
    print(f"model id: qts_src -> optim_id: {OptimVarIdstoModelVarIdsMapping('qts_src').name}")
    """
    sf_active = "sf_active"
    ts_active = "ts_active"
    med_active = "med_active"
    med_vac_state = "med_vacuum_state"
    qsf = "qsf"
    qts_src = "qts_src"
    qmed_s = "qmed_s"
    qmed_f = "qmed_f"
    Tmed_s_in = "Tmed_s_in"
    Tmed_c_out = "Tmed_c_out"

class RealLogicalDecVarDependence(Enum):
    """ Utility class that defines dependence relationship between real 
    decision variables and operation modes / integer ones """
    
    qsf = "sfts_mode"
    qts_src = "sfts_mode"
    qmed_s = "med_mode"
    qmed_f = "med_mode"
    Tmed_s_in = "med_mode"
    Tmed_c_out = "med_mode"
    
@dataclass
class RealDecVarsBoxBounds:
    """ Real decision variables box bounds, as in: (lower bound, upper bound)"""
    qsf: tuple[float, float]
    qts_src: tuple[float, float]
    qmed_s: tuple[float, float]
    qmed_f: tuple[float, float]
    Tmed_s_in: tuple[float, float]
    Tmed_c_out: tuple[float, float]
    
    @classmethod
    def initialize(cls, fmp: FixedModelParameters, Tmed_c_in: float) -> 'RealDecVarsBoxBounds':
    
        return cls(
            qts_src = (fmp.ts.qts_src_min, fmp.ts.qts_src_max),
            qsf = (fmp.sf.qsf_min, fmp.sf.qsf_max),
            Tmed_s_in = (fmp.med.Tmed_s_min, fmp.med.Tmed_s_max),
            Tmed_c_out = (Tmed_c_in+2, Tmed_c_in+10),
            qmed_s = (fmp.med.qmed_s_min, fmp.med.qmed_s_max),
            qmed_f = (fmp.med.qmed_f_min, fmp.med.qmed_f_max),
        )
        
@dataclass
class InitialDecVarsValues:
	sfts_mode: int = 0
	med_mode: int = 0
	qsf: float = 0.0
	qts_src: float = 0.0
	qmed_s: float = 0.0
	qmed_f: float = 0.0
	Tmed_s_in: float = 0.0
	Tmed_c_out: float = 0.0

@dataclass
class RealDecisionVariablesUpdatePeriod:
    qsf: int = 1800
    qts_src: int = 1800
    qmed_s: int = 3600
    qmed_f: int = 3600
    Tmed_s_in: int = 3600
    Tmed_c_out: int = 3600
	
@dataclass
class RealDecisionVariablesUpdateTimes:
    qsf: list[datetime]
    qts_src: list[datetime]
    qmed_s: list[datetime]
    qmed_f: list[datetime]
    Tmed_s_in: list[datetime]
    Tmed_c_out: list[datetime]
@dataclass
class FsmData:
    metadata: dict
    paths_df: pd.DataFrame
    valid_inputs: list[list[list[float]]]
    
@dataclass
class ProblemSamples:
    # Times to samples transformation
    n_evals_mod_in_hor_window: int # Number of model evalations along the optimization window
    n_evals_mod_in_opt_step: int # Number of model evaluations in one optimization step
    episode_samples: int # Number of samples in one episode
    optim_window_samples: int # Number of samples in the optimization window
    max_opt_steps: int # Max number of steps for the optimization scheme
    span: int # Number of previous samples to keep track of
    default_n_dec_var_updates: int # Default number of decision variable updates in the optimization window (depends on sample_time_opt)
    max_dec_var_updates: int # Maximum number of decision variable updates in the optimization window (depends on sample_time_mod)
    min_dec_var_updates: int  = 1 # Minimum number of decision variable updates in the optimization window
    

@dataclass
class ProblemParameters:
    sample_time_mod: int = 400 # Model sample time, seconds
    sample_time_opt: int = 3600 * 0.8 # Optimization evaluations period, seconds
    sample_time_ts: int = 3600 # Thermal storage sample time, seconds (used in nNLP alternative)
    optim_window_time: int = 3600 * 8 # Optimization window size, seconds
    episode_duration: int = None # By default use len(df)
    idx_start: int = None # By default estimate from sf fixed_mod_params.delay_span
    env_params: EnvironmentParameters = field(default_factory=lambda: EnvironmentParameters())
    fixed_model_params: FixedModelParameters = field(default_factory=lambda: FixedModelParameters())
    model_params: ModelParameters = field(default_factory=lambda: ModelParameters())
    fsm_params: FsmParameters = field(default_factory=lambda: FsmParameters(
        med=MedFsmParams(
            vacuum_duration_time = 1*3600, # 1 hour
            brine_emptying_time = 30*60,   # 30 minutes
            startup_duration_time = 20*60, # 20 minutes
            off_cooldown_time = 12*3600,   # 12 hours
            active_cooldown_time = 4*3600, # 3 hours
        ),
        sf_ts=SftsFsmParams(
            recirculating_ts_enabled = False,
            idle_cooldown_time = 1*3600,   # 1 hour
        )
    ))
    fsm_internal_states: FsmInternalState = field(default_factory=lambda: FsmInternalState())
    fsm_valid_sequences: dict[ str, list[list[int]] ] = field(default_factory=lambda: {
        'MED': [
            [MedState.IDLE.value, MedState.STARTING_UP.value, MedState.ACTIVE.value],
            [MedState.GENERATING_VACUUM.value, MedState.STARTING_UP.value, MedState.ACTIVE.value],
        ],
        'SFTS': [
            [SfTsState.HEATING_UP_SF.value, SfTsState.SF_HEATING_TS.value],
        ]
    })
    dec_var_updates: DecisionVariablesUpdates = None # Set automatically in utils.initialization.problem_initialization if not manually defined
    optim_window_days: int = None # Automatically computed from optim_window_time
    initial_states: InitialStates = None # Optional, if specified model will be initialized with these states
    operation_actions: dict[str, list[tuple[str, int]]] = None # Optional for MINLP, required in nNLP alternative. Defines the operation actions/updates for each subsystem
    real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod = field(default_factory=lambda: RealDecisionVariablesUpdatePeriod()) # nNLP
    initial_dec_vars_values: InitialDecVarsValues = field(default_factory=lambda: InitialDecVarsValues())  # nNLP
    
    def __post_init__(self):
        """ Make convenient to initialize this dataclass from dumped instances """
        for fld in fields(self):
            if is_dataclass(fld.type):
                value = getattr(self, fld.name)
                if isinstance(value, dict):
                # if not isinstance(value, fld.type) and value is not None:
                    setattr(self, fld.name, fld.type(**value))
                    
        self.optim_window_days = math.ceil(self.optim_window_time / (24*3600))
        
    @classmethod
    def initialize(cls, problem_type: Literal["MINLP", "nNLP"], **kwargs) -> "ProblemParameters":
        
        if problem_type == "nNLP":
            assert kwargs.get("operation_actions", None) is not None, "operation_actions must be specified for nNLP problem type"
        
        return cls(**kwargs)    

@dataclass
class ProblemData:
    df: pd.DataFrame
    problem_params: ProblemParameters
    problem_samples: ProblemSamples 
    model: SolarMED

@dataclass
class AlgorithmParameters:
    pop_size: int = 32
    n_gen: int = 80
    seed_num: int = 23
    
@dataclass
class PopulationResults:
    pop_per_gen: list[list[float|int]] # (gen, individual, dec.variable)
    fitness_per_gen: list[list[float]] # (gen, individual)
    time_per_gen: list[float] # (gen, )
    time_total: float
    best_idx_per_gen: list[int] # (gen, )
    worst_idx_per_gen: list[int] # (gen, )
    
    # def __post_init__(self, ):
    #     # Check type of attributes, if they are numpy arrays, make them into lists

    @classmethod
    def initialize(cls, problem, pop_size: int, n_gen: int, elapsed_time: int) -> 'PopulationResults':
    
        x_evaluated = problem.x_evaluated
        fitness_record = problem.fitness_history
        x_history: list[list[list[int | float]]] = []
        fitness_history: list[list[float]] = []
        best_idx: list[int] = []
        worst_idx: list[int] = []
        for idx in range(0, len(x_evaluated)-1, pop_size):
            x_history.append( x_evaluated[idx:idx+pop_size] )
            fitness_history.append( fitness_record[idx:idx+pop_size] )
            best_idx.append( int(np.argmin(fitness_history[-1])) )
            worst_idx.append( int(np.argmax(fitness_history[-1])) )
                
        return cls(
            pop_per_gen=x_history,
            fitness_per_gen=fitness_history,
            best_idx_per_gen=best_idx,
            worst_idx_per_gen=worst_idx,
            time_per_gen=elapsed_time/n_gen,
            time_total=elapsed_time
        )
        