# from concurrent.futures import ThreadPoolExecutor
import math
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict, fields
import numpy as np
from typing import Type, get_args
from loguru import logger

from solarmed_modeling.solar_med import SolarMED
from solarmed_modeling.fsms import SolarMedState
from solarmed_modeling.fsms.med import FsmInputs as MedFsmInputs
from solarmed_modeling.fsms.sfts import FsmInputs as SfTsFsmInputs

from solarmed_optimization.problems import BaseMinlpProblem
from solarmed_optimization import (EnvironmentVariables,
                                   DecisionVariables,
                                   DecisionVariablesUpdates,
                                   OptimVarIdstoModelVarIdsMapping,
                                   FsmData,
                                   RealLogicalDecVarDependence,
                                   RealDecVarsBoxBounds,
                                   OptimToFsmsVarIdsMapping,
                                   med_fsm_inputs_table,
                                   sfts_fsm_inputs_table)
from solarmed_optimization.path_explorer.utils import import_results
from solarmed_optimization.utils import (forward_fill_resample, 
                                         decision_vector_to_decision_variables,
                                         get_valid_modes,
                                         flatten_list)
from solarmed_optimization.utils.evaluation import evaluate_model

np.set_printoptions(precision=1, suppress=True)

@dataclass
class BaseProblem(BaseMinlpProblem):
    """
    x: decision vector.
    - shape: ( n inputs x n horizon )
    - structure:
        X = [x[0,0] x[1,0] x[2,0] ... x[Ninputs,0], ... x[0,Nhorizon] ... x[Ninputs,Nhorizon]]
    
    - Decision vector components: See `DecisionVariables`  
    - Environment variables: See `EnvironmentVariables`
    """

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

        self.optim_window_size = optim_window_time
        self.dec_var_updates = dec_var_updates
        self.env_vars = env_vars
        self.use_inequality_contraints = use_inequality_contraints
        
        self.model_dict = model.dump_instance()
        # Add fields not included in the dump_instance method
        self.model_dict.update(dict(
            # Initial states
            ## FSM states
            fsms_internal_states=model.fsms_internal_states,
            med_state=model.med_state,
            sf_ts_state=model.sf_ts_state,
        ))
        # self.model = SolarMED(**self.model_dict) # To make sure we don't modify the original instance
        
        self.initial_state: SolarMedState = model.get_state()
        self.sample_time_mod = model.sample_time
        self.sample_time_opt = sample_time_opt
        self.n_evals_mod_in_hor_window: int = int(optim_window_time // self.sample_time_mod)
        self.n_evals_mod_in_opt_step: int = int(sample_time_opt // self.sample_time_mod)
        
        self.dec_var_ids, self.dec_var_dtypes = zip(*[(field.name, get_args(field.type)[0]) for field in fields(DecisionVariables)])
        self.dec_var_int_ids: list[str] = [var_id for var_id, var_type in zip(self.dec_var_ids, self.dec_var_dtypes) if var_type in [bool, int]]
        self.dec_var_real_ids: list[str] = [var_id for var_id, var_type in zip(self.dec_var_ids, self.dec_var_dtypes) if var_type is float]
        self.dec_var_model_ids: list[str] = [OptimVarIdstoModelVarIdsMapping[var_id].value if var_id in OptimVarIdstoModelVarIdsMapping.__members__ else var_id for var_id in self.dec_var_ids]
        self.ni = len(self.dec_var_int_ids) # TODO: Probably needs to change to use same name as PyGMO
        self.nr = len(self.dec_var_real_ids) # TODO: Probably needs to change to use same name as PyGMO
        self.n_dec_vars = self.ni + self.nr
        self.size_dec_vector = np.sum([getattr(dec_var_updates, var_id) for var_id in self.dec_var_ids])
        self.n_updates_per_opt_step = [math.floor(getattr(dec_var_updates, var_id) * sample_time_opt / optim_window_time) for var_id in self.dec_var_ids]
        
        # Import FSMs data
        ## MED
        system: str = 'MED'
        n_horizon = dec_var_updates.med_mode
        paths_df, valid_inputs, metadata = import_results(
            paths_path=fsm_data_path, system=system, n_horizon=n_horizon,
            return_metadata=True, return_format="value", generate_if_not_found=True,
            initial_states=[state for state in model._med_fsm._state_type],
            params={
                'valid_sequences': fsm_valid_sequences[system], 
                "sample_time": optim_window_time // n_horizon,
                **asdict(model.fsms_params.med)
            },
        )
        self.fsm_med_data = FsmData(metadata=metadata, paths_df=paths_df, valid_inputs=np.array(valid_inputs))

        ## SfTs
        system: str = 'SFTS'
        n_horizon = dec_var_updates.sfts_mode
        paths_df, valid_inputs, metadata = import_results(
            paths_path=fsm_data_path, system=system, n_horizon=n_horizon,
            return_metadata=True, return_format="value", generate_if_not_found=True,
            initial_states=[state for state in model._sf_ts_fsm._state_type if state.name != "RECIRCULATING_TS"],
            params={
                'valid_sequences': fsm_valid_sequences[system], 
                "sample_time": optim_window_time // n_horizon,
                **asdict(model.fsms_params.sf_ts), 
            },
        )
        self.fsm_sfts_data = FsmData(metadata=metadata, paths_df=paths_df, valid_inputs=np.array(valid_inputs))

        # Generate bounds
        self.real_dec_vars_box_bounds = RealDecVarsBoxBounds.initialize(
            fmp=model.fixed_model_params, 
            Tmed_c_in=env_vars.Tmed_c_in.mean()
        )
        self.box_bounds_lower, self.box_bounds_upper, self.integer_dec_vars_mapping = generate_bounds(self, readable_format=True)
        
        # Initialize decision vector history
        self.x_evaluated = []
        self.fitness_history = []

    def __post_init__(self, ) -> None:
        logger.info(f"""{self.get_name()} initialized.
                    {self.get_extra_info()}""")

    def get_name(self) -> str:
        """ Get problem’s name """
        return "SolarMED MINLP problem"

    def get_extra_info(self) -> str:
        """ Get problem’s extra info. """
        
        # TODO: It would be cool to do some qualitative estimation of some aspects such as:
        # - Irradiance availability along prediciton horizon (high, none, low, medium, high-intermitent, etc)
        # - Storage starting energy level (approx. hours of med operation at three-levels of temperature) 
        # lower_bounds, upper_bounds = get_bounds(
        #     problem_instance=self
        # )
        
        return f"""
    -\t Size of decision vector: {self.size_dec_vector} elements
    -\t Decision variable ids: {self.dec_var_ids}
    -\t Decision variable types: {self.dec_var_dtypes}
    -\t Number of updates per dec.var along optim. horizon: {[getattr(self.dec_var_updates, var_id) for var_id in self.dec_var_ids]}
    -\t Model sample time: {self.sample_time_mod} seconds
    -\t Optimization step time: {self.sample_time_opt/60:.1f} minutes
    -\t Optimization horizon time: {self.optim_window_size/3600:.1f} hours
    -\t Number of model evals in optim. window: {self.n_evals_mod_in_hor_window}
    -\t Number of model evals in optim. step: {self.n_evals_mod_in_opt_step}
    -\t System initial state: {self.initial_state.name}
    -\t Lower bounds: {self.box_bounds_lower}
    -\t Upper bounds: {self.box_bounds_upper}"""
    
    
def set_real_var_bounds(problem_instance: BaseProblem, ub: np.ndarray, lb: np.ndarray, 
                              var_id: str, lower_limit: float | np.ndarray[float], 
                              upper_limit: float | np.ndarray[float], aux_logical_var_id: str  = None,
                              integer_dec_vars_mapping: dict[str, np.ndarray[list[int]]] = None) -> None:
    """
    Set the bounds for a real variable in the decision vector
    ~~No need to pass the bounds arrays or return them, as they are modified in place (mutable)~~
    """
    input_idx_in_dec_vars = problem_instance.dec_var_ids.index(var_id)
    n_updates = getattr(problem_instance.dec_var_updates, var_id)
    if aux_logical_var_id is not None:
        # aux_logical_input_idx_in_dec_vars = self.dec_var_ids.index(aux_logical_var_id)
        integer_mapping = integer_dec_vars_mapping[aux_logical_var_id]
        
        integer_upper_value: np.ndarray[int] = np.array( [1 if np.any(bounds > 0) else 0 for bounds in integer_mapping], dtype=int )
        integer_lower_value: np.ndarray[int] = np.array( [np.min(bounds) for bounds in integer_mapping], dtype=int )
        
        # print(f"{self.model.get_state().name} | {aux_logical_var_id}: {integer_upper_value=}, {integer_lower_value=}")
        
        if len(integer_upper_value) < n_updates:
            integer_upper_value = forward_fill_resample(integer_upper_value, n_updates)
            integer_lower_value = forward_fill_resample(integer_lower_value, n_updates)
        upper_value: np.ndarray[float] = upper_limit * integer_upper_value
        lower_value: np.ndarray[float] = lower_limit * integer_lower_value
        
    else:
        upper_value: np.ndarray[float] = upper_limit * np.ones((n_updates, ))
        lower_value: np.ndarray[float] = lower_limit * np.ones((n_updates, ))
    
    ub[input_idx_in_dec_vars] = upper_value
    lb[input_idx_in_dec_vars] = lower_value
    
    return lb, ub
    
    
def generate_bounds(problem_instance: BaseProblem, readable_format: bool = False, debug: bool = False) -> np.ndarray[float | int] | tuple[np.ndarray[float | int], np.ndarray[float | int]]:
        """This method will return the box-bounds of the problem. 
        - Infinities in the bounds are allowed.
        - The order of elements in the bounds should match the order of the variables in the decision vector (`dec_var_ids`)
            
        Decision vector structure:
            x = [ [1,...,n_updates_x1], [1,...,n_updates_x2], ..., [1,...,n_updates_xNdec_vars] ]

            s.a. n_dec_var_i_mín <= n_dec_var_i <= n_dec_var_i_máx
            
            donde: 
            - n_dec.var_i_mín = 1 o n_horizon 
            - n_dec.var_i_máx = n_horizon * Ts_opt/Ts_mod 

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower bounds, upper bounds)
        """

        
        # To simplify, use a list of arrays with shape (n_updates(i), ) to build bounds, later reshape it to (sum(n_updates(i)))
        # Initialization
        integer_dec_vars_mapping: dict[str, np.ndarray[list[int]]] = {
            var_id: np.full((getattr(problem_instance.dec_var_updates, var_id), ), list[int], dtype=object) for var_id in problem_instance.dec_var_int_ids
        }
        # Lower box-bounds
        lbb: list[np.ndarray[float | int]] = [
            np.full((n_updates, ), np.nan, dtype=float) for n_updates in asdict(problem_instance.dec_var_updates).values()
        ]
        # Upper box-bounds
        ubb: list[np.ndarray[float | int]] = [
            np.full((n_updates, ), np.nan, dtype=float) for n_updates in asdict(problem_instance.dec_var_updates).values()
        ]
        
        # Logical / integer variables bounds
        # For each FSM, get the possible paths from the initial state and the valid inputs, to set its bounds and mappings
        for initial_state, fsm_data, \
            lookup_table, fsm_inputs_cls in zip([problem_instance.model_dict["sf_ts_state"], problem_instance.model_dict["med_state"]],
                                                [problem_instance.fsm_sfts_data, problem_instance.fsm_med_data],
                                                [sfts_fsm_inputs_table, med_fsm_inputs_table],
                                                [SfTsFsmInputs, MedFsmInputs]):
                    
            # initial_state = MedState(0) # problem_instance.model_dict["med_state"]
            # fsm_data = problem_instance.fsm_med_data
            # lookup_table = med_fsm_inputs_table
            # fsm_inputs_cls = MedFsmInputs
                    
            paths_df = fsm_data.paths_df
            valid_inputs = fsm_data.valid_inputs
            fsm_input_ids: list[str] = fsm_data.metadata["input_ids"] 

            # Extract indexes of possible paths from initial state
            paths_from_state_idxs: np.ndarray = paths_df[paths_df["0"] == initial_state.value].index.to_numpy() # dim: (n paths, )
            # Get valid inputs from initial states using those indexes
            valid_inputs_from_state: np.ndarray = valid_inputs[paths_from_state_idxs] # dim: (n paths, n horizon, n inputs)

            # Get the unique discrete values for each input
            for optim_var_id in problem_instance.dec_var_int_ids:
                fsm_var_ids: tuple[str] = getattr(OptimToFsmsVarIdsMapping(), optim_var_id)
                # print(f"{list(fsm_var_ids)}, {fsm_input_ids}")
                if not set(fsm_var_ids).issubset(set(fsm_input_ids)): # Is identical or subset
                    continue
                
                n_updates: int = getattr(problem_instance.dec_var_updates, optim_var_id)
                input_idx_in_dec_vars: int = problem_instance.dec_var_ids.index(optim_var_id)
                
                for step_idx in range(n_updates):
                    # Get unique values for each fsm input at step x
                    # discrete_bounds: (n fsm inputs, n unique) [[fsm_input i unique values at step x], ..., [fsm_input N unique values at step x]]
                    fsm_discrete_bounds = [np.unique(valid_inputs_from_state[:, step_idx, fsm_input_idx]) for fsm_input_idx in range(len(fsm_var_ids))]
                    
                    # print(f"{step_idx=}, {fsm_discrete_bounds=}")
                    
                    # Translate fsm inputs discrete bounds to optimization variable discrete bounds
                    if len(fsm_discrete_bounds) == 1:
                        # Optimization variable maps to a single fsm input
                        optim_var_discrete_bounds = fsm_discrete_bounds[0]
                    else:
                        # Optimization variable wraps multiple fsm inputs
                        combinations = list(itertools.product(*fsm_discrete_bounds))
                        optim_var_values = [get_valid_modes(fsm_inputs_cls(*combo), lookup_table=lookup_table) for combo in combinations]
                        
                        optim_var_discrete_bounds = np.unique(flatten_list(optim_var_values))
                        
                        # print(f"{combinations=}, {optim_var_values=}, {optim_var_discrete_bounds=}")

                    # Update mapping
                    integer_dec_vars_mapping[optim_var_id][step_idx] = optim_var_discrete_bounds
                    # Update bounds
                    ubb[input_idx_in_dec_vars][step_idx] = len(optim_var_discrete_bounds)-1
                    lbb[input_idx_in_dec_vars][step_idx] = 0
                
                if debug:
                    vals_str = [f"{i}: {db} --> [{lb}, {ub}]" for i, (db, lb, ub) in \
                        enumerate(zip(integer_dec_vars_mapping[optim_var_id], 
                                      lbb[input_idx_in_dec_vars], 
                                      ubb[input_idx_in_dec_vars]))]
                    print(f"IB | {problem_instance.model_dict['current_state'].value} | {optim_var_id}: {vals_str}")

        # Real variables bounds
        # Thermal storage
        for var_id in problem_instance.dec_var_real_ids:
            lbb, ubb = set_real_var_bounds(
                problem_instance=problem_instance, ub=ubb, lb=lbb,
                var_id = var_id, 
                lower_limit = getattr( problem_instance.real_dec_vars_box_bounds, var_id)[0], 
                upper_limit = getattr( problem_instance.real_dec_vars_box_bounds, var_id)[1], 
                aux_logical_var_id = RealLogicalDecVarDependence[var_id].value,
                integer_dec_vars_mapping=integer_dec_vars_mapping,
            )    

        # np.set_printoptions(precision=1)
        # print(f"{[f'{var_id}: {bounds}' for var_id, bounds in zip(problem_instance.dec_var_ids, lbb)]}")
        # print(f"{[f'{var_id}: {bounds}' for var_id, bounds in zip(problem_instance.dec_var_ids, ubb)]}")
        # print(f"{integer_dec_vars_mapping=}")
        
        # problem_instance.box_bounds_lower = lbb
        # problem_instance.box_bounds_upper = ubb
        if np.any([np.isnan(np.concatenate(lbb))]) or \
            np.any([np.isnan(np.concatenate(ubb))]):
            raise ValueError("Bounds contain NaN values. Check the bounds generation process.")

        # Finally, concatenate each array to get the final bounds
        if not readable_format:
            lbb = np.concatenate(lbb)
            ubb = np.concatenate(ubb)
            
        # integer_dec_vars_mapping = 
        # print(f"{box_bounds_lower=}")
        # print(f"{box_bounds_upper=}")
        

        return lbb, ubb, integer_dec_vars_mapping
    
def evaluate_fitness(problem_instance: BaseProblem, x: np.ndarray[float | int] | list[np.ndarray[float | int]],) -> list[float] | list[list[float]]:
    # print(f"{x=}")
    def evaluate(x: np.ndarray[float | int]) -> list[float]:
        model: SolarMED = SolarMED(**problem_instance.model_dict)
        
        # Sanitize decision vector, sometimes float values are negative even though they are basically zero (float precision?)
        x[np.abs(x) < 1e-6] = 0
        
        # dec_vars = DecisionVariables(**decision_dict)
        dec_vars: DecisionVariables = decision_vector_to_decision_variables(
            x=x,
            dec_var_updates=problem_instance.dec_var_updates,
            span='optim_window',
            sample_time_mod=problem_instance.sample_time_mod,
            optim_window_time=problem_instance.optim_window_size,
        )
        fitness, ics = evaluate_model(model = model, 
                                      n_evals_mod = problem_instance.n_evals_mod_in_hor_window,
                                      mode = "optimization",
                                      dec_vars = dec_vars, 
                                      env_vars = problem_instance.env_vars,
                                      model_dec_var_ids=problem_instance.dec_var_model_ids,)        
        return fitness, ics
    
    # # Check if we are in batch mode
    # batch_mode: bool = isinstance(x, list) and isinstance(x[0], (list, np.ndarray))
    
    # if not batch_mode:
    fitness, ics = evaluate(x)
    if problem_instance.use_inequality_contraints:
        return [np.sum(fitness), *ics]
    else:
        return [np.sum(fitness)]

    # # Batch mode
    # assert len(x[0]) == problem_instance.size_dec_vector, "Number of elements in batch decision vector should match problem's decision vector size"
    # # Parallel evaluation for batch mode
    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(evaluate, x))

    # fitness_list = [np.sum(result[0]) for result in results]
    # icss = [result[1] for result in results]

    # if problem_instance.use_inequality_contraints:
    #     return [
    #         [fitness, *ics] 
    #         for fitness, ics in zip(fitness_list, icss)
    #     ]
    # else:
    #     return fitness_list
    
    
class Problem(BaseProblem):
    
    def get_bounds(self) -> tuple[np.ndarray[float | int], np.ndarray[float | int]]:
        
        # output = (self.box_bounds_lower, self.box_bounds_upper)
        # output = (
        #     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0], order='C'),
        #     np.array([ 0,  0,  1,  1,  0, 1,  0,  0,  0,  0,  0,0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1], order='C')
        # )
        # print(len(output[0]), len(output[1]), self.size_dec_vector)
        # output = (
        #     np.zeros((self.size_dec_vector, ), order='C'),# * np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0.]),
        #     np.ones((self.size_dec_vector, ), order='C') * 2# * np.array([ 0. ,  0. ,  2.0,  2.0,  0. , 2.,  0. ,  0. ,  0. ,  0. ,  0. ,0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  1. ,  0. ,  0. ,  0. ,  1. ])
        # )
        # print(output)
        return flatten_list(self.box_bounds_lower), flatten_list(self.box_bounds_upper)
        
    
    def fitness(self, x: np.ndarray[float | int], store_x: bool = True) -> list[float]:
        # return [0.0]
        
        output = evaluate_fitness(self, x)
        
        # Store decision vector
        if store_x:
            self.x_evaluated.append(x.tolist())
            self.fitness_history.append(output[0])
        
        # return evaluate_fitness(self, x)
        return output
    
    # def batch_fitness(self, dvs: np.ndarray[float | int], store_x: bool = True) -> list[float]:
        
    #     # Convert a batch of decision vectors, dvs, stored contiguously
    #     # to a list of decision vectors: [dv1, dv2, ..., dvn] -> [[dv1], [dv2], ..., [dvn]]
    #     x: list[np.ndarray[float | int]] = []
    #     for idx_start in range(0, len(dvs)-1, step=self.size_dec_vector):
    #         x.append(dvs[idx_start:idx_start+self.size_dec_vector])
        
    #     if store_x:
    #         self.x_evaluated.extend(x)
    #         # TODO: Add fitness
            
    #     output_list = evaluate_fitness(self, x)
        
    #     # Return a contiguous array by converting the list of outputs:
    #     # [[out1], [out2], ..., [outn]] -> [out1, out2, ..., outn] 
    #     return np.array(output_list).flatten()
    
    def get_nic(self) -> int:
        """ Get number of inequality constraints """
        return self.n_dec_vars if self.use_inequality_contraints else 0

    def get_nix(self) -> int:
        """ Get integer dimension """
        return sum([getattr(self.dec_var_updates, var_id) for var_id in self.dec_var_int_ids])
    
    # def gradient(self, x: np.ndarray[float | int]) -> list[float]:
    #     return pg.estimate_gradient(lambda x: self.fitness(x), x)
