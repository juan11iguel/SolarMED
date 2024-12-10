import math
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from solarmed_optimization.path_explorer.utils import import_results
from solarmed_optimization.utils import forward_fill_resample, downsample_by_segments
import numpy as np
from typing import Type, get_args
from loguru import logger

from solarmed_modeling.solar_med import SolarMED
from solarmed_optimization import (EnvironmentVariables,
                                   DecisionVariables,
                                   DecisionVariablesUpdates,
                                   VarIdsOptimToFsmsMapping,
                                   OptimVarIdstoModelVarIdsMapping,
                                   FsmData)

# Step from SolarMED
# def step(
#     self,
#     qts_src: float,  # Thermal storage decision variables
#     qsf: float, # Solar field decision variables
#     qmed_s: float, qmed_f: float, Tmed_s_in: float, Tmed_c_out: float,  # MED decision variables
#     Tmed_c_in: float, Tamb: float, I: float, wmed_f: float = None,  # Environment variables
#     med_vacuum_state: int | MedVacuumState = 2,  # Optional, to provide the MED vacuum state (OFF, LOW, HIGH)
# ) -> None:    

@dataclass
class MinlpProblem:
    """
    x: decision vector.
    - shape: ( n inputs x n horizon )
    - structure:
        X = [x[0,0] x[1,0] x[2,0] ... x[Ninputs,0], ... x[0,Nhorizon] ... x[Ninputs,Nhorizon]]
    
    - Decision vector components: See `DecisionVariables`  
    - Environment variables: See `EnvironmentVariables`
    """

    model: SolarMED  # SolarMED model instance
    optim_window_size: int  # Optimization window size in seconds
    env_vars: EnvironmentVariables # Environment variables
    dec_var_updates: DecisionVariablesUpdates # Decision variables updates
    fsm_med_data: FsmData # FSM data for the MED system
    fsm_sfts_data: FsmData # FSM data for the SFTS systems
    
    # Computed attributes
    x: np.ndarray[float] = None # Decision variables values vector
    dec_var_ids: list[str] = None # All decision variables ids
    dec_var_int_ids: list[str] = None # Logical / integer decision variables ids
    dec_var_real_ids: list[str] = None  # Real decision variables ids
    dec_var_dtypes: list[Type] = None  # Decision variable data types
    ni: int = None # Number of logical / integer decision variables
    nr: int = None # Number of real decision variables
    integer_dec_vars_mapping: dict[str, np.ndarray[list[int]]] = None # Mapping from integer decision variables to FSMs inputs
    model_dict: dict = None # SolarMED model dumped instance
    n_evals_mod_in_hor_window: int = None # Number of model evaluations per optimization window
    n_evals_mod_in_opt_step: int = None # Number of model evaluations per optimization step
    sample_time_mod: int = None # Model sample time
    sample_time_opt: int = None # Optimization sample time
    box_bounds_lower: list[np.ndarray[float | int]] = None # Lower bounds for the decision variables (in list of arrays format). Updated every time `get_bounds` is called
    box_bounds_upper: list[np.ndarray[float | int]] = None # Upper bounds for the decision variables (in list of arrays format). Updated every time `get_bounds` is called
    
    def __init__(self, 
                 model: SolarMED,
                 sample_time_opt: int,
                 optim_window_time: int,
                 dec_var_updates: DecisionVariablesUpdates,
                 env_vars: EnvironmentVariables,
                 fsm_valid_sequences: dict[ str, list[list] ],
                 fsm_data_path: Path = Path("../results"),):

        self.optim_window_size = optim_window_time
        self.dec_var_updates = dec_var_updates
        self.env_vars = env_vars
        
        self.model_dict = model.dump_instance()
        self.model = SolarMED(**self.model_dict) # To make sure we don't modify the original instance
        
        self.sample_time_mod = self.model.sample_time
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
        n_horizon = dec_var_updates.med_active
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
        n_horizon = dec_var_updates.sf_active
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

    def __post_init__(self, ) -> None:
        logger.info(f"""{self.get_name()} initialized.
                    - Size of decision vector: {self.size_dec_vector} elements
                    - Decision variable ids: {self.dec_var_ids}
                    - Number of updates per dec.var: {[getattr(self.dec_var_updates, var_id) for var_id in self.dec_var_ids]}
                    - other...""")

    def get_bounds(self, readable_format: bool = False, debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
            var_id: np.full((getattr(self.dec_var_updates, var_id), ), list[int], dtype=object) for var_id in self.dec_var_int_ids
        }
        box_bounds_lower: list[np.ndarray[float | int]] = [
            np.full((n_updates, ), np.nan, dtype=float) for n_updates in asdict(self.dec_var_updates).values()
        ]
        box_bounds_upper: list[np.ndarray[float | int]] = [
            np.full((n_updates, ), np.nan, dtype=float) for n_updates in asdict(self.dec_var_updates).values()
        ]

        # Handle logical / integer variables for both FSMs
        # For each FSM, get the possible paths from the initial state and the valid inputs, to set its bounds and mappings
        for initial_state, fsm_data in zip([self.model._sf_ts_fsm.state, self.model._med_fsm.state],
                                           [self.fsm_sfts_data, self.fsm_med_data]):
            
            paths_df = fsm_data.paths_df
            valid_inputs = fsm_data.valid_inputs
            input_ids: list[str] = fsm_data.metadata["input_ids"] 

            # Extract indexes of possible paths from initial state
            paths_from_state_idxs: np.ndarray = paths_df[paths_df["0"] == initial_state.value].index.to_numpy() # dim: (n paths, )
            # Get valid inputs from initial states using those indexes
            valid_inputs_from_state: np.ndarray = valid_inputs[paths_from_state_idxs] # dim: (n paths, n horizon, n inputs)

            # Get the unique discrete values for each input
            for input_idx, fsm_input_id in enumerate(input_ids): # For every input in the FSM
                optim_input_id = VarIdsOptimToFsmsMapping(fsm_input_id).name
                input_idx_in_dec_vars = self.dec_var_ids.index(optim_input_id)
                n_updates = getattr(self.dec_var_updates, optim_input_id)

                # Find unique valid inputs per step from all possible paths
                for step_idx in range(n_updates): # For every step
                    discrete_bounds = np.unique(valid_inputs_from_state[:, step_idx, input_idx])
                    # Update mapping
                    integer_dec_vars_mapping[optim_input_id][step_idx] = discrete_bounds
                    # Update bounds
                    box_bounds_upper[input_idx_in_dec_vars][step_idx] = len(discrete_bounds)-1
                    box_bounds_lower[input_idx_in_dec_vars][step_idx] = 0
                    # print(f"{optim_input_id=}, {step_idx=}, {discrete_bounds=}, bbox=[{box_bounds_lower[input_idx_in_dec_vars][step_idx]}, {box_bounds_upper[input_idx_in_dec_vars][step_idx]}]")
                
                if debug:
                    vals_str = [f"{i}: {db} --> [{lb}, {ub}]" for i, (db, lb, ub) in enumerate(zip(integer_dec_vars_mapping[optim_input_id], 
                                                                                        box_bounds_lower[input_idx_in_dec_vars], 
                                                                                        box_bounds_upper[input_idx_in_dec_vars]))]
                    print(f"IB | {self.model.get_state().name} | {optim_input_id}: {vals_str}")

        # Real variables bounds
        # Done manually for now
        # ['qsf', 'qts_src', 'qmed_s', 'qmed_f', 'Tmed_s_in', 'Tmed_c_out']
        def set_real_var_bounds(var_id: str, lower_limit: float | np.ndarray[float], 
                                upper_limit: float | np.ndarray[float], aux_logical_var_id: str  = None,
                                integer_dec_vars_mapping: dict[str, np.ndarray[list[int]]] = None) -> None:
            """
            Set the bounds for a real variable in the decision vector
            No need to pass the bounds arrays or return them, as they are modified in place (mutable)
            """
            input_idx_in_dec_vars = self.dec_var_ids.index(var_id)
            n_updates = getattr(self.dec_var_updates, var_id)
            if aux_logical_var_id is not None:
                # aux_logical_input_idx_in_dec_vars = self.dec_var_ids.index(aux_logical_var_id)
                integer_mapping = integer_dec_vars_mapping[aux_logical_var_id]
                
                # np.vectorize(lambda x: 1 if np.any(x > 0) else 0)(integer_mapping)
                integer_upper_value: np.ndarray[int] = np.array( [1 if np.any(bounds > 0) else 0 for bounds in integer_mapping], dtype=int )
                # np.vectorize(lambda x: np.min(x))(integer_mapping)
                integer_lower_value: np.ndarray[int] = np.array( [np.min(bounds) for bounds in integer_mapping], dtype=int )
                # integer_upper_value: np.ndarray[int] = box_bounds_upper[aux_logical_input_idx_in_dec_vars]
                # integer_lower_value: np.ndarray[int] = box_bounds_lower[aux_logical_input_idx_in_dec_vars]
                
                # print(f"{self.model.get_state().name} | {aux_logical_var_id}: {integer_upper_value=}, {integer_lower_value=}")
                
                if len(integer_upper_value) < n_updates:
                    integer_upper_value = forward_fill_resample(integer_upper_value, n_updates)
                    integer_lower_value = forward_fill_resample(integer_lower_value, n_updates)
                upper_value: np.ndarray[float] = upper_limit * integer_upper_value
                lower_value: np.ndarray[float] = lower_limit * integer_lower_value
                
                
            else:
                upper_value: np.ndarray[float] = upper_limit * np.ones((n_updates, ))
                lower_value: np.ndarray[float] = lower_limit * np.ones((n_updates, ))
            
            box_bounds_upper[input_idx_in_dec_vars] = upper_value
            box_bounds_lower[input_idx_in_dec_vars] = lower_value

        # Thermal storage
        set_real_var_bounds(
            var_id = 'qts_src', 
            lower_limit = self.model.fixed_model_params.ts.qts_src_min, 
            upper_limit = self.model.fixed_model_params.ts.qts_src_max, 
            aux_logical_var_id = 'ts_active',
            integer_dec_vars_mapping=integer_dec_vars_mapping,
        )
        # Solar field
        set_real_var_bounds(
            var_id = 'qsf', 
            lower_limit = self.model.fixed_model_params.sf.qsf_min, 
            upper_limit = self.model.fixed_model_params.sf.qsf_max, 
            aux_logical_var_id = 'sf_active',
            integer_dec_vars_mapping=integer_dec_vars_mapping,
        )
        # MED
        set_real_var_bounds(
            var_id = 'Tmed_s_in', 
            lower_limit = self.model.fixed_model_params.med.Tmed_s_min, 
            upper_limit = self.model.fixed_model_params.med.Tmed_s_max, 
            aux_logical_var_id = 'med_active',
            integer_dec_vars_mapping=integer_dec_vars_mapping,
        )
        
        Tmed_c_in = downsample_by_segments(source_array=self.env_vars.Tmed_c_in, target_size=self.dec_var_updates.Tmed_c_out)
        set_real_var_bounds(
            var_id = 'Tmed_c_out',
            lower_limit = Tmed_c_in, 
            upper_limit = Tmed_c_in+10, # A temperature delta of over 10ºC is unfeasible for the condenser
            aux_logical_var_id = 'med_active',
            integer_dec_vars_mapping=integer_dec_vars_mapping,
        )
        for var_id in ['qmed_s', 'qmed_f']:
            set_real_var_bounds(
                var_id = var_id, 
                lower_limit = getattr(self.model.fixed_model_params.med, f"{var_id}_min"), 
                upper_limit = getattr(self.model.fixed_model_params.med, f"{var_id}_max"), 
                aux_logical_var_id = 'med_active',
                integer_dec_vars_mapping=integer_dec_vars_mapping,
            )

        # np.set_printoptions(precision=1)
        # print(f"{[f'{var_id}: {bounds}' for var_id, bounds in zip(dec_var_ids, box_bounds_lower)]}")
        # print(f"{[f'{var_id}: {bounds}' for var_id, bounds in zip(dec_var_ids, box_bounds_upper)]}")
        # print(f"{integer_dec_vars_mapping=}")
        
        self.box_bounds_lower = box_bounds_lower
        self.box_bounds_upper = box_bounds_upper

        # Finally, concatenate each array to get the final bounds
        if not readable_format:
            box_bounds_lower = np.concatenate(box_bounds_lower)
            box_bounds_upper = np.concatenate(box_bounds_upper)
        self.integer_dec_vars_mapping = integer_dec_vars_mapping

        # print(f"{box_bounds_lower=}")
        # print(f"{box_bounds_upper=}")

        return box_bounds_lower, box_bounds_upper
    
    def fitness(self, x: np.ndarray | int) -> np.ndarray[float] | list[float]:

        model: SolarMED = SolarMED(**self.model_dict)
        benefit: np.ndarray[float] = np.zeros((self.n_evals_mod_in_hor_window, ))
        decision_dict: dict[str, np.ndarray] = {}
        
        # Build the decision variables dictionary in which every variable is "resampled" to the model sample time
        cnt = 0
        for var_id, num_updates in asdict(self.dec_var_updates).items():
            decision_dict[var_id] = forward_fill_resample(x[cnt:cnt+num_updates], target_size=self.n_evals_mod_in_hor_window)
            cnt += num_updates
        
        # TODO: All of this evaluation should be performed in a different function
        # not part of the class so it can be reused in different problem formulations
        dv = DecisionVariables(**decision_dict) 
        for step_idx in range(self.n_evals_mod_in_hor_window):
            
            model.step(
                # Decision variables
                ## Thermal storage
                qts_src = dv.qts_src[step_idx] * dv.ts_active[step_idx],
                
                ## Solar field
                qsf = dv.qsf[step_idx] * dv.sf_active[step_idx],
                
                ## MED
                qmed_s = dv.qmed_s[step_idx] * dv.med_active[step_idx],
                qmed_f = dv.qmed_f[step_idx] * dv.med_active[step_idx],
                Tmed_s_in = dv.Tmed_s_in[step_idx],
                Tmed_c_out = dv.Tmed_c_out[step_idx],
                med_vacuum_state = int(dv.med_vac_state[step_idx]),
                
                ## Environment
                I=self.env_vars.I[step_idx],
                Tamb=self.env_vars.Tamb[step_idx],
                Tmed_c_in=self.env_vars.Tmed_c_in[step_idx],
                wmed_f=self.env_vars.wmed_f[step_idx] if self.env_vars.wmed_f is not None else None,
            )
            
            benefit[step_idx] = model.evaluate_fitness_function(
                cost_e=self.env_vars.cost_e[step_idx],
                cost_w=self.env_vars.cost_w[step_idx]
            )
            
        # TODO: Add inequality constraints, at least for logical variables
        return np.sum(benefit)        

    def get_name(self) -> str:
        """ Problem’s name """
        return "SolarMED MINLP problem"

    def get_extra_info(self) -> str:
        """ Problem’s extra info. """
        return "\tDimensions: " + str(self.dim)