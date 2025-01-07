import numpy as np
import pygmo as pg
from solarmed_optimization.utils import flatten_list
from solarmed_optimization.problems import BaseMinlpProblem, evaluate_fitness

class MinlpProblem(BaseMinlpProblem):
    
    # def generate_bounds(self, readable_format: bool = False) -> tuple[np.ndarray[float | int], np.ndarray[float | int]] | tuple[list[np.ndarray[float | int]], list[np.ndarray[float | int]]]:
        
    #     problem_instance = self
        
    #     integer_dec_vars_mapping: dict[str, np.ndarray[list[int]]] = {
    #         var_id: np.full((getattr(problem_instance.dec_var_updates, var_id), ), list[int], dtype=object) for var_id in problem_instance.dec_var_int_ids
    #     }
    #     # Lower box-bounds
    #     lbb: list[np.ndarray[float | int]] = [
    #         np.full((n_updates, ), np.nan, dtype=float) for n_updates in asdict(problem_instance.dec_var_updates).values()
    #     ]
    #     # Upper box-bounds
    #     ubb: list[np.ndarray[float | int]] = [
    #         np.full((n_updates, ), np.nan, dtype=float) for n_updates in asdict(problem_instance.dec_var_updates).values()
    #     ]
        
    #     # Logical / integer variables bounds
    #     # For each FSM, get the possible paths from the initial state and the valid inputs, to set its bounds and mappings
    #     for initial_state, fsm_data in zip([problem_instance.model_dict["sf_ts_state"], problem_instance.model_dict["med_state"]],
    #                                        [problem_instance.fsm_sfts_data, problem_instance.fsm_med_data]):
            
    #         paths_df = fsm_data.paths_df
    #         valid_inputs = fsm_data.valid_inputs
    #         input_ids: list[str] = fsm_data.metadata["input_ids"] 

    #         # Extract indexes of possible paths from initial state
    #         paths_from_state_idxs: np.ndarray = paths_df[paths_df["0"] == initial_state.value].index.to_numpy() # dim: (n paths, )
    #         # Get valid inputs from initial states using those indexes
    #         valid_inputs_from_state: np.ndarray = valid_inputs[paths_from_state_idxs] # dim: (n paths, n horizon, n inputs)

    #         # Get the unique discrete values for each input
    #         for input_idx, fsm_input_id in enumerate(input_ids): # For every input in the FSM
    #             optim_input_id = VarIdsOptimToFsmsMapping(fsm_input_id).name
    #             input_idx_in_dec_vars = problem_instance.dec_var_ids.index(optim_input_id)
    #             n_updates = getattr(problem_instance.dec_var_updates, optim_input_id)

    #             # Find unique valid inputs per step from all possible paths
    #             for step_idx in range(n_updates): # For every step
    #                 discrete_bounds = np.unique(valid_inputs_from_state[:, step_idx, input_idx])
    #                 # Update mapping
    #                 integer_dec_vars_mapping[optim_input_id][step_idx] = discrete_bounds
    #                 # Update bounds
    #                 ubb[input_idx_in_dec_vars][step_idx] = float(len(discrete_bounds)-1)
    #                 lbb[input_idx_in_dec_vars][step_idx] = 0.0
    #                 # print(f"{optim_input_id=}, {step_idx=}, {discrete_bounds=}, bbox=[{box_bounds_lower[input_idx_in_dec_vars][step_idx]}, {box_bounds_upper[input_idx_in_dec_vars][step_idx]}]")
        
    #     # Real variables bounds
    #     # Thermal storage
    #     lbb, ubb = set_real_var_bounds(
    #         problem_instance=problem_instance, ub=ubb, lb=lbb,
    #         var_id = 'qts_src', 
    #         lower_limit = problem_instance.model_dict["fixed_model_params"]["ts"]["qts_src_min"], 
    #         upper_limit = problem_instance.model_dict["fixed_model_params"]["ts"]["qts_src_max"], 
    #         aux_logical_var_id = 'ts_active',
    #         integer_dec_vars_mapping=integer_dec_vars_mapping,
    #     )
    #     # Solar field
    #     lbb, ubb = set_real_var_bounds(
    #         problem_instance=problem_instance, ub=ubb, lb=lbb,
    #         var_id = 'qsf', 
    #         lower_limit = problem_instance.model_dict["fixed_model_params"]["sf"]["qsf_min"], 
    #         upper_limit = problem_instance.model_dict["fixed_model_params"]["sf"]["qsf_max"], 
    #         aux_logical_var_id = 'sf_active',
    #         integer_dec_vars_mapping=integer_dec_vars_mapping,
    #     )
    #     # MED
    #     lbb, ubb = set_real_var_bounds(
    #         problem_instance=problem_instance, ub=ubb, lb=lbb,
    #         var_id = 'Tmed_s_in', 
    #         lower_limit = problem_instance.model_dict["fixed_model_params"]["med"]["Tmed_s_min"], 
    #         upper_limit = problem_instance.model_dict["fixed_model_params"]["med"]["Tmed_s_max"], 
    #         aux_logical_var_id = 'med_active',
    #         integer_dec_vars_mapping=integer_dec_vars_mapping,
    #     )
        
    #     Tmed_c_in = downsample_by_segments(source_array=problem_instance.env_vars.Tmed_c_in, target_size=problem_instance.dec_var_updates.Tmed_c_out)
    #     lbb, ubb = set_real_var_bounds(
    #         problem_instance=problem_instance, ub=ubb, lb=lbb,
    #         var_id = 'Tmed_c_out',
    #         lower_limit = Tmed_c_in, 
    #         upper_limit = Tmed_c_in+10, # A temperature delta of over 10ºC is unfeasible for the condenser
    #         aux_logical_var_id = 'med_active',
    #         integer_dec_vars_mapping=integer_dec_vars_mapping,
    #     )
    #     for var_id in ['qmed_s', 'qmed_f']:
    #         lbb, ubb = set_real_var_bounds(
    #             problem_instance=problem_instance, ub=ubb, lb=lbb,
    #             var_id = var_id, 
    #             lower_limit = problem_instance.model_dict["fixed_model_params"]["med"][f"{var_id}_min"], 
    #             upper_limit = problem_instance.model_dict["fixed_model_params"]["med"][f"{var_id}_max"], 
    #             aux_logical_var_id = 'med_active',
    #             integer_dec_vars_mapping=integer_dec_vars_mapping,
    #         )
    #     problem_instance.integer_dec_vars_mapping = integer_dec_vars_mapping

    #     lbb = np.concatenate(lbb)
    #     ubb = np.concatenate(ubb)
        
    #     problem_instance.box_bounds_lower = lbb
    #     problem_instance.box_bounds_upper = ubb
        
    #     # assert len(lbb) == len(ubb) == problem_instance.size_dec_vector

    #     # return np.zeros((self.size_dec_vector, )), np.random.uniform(5, 10, self.size_dec_vector) #np.ones((self.size_dec_vector, ))
    #     # return get_bounds(self, readable_format=readable_format)
    #     return lbb, ubb
    
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
