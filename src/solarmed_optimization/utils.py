from dataclasses import asdict
import math
from typing import Any, Type, Literal
import time
from loguru import logger
import numpy as np
import pandas as pd
from solarmed_optimization import (DecisionVariables, 
                                   DecisionVariablesUpdates, 
                                   EnvironmentVariables, 
                                   dump_at_index_dec_vars)
from solarmed_modeling.solar_med import SolarMED


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        initial_states = kwargs.get('initial_states', None)
        max_step_idx = kwargs.get('max_step_idx', None)

        logger.info(f"Function {func.__name__} took {(end_time - start_time):.2f} seconds to run. Evaluated initial states {initial_states} for {max_step_idx} steps")
        return result
    return wrapper

def get_nested_attr(d: dict, attr: str) -> Any:
    """ Get nested attributes separated by a dot in a dictionary """
    keys = attr.split('.')
    for key in keys:
        d = d.get(key, None)
    return d
    
    
def forward_fill_resample(source_array: np.ndarray, target_size: int, dtype: Type = None) -> np.ndarray:
    """Resample a smaller array to a larger array by forward filling

        Args:
            source_array (np.ndarray): Original array
            target_size (int): Size of the target array

        Returns:
            np.ndarray: Resampled array
            
        Example usage
            source_array = np.array([1, 2, 3])
            target_size = 10
            resampled_array = resample_array(source_array, target_size)
            assert len(resampled_array) == target_size, "Resampled array size mismatch"
    """
    if len(source_array) > target_size:
        logger.warning(f"Source array size {len(source_array)} is greater than to target size {target_size}. Returning trimmed source array")
        return source_array[:target_size]
    if len(source_array) == target_size:
        return source_array
    
    dtype: Type = type(source_array[0]) if dtype is None else dtype
    
    span = target_size // len(source_array)  # Integer division
    remainder = target_size % len(source_array)  # Remaining slots to fill
    
    # Repeat each element by span times
    resampled_array = np.repeat(source_array, span)
    
    # Forward fill the remaining slots if target_size isn't a multiple of source_array length
    if remainder > 0:
        resampled_array = np.concatenate((resampled_array, np.repeat(source_array[-1], remainder)))
    
    return resampled_array.astype(dtype)


def downsample_by_segments(source_array:  np.ndarray, target_size: int, dtype: Type = None) -> np.ndarray:
    """
    Downsamples the source array to the target size by selecting the mean value 
    in each segment of the array.

    Parameters:
        source_array (np.ndarray): The input array to be downsampled.
        target_size (int): The desired size of the output array.

    Returns:
        np.ndarray: The downsampled array, where each element is the mean value 
        of a segment from the input array.
    
    Raises:
        ValueError: If target_size is less than 1 or greater than the size of source_array.
        
    Example usage
        source_array = np.array([1, 1000, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        target_size = 5
        downsampled_array = downsample_by_segments(source_array, target_size)

        print("Original array:", source_array)
        print("Downsampled array:", downsampled_array)
        assert len(downsampled_array) == target_size, "Downsampled array size mismatch"
    """
    if target_size < 1 or target_size > len(source_array):
        raise ValueError("target_size must be between 1 and the size of the source array.")

    dtype = dtype or source_array[0].dtype

    segment_size = len(source_array) / target_size
    return np.array([
        source_array[int(i * segment_size):int((i + 1) * segment_size)].mean()
        for i in range(target_size)
    ]).astype(dtype)


def decision_vector_to_decision_variables(x: np.ndarray, dec_var_updates: DecisionVariablesUpdates,
                                          span: Literal["optim_window", "optim_step"],
                                          sample_time_mod: int, 
                                          optim_window_time: int,
                                          sample_time_opt: int = None) -> DecisionVariables:
    """ From decision vector x to DecisionVariables instance
        The decision vector is the one defined in the problem class:
        x = [ [1,...,n_updates_x1], [1,...,n_updates_x2], ..., [1,...,n_updates_xNdec_vars] ]
        where n_updates_xi is the number of updates for decision variable i along the prediction horizon
        
        The resulting instance contains a vector for every decision variable with
        a sample time equal to the one of the model.
    
        - When span is set to optim window, for every decision variable, all its
        updates are included in the resulting Decision Variables instance and the
        number of elements is equal to the number of model evaluations along the 
        optimization window.
        
        - When span is set to optim step, for every decision variable, only the updates
        contained within one optimization step are included in the resulting instance, 
        and the number of elements is equal to the number of model evaluations along 
        one optimization step.
    """
    
    assert span in ["optim_window", "optim_step"], "span should be either 'optim_window' or 'optim_step'"
    if span == "optim_step":
        assert sample_time_opt is not None, "If span is 'optim_step', sample_time_opt should be provided"
    
    if span == "optim_step":
        # As many model evaluations as samples that fit in the optimization step 
        n_evals_mod = math.floor(sample_time_opt / sample_time_mod)
    else:
        # As many model evaluations as samples that fit in the optimization window
        n_evals_mod = math.floor(optim_window_time / sample_time_mod)
        
    # Build the decision variables dictionary in which every variable is "resampled" to the model sample time
    decision_dict: dict[str, np.ndarray] = {}
    cnt = 0
    for var_id, num_updates_optim_window in asdict(dec_var_updates).items():
        
        if span == "optim_step":
            # Only the updates in the optimization step are considered
            # |o----o----o----o--| : 4 updates in the optimization window (size = optim_window_time)
            # |________| : 2 updates in the optimization step (size = sample_time_opt)
            num_updates = math.floor(num_updates_optim_window * sample_time_opt / optim_window_time)
            # print(f"{num_updates=}")
        else:
            # All the updates in the optimization window are considered
            num_updates = num_updates_optim_window
            
        decision_dict[var_id] = forward_fill_resample(x[cnt:cnt+num_updates], target_size=n_evals_mod)
        cnt += num_updates_optim_window
    
    return DecisionVariables(**decision_dict)
        
def compute_dec_var_differences(dec_vars: dict[str, float], model_dec_vars: dict[str, float], model_dec_var_ids: list[str]) -> np.ndarray[float]:
    """ This implementation compared to initializing two dataframes is much faster:
    # Results
    {
        'list_comprehension_time': 0.00038123130798339844,
        'dataframe_time': 0.0037851333618164062,
    }
    """
    # Align both dictionaries' values using the common order of model_dec_var_ids
    # ordered_dv_values = [dv_dict[key] for key in model_dec_var_ids] # Already ordered
    dec_vars_values: list[float] = list(dec_vars.values())
    ordered_model_values: list[float] = [model_dec_vars[key] for key in model_dec_var_ids]

    # Compute absolute differences
    return np.abs(
        np.array(dec_vars_values, dtype=float) - np.array(ordered_model_values, dtype=float)
    )
    
def validate_dec_var_updates(dec_var_updates: DecisionVariablesUpdates, optim_window_time: int, sample_time_mod: int) -> None:
    """ 
    Validate that the number of updates for each decision variable is within the bounds
    of the optimization window or optimization step.
    """
    n_evals_mod_in_hor_window: int = math.floor(optim_window_time  / sample_time_mod)
    max_dec_var_updates: int = n_evals_mod_in_hor_window
    min_dec_var_updates: int = 1

    for var_id, value in asdict(dec_var_updates).items():
        try:
            assert value <= max_dec_var_updates and value >= min_dec_var_updates, \
            f"Invalid number of updates for variable {var_id}, n updates = {value}, should be {min_dec_var_updates} <= n updates <= {max_dec_var_updates}"
        except AssertionError as e:
            logger.warning(f"{e}. Setting value to bound")
            
            if value < min_dec_var_updates:
                setattr(dec_var_updates, var_id, min_dec_var_updates)
            else:
                setattr(dec_var_updates, var_id, max_dec_var_updates)
            

def evaluate_model(model: SolarMED, 
                   dec_vars: DecisionVariables, 
                   env_vars: EnvironmentVariables,
                   n_evals_mod: int,
                   mode: Literal["optimization", "evaluation"] = "optimization",
                   model_dec_var_ids: list[str] = None,
                   df_mod: pd.DataFrame = None,
                   df_start_idx: int = None) -> pd.DataFrame | float:
    """ Evaluate the model for a given decision vector and environment variables
        n_evals_mod is the number of model evaluations, whose value depends on what
        is being evaluated:
        - If mode is optimization, n_evals_mod should be the number of model evaluations in the optimization window (optim_window_time // sample_time_mod)
        - If mode is evaluation, n_evals_mod should be the number of model evaluations in one optimization step (sample_time_opt // sample_time_mod)
    """
    if mode == "optimization":
        assert model_dec_var_ids is not None, "`model_dec_var_ids` is required in `mode` is set to 'optimization'"
    
    # if df_mod is None and mode == "evaluation":
    #     df_mod = model.to_dataframe()
        
    if mode == "optimization":
        benefit: np.ndarray[float] = np.zeros((n_evals_mod, ))
        ics: np.ndarray[float] = np.zeros((n_evals_mod, len(model_dec_var_ids)))
    
    # dec_var_ids = list(asdict(dec_vars).keys())
    for step_idx in range(n_evals_mod):
        dv: DecisionVariables = dump_at_index_dec_vars(dec_vars, step_idx)
        
        model.step(
            # Decision variables
            ## Thermal storage
            qts_src = dv.qts_src * dv.ts_active,
            
            ## Solar field
            qsf = dv.qsf * dv.sf_active,
            
            ## MED
            qmed_s = dv.qmed_s * dv.med_active,
            qmed_f = dv.qmed_f * dv.med_active,
            Tmed_s_in = dv.Tmed_s_in,
            Tmed_c_out = dv.Tmed_c_out,
            med_vacuum_state = dv.med_vac_state,
            
            ## Environment
            I=env_vars.I[step_idx],
            Tamb=env_vars.Tamb[step_idx],
            Tmed_c_in=env_vars.Tmed_c_in[step_idx],
            wmed_f=env_vars.wmed_f[step_idx] if env_vars.wmed_f is not None else None,
        )
        
        if mode == "optimization":
            # Inequality contraints, decision variables should be the same after model evaluation: |dec_vars-dec_vars_model| < tol
            ics[step_idx, :] = compute_dec_var_differences(dec_vars=asdict(dv), 
                                                        model_dec_vars=model.model_dump(include=model_dec_var_ids),
                                                        model_dec_var_ids=model_dec_var_ids)
            benefit[step_idx] = model.evaluate_fitness_function(
                cost_e=env_vars.cost_e[step_idx],
                cost_w=env_vars.cost_w[step_idx]
            )
        if mode == "evaluation":
            df_mod = model.to_dataframe(df_mod)
                
    if mode == "optimization":
        # TODO: Add inequality constraints, at least for logical variables
        return np.sum(benefit), ics.mean(axis=0)
    elif mode == "evaluation":
        if df_start_idx is not None:
            df_mod.index = pd.RangeIndex(start=df_start_idx, stop=len(df_mod)+df_start_idx)
        return df_mod#, ics # ic temporary to validate
    else:
        raise ValueError(f"Invalid mode: {mode}")