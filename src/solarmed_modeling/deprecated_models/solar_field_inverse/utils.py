import math
from typing import Literal
import time
import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props
from solarmed_modeling.metrics import calculate_metrics
from solarmed_modeling.utils import resample_results

from . import supported_eval_alternatives, ModelParameters, FixedModelParameters
from . import solar_field_inverse_model as model

def evaluate_model(
    df: pd.DataFrame, sample_rate: int, model_params: ModelParameters,
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    alternatives_to_eval: list[Literal["fsolve-direct-model"]] = supported_eval_alternatives,
    log_iteration: bool = False, base_df: pd.DataFrame = None,
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    """
    Evaluate the inverse solar field model using different alternatives and calculate performance metrics.

    Args:
        df: DataFrame containing the input data for the model.
        sample_rate: Sampling rate in seconds.
        model_params: ModelParameters object containing the model parameters.
        alternatives_to_eval: List of alternatives to evaluate. Supported alternatives are "standard", "no-delay", and "constant-water-props".
        log_iteration: Boolean flag to log each iteration.
        base_df: Dataframe with a base sample rate. If provided, the model outputs will be resampled to its sample rate 
        and used to calculate the metrics. Optional

    Raises:
        ValueError: If an unsupported alternative is provided in alternatives_to_eval.

    Returns:
        tuple: A tuple containing a list of DataFrames with the model outputs and a list of dictionaries with the performance metrics.
    """
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval])
    out_var_id: str = "qsf"
    
    span = math.ceil(600 / sample_rate) # 600 s
    idx_start = np.max([span, 2]) # idx_start-1 should at least be one 
    
    # Experimental (reference) outputs, used lated in performance metrics evaluation
    if base_df is None:
        idx_start_ref = idx_start
        ref_df = df
    else:
        if base_df.index.freq.n > df.index.freq.n:
            raise ValueError(f"Base dataframe can't have a lower sample rate than the input dataframe: base df rate ({base_df.index.freq.n}) < evaluting df rate({df.index.freq.n})")
        
        idx_start_ref = int(round(600 / base_df.index.freq.n, 0))
        ref_df = base_df
    out_ref = ref_df.iloc[idx_start_ref:][out_var_id].values
    
    # Initialize particular variables for earch alternative that requires it
    # water_props = None
    # if "constant-water-props" in alternatives_to_eval:
    water_props: w_props = w_props(
        P=0.2, T=90 + 273.15
    )  # P=2 bar  -> 0.2MPa, T in K, average working solar field temperature
        
    # Initialize result vectors
    # q_sf_mod   = np.zeros(len(df), dtype=float)
    outs_mod: list[np.ndarray[float]] = [np.zeros(len(df) - idx_start, dtype=float) for _ in alternatives_to_eval]
    stats = []
    
    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        out_mod = outs_mod[alt_idx]
        out_mod[0] = df.iloc[idx_start][out_var_id]
        
        # Initial values
        qsf_ant = df.iloc[0:idx_start]["qsf"].values
        out_mod[0] = df.iloc[idx_start]["qsf"]
        
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        # Evaluate model
        start_time_alt = time.time()
        for i in range(idx_start + 1, len(df)):
            ds = df.iloc[i]
            j = i - idx_start
            start_time = time.time()
            
            qsf_ant = np.roll(qsf_ant, -1)
            qsf_ant[-1] = out_mod[j-1]
            
            if alt_id == "fsolve-direct-model":
                out_mod[j] = model(
                    Tin=df.iloc[i-span:i]["Tsf_in"].values, 
                    Tout=df.iloc[i]["Tsf_out"],
                    Tout_ant=df.iloc[i-1]["Tsf_out"],
                    q_ant=qsf_ant, 
                    I=ds["I"],
                    Tamb=ds["Tamb"],
                    
                    sample_time=sample_rate,
                    water_props=water_props,
                    model_params=model_params,
                    fixed_model_params=fixed_model_params,
                )
            else:
                raise ValueError(
                    f"Unsupported alternative {alt_id}"
                )
                
            elapsed_time = time.time() - start_time
            if log_iteration:
                logger.info(
                    f"[{alt_id}] Iteration {i} / {len(df)}. Elapsed time: {elapsed_time:.5f} s. Error: {abs(out_mod[j]-out_ref[j]):.2f}"
                )
            # End of iteration evaluation
                
        # End of alternative evaluation
        elapsed_time = time.time() - start_time_alt
        
        if base_df is None:
            out_metrics = out_mod
        else:
            # Resample out to base_df sample rate using ffill
            out_metrics = resample_results(out_mod, new_index=base_df.index[idx_start_ref:], 
                                           current_index=df.index[idx_start:], reshape=True)
            
        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out_metrics, out_ref), 
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "model_parameters": model_params.__dict__,
            "sample_rate": sample_rate
        })
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.1f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} m³/h")

        dfs: list[pd.DataFrame] = [
            pd.DataFrame(out, columns=[out_var_id], index=df.index[idx_start:])
            for out in outs_mod
        ]

    return dfs, stats