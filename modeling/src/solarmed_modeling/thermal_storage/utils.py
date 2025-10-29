from typing import Literal
import time
import numpy as np
import pandas as pd
from iapws import IAPWS97 as w_props
from loguru import logger

from solarmed_modeling.metrics import calculate_metrics
from solarmed_modeling.utils import resample_results

from . import ModelParameters, supported_eval_alternatives, FixedModelParameters
from . import thermal_storage_two_tanks_model as model

Th_labels: list[str] = ['Tts_h_t', 'Tts_h_m', 'Tts_h_b']
Tc_labels: list[str] = ['Tts_c_t', 'Tts_c_m', 'Tts_c_b']
T_labels: list[str] = Th_labels + Tc_labels

def evaluate_model(
    df: pd.DataFrame, 
    sample_rate: int, 
    model_params: ModelParameters, 
    fixed_model_params: FixedModelParameters,
    # fsm_params: None = None,
    alternatives_to_eval: list[Literal["standard", "constant-water-props"]] = supported_eval_alternatives,
    log_iteration: bool = False, base_df: pd.DataFrame = None,
    Th_labels: list[str] = ['Tts_h_t', 'Tts_h_m', 'Tts_h_b'],
    Tc_labels: list[str] = ['Tts_c_t', 'Tts_c_m', 'Tts_c_b'],
    **kwargs
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    """
    Evaluate the thermal storage model using different alternatives and calculate performance metrics.

    Args:
        df: DataFrame containing the input data for the model.
        sample_rate: Sampling rate in seconds.
        model_params: ModelParameters object containing the model parameters.
        alternatives_to_eval: List of alternatives to evaluate. Supported alternatives are "standard", and "constant-water-props".
        log_iteration: Boolean flag to log each iteration.
        base_df: Dataframe with a base sample rate. If provided, the model outputs will be resampled to its sample rate 
        and used to calculate the metrics. Optional

    Raises:
        ValueError: If an unsupported alternative is provided in alternatives_to_eval.

    Returns:
        tuple: A tuple containing a list of DataFrames with the model outputs and a list of dictionaries with the performance metrics.
    """
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval])

    idx_start = 0
    N = len(Th_labels)
    
    out_var_ids: list[str] = Th_labels + Tc_labels  # Output variable ids for the model metrics

    # Experimental (reference) outputs, used later in performance metrics evaluation
    
    # if base_df is None:
    #     out_ref = np.concatenate((df[Th_labels].values[idx_start:], df[Tc_labels].values[idx_start:]), axis=1)
    # else:
    #     out_ref = np.concatenate((base_df[Th_labels].values[idx_start:], base_df[Tc_labels].values[idx_start:]), axis=1)
        
    ref_df = df if base_df is None else base_df
    out_ref = ref_df.iloc[idx_start:][T_labels].values
        
    # Initialize particular variables for earch alternative that requires it
    water_props = None
    if "constant-water-props" in alternatives_to_eval:
        water_props: tuple[w_props, w_props] = (
            w_props(P=0.2, T=90 + 273.15), # P=2 bar  -> 0.2MPa, T in K, average working temperature of hot tank
            w_props(P=0.2, T=65 + 273.15)  # P=2 bar  -> 0.2MPa, T in K, average working temperature of cold tank
        )

    # Initialize result vectors
    outs_mod: list[np.ndarray[float]] = [np.zeros((len(df) - idx_start, N*2), dtype=float) for _ in alternatives_to_eval]
    stats = []

    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        out = outs_mod[alt_idx]
        out[0] = np.array( [df.iloc[idx_start][T] for T in Th_labels + Tc_labels] )
        
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        
        # Evaluate model
        start_time_alt = time.time()
        for i in range(idx_start + 1, len(df)):
            ds = df.iloc[i]
            j = i - idx_start
            start_time = time.time()
            
            if alt_id == "standard":
                out_h, out_c = model(
                    Ti_ant_h=out[j-1][:N], Ti_ant_c=out[j-1][N:], # ºC, ºC
                    Tt_in=ds["Tts_h_in"],  # ºC
                    Tb_in=ds["Tts_c_in"],  # ºC
                    Tamb=ds["Tamb"],  # ºC

                    qsrc=ds["qts_src"],  # m³/h
                    qdis=ds["qts_dis"],  # m³/h

                    model_params=model_params,
                    sample_time=sample_rate, 
                    fixed_model_params=fixed_model_params,
                    water_props=None,
                )
            elif alt_id == "constant-water-props":
                out_h, out_c = model(
                    Ti_ant_h=out[j-1][:N], Ti_ant_c=out[j-1][N:], # ºC, ºC
                    Tt_in=ds["Tts_h_in"],  # ºC
                    Tb_in=ds["Tts_c_in"],  # ºC
                    Tamb=ds["Tamb"],  # ºC

                    qsrc=ds["qts_src"],  # m³/h
                    qdis=ds["qts_dis"],  # m³/h

                    model_params=model_params,
                    fixed_model_params=fixed_model_params,
                    sample_time=sample_rate, 
                    water_props=water_props,
                )
            else:
                raise ValueError(
                    f"Unsupported alternative {alt_id}, options are: {supported_eval_alternatives}"
                )
                
            out[j] = np.concatenate((out_h, out_c), axis=0)
            elapsed_time = time.time() - start_time

            if log_iteration:
                logger.info(
                    f"[{alt_id}] Iteration {i} / {len(df)}. Elapsed time: {elapsed_time:.5f} s. Error: {abs(out[j]-out_ref[j]):.2f}"
                )
        
        elapsed_time = time.time() - start_time_alt
        
        if base_df is None:
            out_metrics = out
        else:
            # Resample out to base_df sample rate using ffill
            out_metrics = resample_results(out, new_index=base_df.index, current_index=df.index[idx_start:])
                
        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out_metrics, out_ref),
            "metrics_per_variable": {
                out_var_id: calculate_metrics(out_metrics[:,out_idx], out_ref[:,out_idx], metrics=['RMSE', 'MAE', 'MSE', 'R2', 'NRMSE', 'MAPE'])
                for out_idx, out_var_id in enumerate(out_var_ids)    
            },
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "model_parameters": model_params.__dict__,
            "sample_rate": sample_rate
        })
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.3f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")

        # l[i] = calculate_total_pipe_length(q=df.iloc[i:i-idx_start:-1]['qsf'].values, n=60, sample_time=10, equivalent_pipe_area=7.85e-5)

        # df['Tsf_l2_pred'] = Tsf_out_mod

        # Estimate pipe length
        # l[l>1].mean() # 6e7

        dfs: list[pd.DataFrame] = [
            pd.DataFrame(out, columns=Th_labels + Tc_labels, index=df.index[idx_start:])
            for out in outs_mod
        ]

    return dfs, stats
