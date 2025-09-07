from typing import Literal
import time
import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props
from solarmed_modeling.metrics import calculate_metrics
from solarmed_modeling.utils import resample_results
from solarmed_modeling.data_validation import within_range_or_nan_or_max, within_range_or_zero_or_max

from . import FixedModelParameters, supported_eval_alternatives
from . import MedModel


def evaluate_model(
    df: pd.DataFrame, 
    sample_rate: int,
    model_params: None = None,
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    alternatives_to_eval: list[Literal["standard"]] = supported_eval_alternatives,
    log_iteration: bool = False, base_df: pd.DataFrame = None,
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    """
    Evaluate the solar field model using different alternatives and calculate performance metrics.

    Args:
        df: DataFrame containing the input data for the model.
        sample_rate: Sampling rate in seconds.
        model_params: ModelParameters object containing the model parameters.
        alternatives_to_eval: List of alternatives to evaluate. Supported alternatives are "standard", "no-delay", and 
        "constant-water-props".
        log_iteration: Boolean flag to log each iteration.
        base_df: Dataframe with a base sample rate. If provided, the model outputs will be resampled to its sample rate 
        and used to calculate the metrics. Optional

    Raises:
        ValueError: If an unsupported alternative is provided in alternatives_to_eval.

    Returns:
        tuple: A tuple containing a list of DataFrames with the model outputs and a list of dictionaries with the 
        performance metrics.
    """
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval])

    idx_start = 0
    out_var_ids: list[str] = ["qmed_d", "qmed_c", "Tmed_s_out"]
    N = len(out_var_ids)
    fmp = fixed_model_params
    model = MedModel()

    # Experimental (reference) outputs, used lated in performance metrics evaluation
    # out_ref = np.concatenate(df.iloc[idx_start:]["Thx_p_out"].values, df.iloc[idx_start:]["Thx_s_out"].values)
    if base_df is None:
        out_ref = np.concatenate([df[out_var_ids].values[idx_start:]], axis=1)
    else:
        if base_df.index.freq.n > df.index.freq.n:
            raise ValueError(f"Base dataframe can't have a lower sample rate than the input dataframe: base df rate ({base_df.index.freq.n}) < evaluting df rate({df.index.freq.n})")
        
        out_ref = np.concatenate([base_df[out_var_ids].values[idx_start:]], axis=1)

    # Initialize particular variables for earch alternative that requires it

    # Initialize result vectors
    outs_mod: list[np.ndarray[float]] = [np.zeros((len(df) - idx_start, N), dtype=float) for _ in alternatives_to_eval]
    stats = []

    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        out = outs_mod[alt_idx]
        
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        # Evaluate model
        start_time_alt = time.time()
        for i in range(idx_start + 1, len(df)):
            ds = df.iloc[i]
            j = i - idx_start
            start_time = time.time()
            
            mmed_s = within_range_or_zero_or_max(ds["qmed_s"], range=(fmp.qmed_s_min, fmp.qmed_s_max), var_name="qmed_s")  # m³/h
            Tmed_s_in = within_range_or_nan_or_max(ds["Tmed_s_in"], range=(fmp.Tmed_s_min, fmp.Tmed_s_max), var_name="Tmed_s_in")  # ºC
            mmed_f = within_range_or_nan_or_max(ds["qmed_f"], range=(fmp.qmed_f_min, fmp.qmed_f_max), var_name="qmed_f")  # m³/h
            Tmed_c_out = ds["Tmed_c_out"]  # ºC
            Tmed_c_in = ds["Tmed_c_in"]  # ºC
            
            inputs = np.array([[
                    mmed_s,      # m³/h
                    Tmed_s_in,   # ºC
                    mmed_f,      # m³/h
                    Tmed_c_out,  # ºC
                    Tmed_c_in    # ºC
                ]])
                
            if alt_id == "standard":
                med_model_solved = False
                qmed_d = 0
                Tmed_s_out = Tmed_s_in
                qmed_c = 0
                while not med_model_solved and (Tmed_c_in < Tmed_c_out < 45):   
                    qmed_d, Tmed_s_out, qmed_c = model.predict(inputs)[0, :]
                    
                    if qmed_c > fmp.qmed_c_max:
                        Tmed_c_out = Tmed_c_out + 1
                    elif qmed_c < fmp.qmed_c_min:
                        Tmed_c_out = Tmed_c_out - 1
                    else:
                        med_model_solved = True
                
                # if not med_model_solved:
                #     qmed_d = 0.0
                #     Tmed_s_out = np.nan
                #     qmed_c = 0.0
            else:
                raise ValueError(
                    f"Unsupported alternative {alt_id}, options are: {supported_eval_alternatives}"
                )

            out[j] = np.array([qmed_d, Tmed_s_out, qmed_c])
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
            
            # if out_metrics.shape != out_ref.shape:
            #     raise ValueError(f"Output shape {out_metrics.shape} does not match reference shape {out_ref.shape}")
            
            
        df_mod = pd.DataFrame(out, columns=out_var_ids, index=df.index[idx_start:])
        
        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out_metrics, out_ref),
            "metrics_per_variable": {
                out_var_id: calculate_metrics(df_mod[out_var_id].values, df[out_var_id].values)
                for out_var_id in out_var_ids    
            },
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "fixed_model_parameters": fixed_model_params.__dict__,
            "sample_rate": sample_rate
        })
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.1f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")

    dfs: list[pd.DataFrame] = [
        pd.DataFrame(out, columns=out_var_ids, index=df.index[idx_start:])
        for out in outs_mod
    ]

    return dfs, stats
