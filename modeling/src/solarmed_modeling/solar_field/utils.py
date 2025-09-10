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
from . import solar_field_model as model

out_var_ids: list[str] = ["Tsf_out"]
input_var_ids: list[str] = ["Tsf_in", "qsf", "I", "Tamb"]

def evaluate_model(
    df: pd.DataFrame, 
    sample_rate: int, 
    model_params: ModelParameters, 
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    fsm_params = None,
    alternatives_to_eval: list[Literal["standard", "no-delay", "constant-water-props"]] = supported_eval_alternatives,
    log_iteration: bool = False, base_df: pd.DataFrame = None,
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    """
    Evaluate the solar field model using different alternatives and calculate performance metrics.

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
    out_var_id: str = out_var_ids[0]
    
    span = math.ceil(600 / sample_rate) # 600 s
    idx_start = np.max([span, 2]) # idx_start-1 should at least be one 

    # Experimental (reference) outputs, used lated in performance metrics evaluation
    if base_df is None:
        idx_start_ref = idx_start
        ref_df = df
    else:
        if base_df.index.freq.n > df.index.freq.n:
            raise ValueError(f"Base dataframe can't have a lower sample rate than the input dataframe: base df rate ({base_df.index.freq.n}) < evaluting df rate ({df.index.freq.n})")
        
        idx_start_ref = int(round(600 / base_df.index.freq.n, 0))
        ref_df = base_df
    out_ref = ref_df.iloc[idx_start_ref:][out_var_id].values
    
    # Assert that input and output variables are present in the dataframe, and that there are no NaNs
    for var_id in input_var_ids + [out_var_id]:
        assert var_id in df.columns, f"Variable {var_id} not found in dataframe columns: {df.columns}"
        assert not df[var_id].isna().any(), f"Variable {var_id} contains NaN values: {df[var_id].isna().sum()} NaNs"

    # Initialize particular variables for earch alternative that requires it
    water_props = None
    if "constant-water-props" in alternatives_to_eval:
        water_props: w_props = w_props(
            P=0.2, T=90 + 273.15
        )  # P=2 bar  -> 0.2MPa, T in K, average working solar field temperature
        
    if "no-delay" in alternatives_to_eval:
        logger.warning("The 'no-delay' alternative has hardcoded parameters. Make sure to adjust them if needed.")

    # Initialize result vectors
    outs_mod: list[np.ndarray[float]] = [np.zeros(len(df) - idx_start, dtype=float) for _ in alternatives_to_eval]
    stats = []

    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        out = outs_mod[alt_idx]
        out[0] = df.iloc[idx_start][out_var_id]
        
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        # Evaluate model
        start_time_alt = time.time()
        for i in range(idx_start + 1, len(df)):
            ds = df.iloc[i]
            j = i - idx_start
            start_time = time.time()
            
            if alt_id == "no-delay":
                out[j] = model(
                    Tin=ds["Tsf_in"],
                    q=ds["qsf"],
                    I=ds["I"],
                    Tamb=ds["Tamb"],
                    Tout_ant=out[j - 1],
                    sample_time=sample_rate,
                    consider_transport_delay=False,
                    model_params=ModelParameters(
                        beta=1.1578e-2,
                        H=3.1260,
                        gamma=0.0471,
                    )
                )
            elif alt_id == "standard":
                out[j] = model(
                    Tin=df.iloc[i - span : i][
                        "Tsf_in"
                    ].values,  # From current value, up to idx_start samples before
                    q=df.iloc[i - span : i][
                        "qsf"
                    ].values,  # From current value, up to idx_start samples before
                    I=ds["I"],
                    Tamb=ds["Tamb"],
                    Tout_ant=out[j - 1],
                    sample_time=sample_rate,
                    consider_transport_delay=True,
                    water_props=None,
                    model_params=model_params,
                    fixed_model_params=fixed_model_params
                )
            elif alt_id == "constant-water-props":
                out[j] = model(
                    Tin=df.iloc[i - span : i][
                        "Tsf_in"
                    ].values,  # From current value, up to idx_start samples before
                    q=df.iloc[i - span : i][
                        "qsf"
                    ].values,  # From current value, up to idx_start samples before
                    I=ds["I"],
                    Tamb=ds["Tamb"],
                    Tout_ant=out[j - 1],
                    sample_time=sample_rate,
                    consider_transport_delay=True,
                    water_props=water_props,
                    model_params=model_params,
                    fixed_model_params=fixed_model_params
                )
            else:
                raise ValueError(
                    f"Unsupported alternative {alt_id}, options are: {supported_eval_alternatives}"
                )

            elapsed_time = time.time() - start_time
            if log_iteration:
                logger.info(
                    f"[{alt_id}] Iteration {i} / {len(df)}. Elapsed time: {elapsed_time:.5f} s. Error: {abs(out[j]-out_ref[j]):.2f}"
                )
            # End of iteration evaluation
                
        # End of alternative evaluation
        elapsed_time = time.time() - start_time_alt
        
        if base_df is None:
            out_metrics = out
        else:
            # Resample out to base_df sample rate using ffill
            out_metrics = resample_results(out, new_index=base_df.index[idx_start_ref:], 
                                           current_index=df.index[idx_start:], reshape=True)
        
        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out_metrics, out_ref),
            "metrics_per_variable": {
                out_var_id: calculate_metrics(out_metrics, out_ref)
            },
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "model_parameters": model_params.__dict__,
            "sample_rate": sample_rate
        })
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.1f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")

        dfs: list[pd.DataFrame] = [
            pd.DataFrame(out, columns=[out_var_id], index=df.index[idx_start:])
            for out in outs_mod
        ]

    return dfs, stats