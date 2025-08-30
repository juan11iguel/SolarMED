from typing import Literal
import time
import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props
from solarmed_modeling.metrics import calculate_metrics
from solarmed_modeling.utils import resample_results

from . import ModelParameters, supported_eval_alternatives
from . import heat_exchanger_model as model

def estimate_flow_secondary(Tp_in: float | np.ndarray[float], Ts_in: float | np.ndarray[float], 
                            qp: float | np.ndarray[float], Tp_out: float | np.ndarray[float], 
                            Ts_out: float | np.ndarray[float], water_props: tuple[w_props, w_props] = None) -> float | np.ndarray[float]:
    """
    Estimate the secondary flow rate in a heat exchanger.
    
    Parameters:
    Tp_in (float): Inlet temperature of the primary fluid (°C).
    Ts_in (float): Inlet temperature of the secondary fluid (°C).
    qp (float): Flow rate of the primary fluid (kg/s).
    Tp_out (float): Outlet temperature of the primary fluid (°C).
    Ts_out (float): Outlet temperature of the secondary fluid (°C).
    water_props (tuple[w_props, w_props], optional): Tuple containing water properties functions for primary and secondary fluids. Defaults to None.
    
    Returns:
    float: Estimated flow rate of the secondary fluid (kg/s). Returns NaN if the temperature difference between Ts_out and Ts_in is less than 1°C or if there are invalid temperature inputs.
    """
    # Pre-processing
    vectorized: bool = not isinstance(Tp_in, float)
    if water_props is not None:
        w_props_Tp_in, w_props_Ts_in = water_props
    
    if vectorized:
        if water_props is None:
            w_props_Tp_in = w_props(P=0.16, T=(Tp_in.mean() + Tp_out.mean()) / 2 + 273.15)
            w_props_Ts_in = w_props(P=0.16, T=(Ts_in.mean() + Ts_out.mean()) / 2 + 273.15)
    else:
        if np.abs(Ts_out - Ts_in) < 1:
            return np.nan
        try:
            if water_props is None:
                w_props_Tp_in = w_props(P=0.16, T=(Tp_in + Tp_out) / 2 + 273.15)
                w_props_Ts_in = w_props(P=0.16, T=(Ts_in + Ts_out) / 2 + 273.15)
        except Exception as e:
            logger.warning(f'Invalid temperature input values: Tp_in={Tp_in}, Ts_in={Ts_in}, Tp_out={Tp_out}, Ts_out={Ts_out} (ºC), returning NaN')
            return np.nan

    # Calculate secondary flow rate
    cp_Tp_in = w_props_Tp_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    cp_Ts_in = w_props_Ts_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    qs = qp * (cp_Tp_in * (Tp_in - Tp_out)) / (cp_Ts_in * (Ts_out - Ts_in))
    
    # Post-processing
    if vectorized:
        qs[ np.abs(Ts_out - Ts_in) < 1 ] = np.nan
        qs[ qs < 0 ] = 0
    else:
        qs = np.max([qs, 0])
        
    return qs


def evaluate_model(
    df: pd.DataFrame, sample_rate: int, model_params: ModelParameters,
    fixed_model_params: None = None,
    alternatives_to_eval: list[Literal["standard", "no-delay", "constant-water-props"]] = supported_eval_alternatives,
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
    out_var_ids: list[str] = ["Thx_p_out", "Thx_s_out"]
    N = len(out_var_ids)

    # Experimental (reference) outputs, used lated in performance metrics evaluation
    # out_ref = np.concatenate(df.iloc[idx_start:]["Thx_p_out"].values, df.iloc[idx_start:]["Thx_s_out"].values)
    if base_df is None:
        out_ref = np.concatenate([df[out_var_ids].values[idx_start:]], axis=1)
    else:
        if base_df.index.freq.n > df.index.freq.n:
            raise ValueError(f"Base dataframe can't have a lower sample rate than the input dataframe: base df rate ({base_df.index.freq.n}) < evaluting df rate({df.index.freq.n})")
        
        out_ref = np.concatenate([base_df[out_var_ids].values[idx_start:]], axis=1)

    # Initialize particular variables for earch alternative that requires it
    water_props = None
    if "constant-water-props" in alternatives_to_eval:
        water_props: tuple[w_props, w_props] = (
            w_props(P=0.2, T=90 + 273.15), # P=2 bar  -> 0.2MPa, T in K, average working temperature of primary circuit
            w_props(P=0.2, T=65 + 273.15)  # P=2 bar  -> 0.2MPa, T in K, average working temperature of secondary circuit
        )

    # Initialize result vectors
    outs_mod: list[np.ndarray[float]] = [np.zeros((len(df) - idx_start, N), dtype=float) for _ in alternatives_to_eval]
    stats = []

    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        out = outs_mod[alt_idx]
        out[0] = df.iloc[idx_start]["Tsf_out"]
        
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        # Evaluate model
        start_time_alt = time.time()
        for i in range(idx_start + 1, len(df)):
            ds = df.iloc[i]
            j = i - idx_start
            start_time = time.time()
            
            if alt_id == "standard":
                out_p, out_s = model(
                    Tp_in=ds['Thx_p_in'], 
                    Ts_in=ds['Thx_s_in'], 
                    qp=ds['qhx_p'], 
                    qs=ds['qhx_s'], 
                    Tamb=ds['Tamb'], 
                    model_params=model_params,
                    water_props=None
                )
            elif alt_id == "constant-water-props":
                out_p, out_s = model(
                    Tp_in=ds['Thx_p_in'], 
                    Ts_in=ds['Thx_s_in'], 
                    qp=ds['qhx_p'], 
                    qs=ds['qhx_s'], 
                    Tamb=ds['Tamb'], 
                    model_params=model_params,
                    water_props=water_props
                )
            else:
                raise ValueError(
                    f"Unsupported alternative {alt_id}, options are: {supported_eval_alternatives}"
                )

            out[j] = np.array([out_p, out_s])
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
            "model_parameters": model_params.__dict__,
            "sample_rate": sample_rate
        })
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.1f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")

    dfs: list[pd.DataFrame] = [
        pd.DataFrame(out, columns=out_var_ids, index=df.index[idx_start:])
        for out in outs_mod
    ]

    return dfs, stats
