import math
import time
import numpy as np
import pandas as pd
from loguru import logger

from solarmed_modeling.metrics import calculate_metrics
from solarmed_modeling.utils import resample_results
from solarmed_modeling.thermal_storage.utils import Th_labels, Tc_labels

from solarmed_modeling.solar_med import (
    supported_eval_alternatives,
    SupportedAlternativesLiteral,
    ModelParameters, 
    FixedModelParameters,
    FsmParameters,
    EnvironmentParameters,
    MedFsmParams,
    SolarMED
)

out_var_ids: list[str] = ["Tsf_in", "Tsf_out", "Thx_s_out", *Th_labels, *Tc_labels, "qmed_d", "qmed_c", "Tmed_s_out", "Pth_sf", "Pth_ts_src", "Pth_ts_dis", "Ets_h", "Ets_c"]

def evaluate_model(
    df: pd.DataFrame, 
    sample_rate: int, 
    model_params: ModelParameters, 
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    fsm_params: FsmParameters = FsmParameters(
        med=MedFsmParams(
            # Effectively disable waiting times and cooldowns
            vacuum_duration_time=0, 
            off_cooldown_time=0,
            active_cooldown_time=0,
            brine_emptying_time=0,
            startup_duration_time=0
        )
    ),
    env_params: EnvironmentParameters = EnvironmentParameters(),
    alternatives_to_eval: list[SupportedAlternativesLiteral] = "constant-water-props",
    log_iteration: bool = False, 
    base_df: pd.DataFrame = None,
    horizon_time: int | None = None,
    **kwargs,
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    """_summary_
    
    Args:
        horizon_time: int | None. If not None, feedback experimental data every `horizon_time` seconds.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Invalid `alternatives_to_eval`: {alternatives_to_eval}, should be a list with any of these elements: {supported_eval_alternatives}"
    
    span = math.ceil(600 / sample_rate) # 600 s
    idx_start = np.max([span, 2]) # idx_start-1 should at least be one 
    idx_end = len(df)
    
    if horizon_time is not None:
        horizon_size = horizon_time // sample_rate
    else:
        horizon_size = idx_end - idx_start
        
    # Experimental (reference) outputs, used later in performance metrics evaluation
    if base_df is None:
        idx_start_ref = idx_start
        ref_df = df
    else:
        if base_df.index.freq.n > df.index.freq.n:
            raise ValueError(f"Base dataframe can't have a lower sample rate than the input dataframe: base df rate ({base_df.index.freq.n}) < evaluting df rate({df.index.freq.n})")
        
        idx_start_ref = int(round(600 / base_df.index.freq.n, 0))
        ref_df = base_df
    out_ref = ref_df.iloc[idx_start_ref:][out_var_ids].values
        
    # Initialize result vectors
    stats = []
    dfs: list[pd.DataFrame] = []
    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        # Initial values
        # Initialize model
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s. Prediction horizon = {horizon_time} s")
        start_time_alt = time.time()
        
        model = SolarMED(
            resolution_mode=alt_id,
            use_models=True,
            use_finite_state_machine=True,
            model_params=model_params,
            fixed_model_params=fixed_model_params,
            fsms_params=fsm_params,
            env_params=env_params,
            sample_time=sample_rate,
            on_limits_violation_policy="clip",
            
            # Initial states
            ## Thermal storage
            Tts_h=[df['Tts_h_t'].iloc[idx_start], df['Tts_h_m'].iloc[idx_start], df['Tts_h_b'].iloc[idx_start]], 
            Tts_c=[df['Tts_c_t'].iloc[idx_start], df['Tts_c_m'].iloc[idx_start], df['Tts_c_b'].iloc[idx_start]],
            
            ## Solar field
            Tsf_in_ant=df['Tsf_in'].iloc[idx_start-span:idx_start].values,
            qsf_ant=df['qsf'].iloc[idx_start-span:idx_start].values,
        )
        
        df_mod = model.to_dataframe()
        
        # Evaluate model
        for i in range(idx_start + 1, idx_end):
            ds = df.iloc[i]
            start_time = time.time()
            
            # print(f"{i=} | {i-idx_start=} | {horizon_size=} | {(i-idx_start) % horizon_size=}")

            # Feedback experimental outputs every prediction horizon
            if (i-idx_start) % horizon_size == 0:
                # print(f"{i=} | {i-idx_start=} | {horizon_size=} | Feedbacking experimental data")
                ds_exp = df.iloc[i-1]
                
                model_instance = model.dump_instance()
                model_instance.update(dict(
                    # Initial states
                    ## FSM states
                    # fsms_internal_states=model.fsms_internal_states, # From previous step
                    # med_state=model.med_state,
                    # sf_ts_state=model.sf_ts_state,
                    ## Thermal storage
                    Tts_h=[
                        ds_exp["Tts_h_t"],
                        ds_exp["Tts_h_m"],
                        ds_exp["Tts_h_b"],
                    ],
                    Tts_c=[
                        ds_exp["Tts_c_t"],
                        ds_exp["Tts_c_m"],
                        ds_exp["Tts_c_b"],
                    ],
                    ## Solar field
                    Tsf_in_ant=df["Tsf_in"].iloc[i-1 - span : i-1].values,
                    qsf_ant=df["qsf"].iloc[i-1 - span : i-1].values,
                ))
                model = SolarMED(**model_instance)
            
            model.step(
                # Decision variables
                ## MED
                qmed_s=ds['qmed_s'],
                qmed_f=ds['qmed_f'],
                Tmed_s_in=ds['Tmed_s_in'],
                Tmed_c_out=ds['Tmed_c_out'],
                ## Thermal storage
                qts_src=ds['qhx_s'],
                ## Solar field
                qsf=ds['qsf'],
                
                med_vacuum_state=2,
                        
                # Environment variables
                Tmed_c_in=ds['Tmed_c_in'],
                Tamb=ds['Tamb'],
                I=ds['I'],
            )
            df_mod = model.to_dataframe(df_mod)
            
            elapsed_time = time.time() - start_time

            # if log_iteration:
            #     logger.info(
            #         f"[{alt_id}] Iteration {i} / {len(df)}. Elapsed time: {elapsed_time:.5f} s. Error: {abs(out_mod[j]-out_ref[j]):.2f}"
            #     )
                
        elapsed_time = time.time() - start_time_alt
        
        out_mod = df_mod[out_var_ids].values
        if base_df is None:
            out_metrics = out_mod
        else:
            # Resample out to base_df sample rate using ffill
            out_metrics = resample_results(out_mod, new_index=base_df.index[idx_start_ref:], 
                                           current_index=df.index[idx_start:], reshape=False)
            
        for var_id in ["Pth_ts_src", "Pth_ts_dis"]:
            if var_id in out_var_ids:
                var_idx = out_var_ids.index(var_id)
                
                # TODO: This should not be needed! Why is qts_src kept active after sf shutdown? Why does it not affect experimental data?
                out_metrics[:, var_idx] = np.maximum(out_metrics[:, var_idx], 0)
                # Also filter low experimental values (< 10 kW) since they are not relevant and skew metrics (e.g., MAPE) 
                out_ref[out_ref[:, var_idx] < 10.0, var_idx] = 0.0

        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out_metrics, out_ref),
            "metrics_per_variable": {
                out_var_id: calculate_metrics(out_metrics[:,out_idx], out_ref[:,out_idx])
                for out_idx, out_var_id in enumerate(out_var_ids)    
            },
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "model_parameters": model.model_dump_configuration(),
            "sample_rate": sample_rate,
            "horizon_time": horizon_time,
        })
        
        # Sync model index with reference dataframe and append to results
        df_mod.index = df.index[idx_start:i if i<idx_end-1 else idx_end]
        dfs.append(df_mod)

        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.3f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")
        
    return dfs, stats