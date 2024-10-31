import math
from typing import Literal
import time
import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props

from solarmed_modeling.metrics import calculate_metrics
from solarmed_modeling.utils import resample_results
from solarmed_modeling.thermal_storage.utils import Th_labels, Tc_labels
from solarmed_modeling.fsms import FsmParameters

from . import (supported_eval_alternatives,
               ModelParameters, 
               FixedModelParameters,
               SolarMED)

out_var_ids: list[str] = ["Tsf_in", "Tsf_out", "Tts_h_in", *Th_labels, *Tc_labels]

def evaluate_model(
    df: pd.DataFrame, sample_rate: int, 
    model_params: ModelParameters, 
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    fsm_params: FsmParameters = FsmParameters(),
    alternatives_to_eval: list[Literal["standard", "constant-water-props"]] = supported_eval_alternatives,
    log_iteration: bool = False, base_df: pd.DataFrame = None,
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval]), f"Invalid `alternatives_to_eval`: {alternatives_to_eval}, should be a list with any of these elements: {supported_eval_alternatives}"
    
    span = math.ceil(600 / sample_rate) # 600 s
    idx_start = np.max([span, 2]) # idx_start-1 should at least be one 
    idx_end = len(df)
        
    # Experimental (reference) outputs, used lated in performance metrics evaluation
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
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        start_time_alt = time.time()
        
        model = SolarMED(
            resolution_mode=alt_id,
            use_models=True,
            use_finite_state_machine=True,
            model_params=model_params,
            fixed_model_params=fixed_model_params,
            fsms_params=fsm_params,
            sample_time=sample_rate,
            
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

        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out_metrics, out_ref), 
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "model_parameters": model.model_dump_configuration(),
            "sample_rate": sample_rate
        })
        
        # Sync model index with reference dataframe and append to results
        df_mod.index = df.index[idx_start:i if i<idx_end-1 else idx_end]
        dfs.append(df_mod)

        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.3f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")
        
    return dfs, stats