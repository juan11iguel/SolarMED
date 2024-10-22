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

from . import supported_eval_alternatives, ModelParameters, FixedModelParameters
from . import heat_generation_and_storage_subproblem as model

out_var_ids: list[str] = ["Tsf_in", "Tsf_out", "Tts_h_in", *Th_labels, *Tc_labels]

def evaluate_model(
    df: pd.DataFrame, sample_rate: int, 
    model_params: ModelParameters, fixed_model_params: FixedModelParameters = FixedModelParameters(),
    alternatives_to_eval: list[Literal["standard", ]] = supported_eval_alternatives,
    log_iteration: bool = False, base_df: pd.DataFrame = None,
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval])
    
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
    out_ref = ref_df.iloc[idx_start_ref:][out_var_ids].values
    
    # Initialize particular variables for earch alternative that requires it
    # water_props = None
    # if "standard" in alternatives_to_eval:
    water_props: tuple[w_props, w_props] = (
        w_props(P=0.2, T=90 + 273.15), # P=2 bar  -> 0.2MPa, T in K, average working temperature of hot tank
        w_props(P=0.2, T=65 + 273.15)  # P=2 bar  -> 0.2MPa, T in K, average working temperature of cold tank
    )
    
    # Initialize result vectors
    outs_mod: list[np.ndarray[float]] = [np.zeros((len(df) - idx_start, len(out_var_ids)), dtype=float) for _ in alternatives_to_eval]
    stats = []
    idx_span: list[int, int] = [None, None]
    
    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        out_mod = outs_mod[alt_idx]
        # Initial values
        Tsf_in_ant = df.iloc[0:idx_start-1]["Tsf_in"].values
        out_mod[0,0] = df.iloc[idx_start]["Tsf_in"]
        out_mod[0,1] = df.iloc[idx_start]["Tsf_out"]
        out_mod[0,3] = df.iloc[idx_start]["Tts_h_in"]
        idx_span[0] = 3
        idx_span[1] = idx_span[0] + len(Th_labels)
        out_mod[0, idx_span[0]:idx_span[1]] = np.array([df[label].values[idx_start] for label in Th_labels])
        idx_span[0] = idx_span[1]
        idx_span[1] = idx_span[0] + len(Tc_labels)
        out_mod[0, idx_span[0]:idx_span[1]] = np.array([df[label].values[idx_start] for label in Tc_labels])

        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        
        # Evaluate model
        start_time_alt = time.time()
        for i in range(idx_start + 1, len(df)):
            ds = df.iloc[i]
            j = i - idx_start
            start_time = time.time()
            
            Tsf_in_ant = np.roll(Tsf_in_ant, -1)
            Tsf_in_ant[-1] = out_mod[j-1, 0]
            Tsf_out_ant= out_mod[j-1, 1]
            idx_span[0] = 3
            idx_span[1] = idx_span[0] + len(Th_labels)
            Tts_h= out_mod[j-1, idx_span[0]:idx_span[1]]
            idx_span[0] = idx_span[1]
            idx_span[1] = idx_span[0] + len(Tc_labels)
            Tts_c= out_mod[j-1, idx_span[0]:idx_span[1]]
            
            if alt_id == "standard":
                Tsf_in, Tsf_out, Tts_t_in, Tts_h, Tts_c = model(
                    # Solar field
                    qsf= df.iloc[i-span: i]["qsf"].values,
                    Tsf_in_ant = Tsf_in_ant,
                    Tsf_out_ant= Tsf_out_ant,
                    
                    # Thermal storage
                    qts_src= ds["qts_src"], qts_dis= ds["qts_dis"],
                    Tts_b_in= ds["Tts_c_in"], 
                    Tts_h= Tts_h, 
                    Tts_c= Tts_c, 
                    
                    # Environment
                    Tamb=ds["Tamb"], I=ds["I"],  
                    
                    # Parameters
                    model_params=model_params,
                    fixed_model_params=fixed_model_params,
                    sample_time = sample_rate,
                    water_props = water_props,
                    problem_type = "1p2x",
                    solver = "scipy",
                    solver_method = "dogbox"
                )
            # elif alt_id == "alternative2":
            else:
                raise ValueError(
                    f"Unsupported alternative {alt_id}, options are: {supported_eval_alternatives}"
                )
                
            out_mod[j, 0] = Tsf_in
            out_mod[j, 1] = Tsf_out
            out_mod[j, 2] = Tts_t_in
            idx_span[0] = 3
            idx_span[1] = idx_span[0] + len(Th_labels)
            out_mod[j, idx_span[0]:idx_span[1]] = Tts_h
            idx_span[0] = idx_span[1]
            idx_span[1] = idx_span[0] + len(Tc_labels)
            out_mod[j, idx_span[0]:idx_span[1]] = Tts_c
            
            elapsed_time = time.time() - start_time

            if log_iteration:
                logger.info(
                    f"[{alt_id}] Iteration {i} / {len(df)}. Elapsed time: {elapsed_time:.5f} s. Error: {abs(out_mod[j]-out_ref[j]):.2f}"
                )
                
        elapsed_time = time.time() - start_time_alt
        
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
            "model_parameters": model_params.__dict__,
            "sample_rate": sample_rate
        })

        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.3f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")
        
        dfs: list[pd.DataFrame] = []
        for out_mod in outs_mod:
            dfs.append(
                pd.DataFrame(out_mod, columns=out_var_ids, index=df.index[idx_start:])
            )
            df_mod = dfs[-1]
            df_mod["Thx_p_in"]  = df_mod["Tsf_out"]
            df_mod["Thx_p_out"] = df_mod["Tsf_in"]
            df_mod["Thx_s_in"]  = df_mod["Tts_c_b"]
            df_mod["Thx_s_out"] = df_mod["Tts_h_in"]
            # When secondary flow is null, equating HEX temperatures to solar field is more accurate
            df_mod.loc[df["qts_src"] < 0.1, "Thx_s_in"] = df_mod.loc[df["qts_src"] < 0.1, "Thx_p_in"]

    return dfs, stats