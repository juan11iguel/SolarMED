from typing import Literal, Any
from dataclasses import dataclass
import time
import math
import numpy as np
from loguru import logger
from iapws import IAPWS97 as w_props
import pandas as pd
from solarmed_modeling.metrics import calculate_metrics
from solarmed_modeling.utils.benchmark import resample_results

supported_eval_alternatives: list[str] = ["standard", "constant-water-props"]
""" 
    - standard: Counter-flow heat exchanger steady state model based on the effectiveness-NTU method
    - constant-water-props: Standard model but with constant water properties evaluated at some average temperature
"""

@dataclass
class ModelParameters:
    UA: float  # Heat transfer coefficient (W/K)
    H: float  # Losses to the environment (W/m²)
   

def estimate_flow_secondary(Tp_in: float, Ts_in: float, qp: float, Tp_out: float, Ts_out: float) -> float:
    if np.abs(Ts_out - Ts_in) < 1:
        return np.nan

    try:
        w_props_Tp_in = w_props(P=0.16, T=(Tp_in + Tp_out) / 2 + 273.15)
        w_props_Ts_in = w_props(P=0.16, T=(Ts_in + Ts_out) / 2 + 273.15)
    except Exception as e:
        logger.warning(f'Invalid temperature input values: Tp_in={Tp_in}, Ts_in={Ts_in}, Tp_out={Tp_out}, Ts_out={Ts_out} (ºC), returning NaN')
        return np.nan

    cp_Tp_in = w_props_Tp_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    cp_Ts_in = w_props_Ts_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]

    qs = qp * (cp_Tp_in * (Tp_in - Tp_out)) / (cp_Ts_in * (Ts_out - Ts_in))

    return np.max([qs, 0])


def heat_exchanger_model(
    Tp_in: float, Ts_in: float, qp: float, qs: float, Tamb: float, 
    UA: float = 13536.596, H: float = 0, log: bool = True, 
    hex_type: Literal['counter_flow',] = 'counter_flow',
    water_props: tuple[w_props, w_props] = None, return_epsilon: bool = False, 
    epsilon: float = None
) -> tuple[float, float] | tuple[float, float, float]:

    """Counter-flow heat exchanger steady state model.

    Based on the effectiveness-NTU method [2] - Chapter Heat exchangers 11-5:

    ΔTa: Temperature difference between primary circuit inlet and secondary circuit outlet
    ΔTb: Temperature difference between primary circuit outlet and secondary circuit inlet

    `p` references the primary circuit, usually the hot side, unless the heat exchanger is inverted.
    `s` references the secondary circuit, usually the cold side, unless the heat exchanger is inverted.
    `Qdot` is the heat transfer rate
    `C` is the capacity ratio, defined as the ratio of the heat capacities of the two fluids, C = Cmin/Cmax

    To avoid confussion, whichever the heat exchange direction is, the hotter side will be referenced as `h` and the 
    colder side as `c`.

   T|  Tp,in
    |   ---->
    |    .   \---->         Tp,out
    |   ΔTa       \----------->
    |    .                ΔTb
    |    <---              .
    |       \<----------------<
    |    Ts,out               Ts,in
    |_______________________________
                                   z

    Limitations (from [2]):
    - It has been assumed that the rate of change for the temperature of both fluids is proportional to the temperature 
    difference; this assumption is valid for fluids with a constant specific heat, which is a good description of fluids 
    changing temperature over a relatively small range. However, if the specific heat changes, the LMTD approach will no 
    longer be accurate.
    - A particular case for the LMTD are condensers and reboilers, where the latent heat associated to phase change is a
    special case of the hypothesis. For a condenser, the hot fluid inlet temperature is then equivalent to the hot fluid 
    exit temperature.
    - It has also been assumed that the heat transfer coefficient (U) is constant, and not a function of temperature. 
    If this is not the case, the LMTD approach will again be less valid
    - The LMTD is a steady-state concept, and cannot be used in dynamic analyses. In particular, if the LMTD were to be 
    applied on a transient in which, for a brief time, the temperature difference had different signs on the two sides 
    of the exchanger, the argument to the logarithm function would be negative, which is not allowable.
    - No phase change during heat transfer
    - Changes in kinetic energy and potential energy are neglected
    
    [1] W. M. Kays and A. L. London, Compact heat exchangers: A summary of basic heat transfer and flow friction design 
    data. McGraw-Hill, 1958. [Online]. Available: https://books.google.com.br/books?id=-tpSAAAAMAAJ
    [2] Y. A. Çengel and A. J. Ghajar, Heat and mass transfer: fundamentals & applications, Fifth edition. 
    New York, NY: McGraw Hill Education, 2015.

    Args:
        Tp_in (float): Primary circuit inlet temperature [C]
        Ts_in (float): Secondary circuit inlet temperature [C]
        qp (float): Primary circuit volumetric flow rate [m^3/h]
        qs (float): Secondary circuit volumetric flow rate [m^3/h]
        UA (float, optional): Heat transfer coefficient multiplied by the exchange surface area [W·ºC^-1]. 
        Defaults to 28000.

    Returns:
        Tp_out: Primary circuit outlet temperature [C]
        Ts_out: Secondary circuit outlet temperature [C]
    """

    assert hex_type == 'counter_flow', 'Only counter-flow heat exchangers are supported'
    inverted_hex: bool = False


    if water_props is not None:
        w_props_Tp_in, w_props_Ts_in = water_props
    else:
        try:
            w_props_Tp_in = w_props(P=0.16, T=Tp_in + 273.15)
            w_props_Ts_in = w_props(P=0.16, T=Ts_in + 273.15)
        except Exception as e:
            logger.info(f'Invalid temperature input values: Tp_in={Tp_in}, Ts_in={Ts_in} (ºC)')
            raise e

    cp_Tp_in = w_props_Tp_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    cp_Ts_in = w_props_Ts_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]

    mp = qp / 3600 * w_props_Tp_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s
    ms = qs / 3600 * w_props_Ts_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s

    Cp = mp * cp_Tp_in
    Cs = ms * cp_Ts_in
    Cmin = np.min([Cp, Cs])
    Cmax = np.max([Cp, Cs])

    if qp < 0.1:
        Tp_out = Tp_in - H * (Tp_in - Tamb)  # Just losses to the environment
        if qs < 0.1:
            Ts_out = Ts_in - H * (Ts_in - Tamb)  # Just losses to the environment
        else:
            Ts_out = Ts_in

        if return_epsilon:
            return Tp_out, Ts_out, 0
        else:
            return Tp_out, Ts_out

    if qs < 0.1:
        Ts_out = Ts_in - H * (Ts_in - Tamb)  # Just losses to the environment
        if qp < 0.1:
            Tp_out = Tp_in - H * (Tp_in - Tamb)  # Just losses to the environment
        else:
            Tp_out = Tp_in

        if return_epsilon:
            return Tp_out, Ts_out, 0
        else:
            return Tp_out, Ts_out

    if Tp_in < Ts_in:
        inverted_hex = True

        if log: 
            logger.warning('Inverted operation in heat exchanger')

        Ch = Cs
        Cc = Cp
        Th_in = Ts_in
        Tc_in = Tp_in

    else:
        Ch = Cp
        Cc = Cs
        Th_in = Tp_in
        Tc_in = Ts_in

    # Calculate the effectiveness
    if epsilon is None:
        C = Cmin / Cmax
        NTU = UA / Cmin
        epsilon = (1 - math.e ** (-NTU * (1 - C))) / (1 - C * math.e ** (-NTU * (1 - C)))

    # Calculate the heat transfer rate
    Qdot_max = Cmin * (Th_in - Tc_in)

    # Calculate the outlet temperatures
    # Assume that the losses to the environment are dominated from the inlet hot side temperature (maximun temperature difference)
    Th_out = Th_in - (Qdot_max * epsilon) / (Ch)  # - H * (Th_in - Tamb)
    # Tc,out = Tc_in + (Qdot*epsilon) / (Cc) - H * (Tc_in + Qdot,max/Cc - Tamb)
    # Assume that the maximum heat transfer rate is achived to the cold side for the thermal losses (maximun temperature difference)
    Tc_out = Tc_in + (Qdot_max * epsilon) / (Cc)  # - H * (Tc_in + Cmin*(Th_in-Tc_in)/Cc - Tamb)

    if inverted_hex:
        Tp_out = Tc_out
        Ts_out = Th_out

    else:
        Tp_out = Th_out
        Ts_out = Tc_out

    if inverted_hex and Tp_out < Tp_in:
        raise ValueError(f'If heat exchanger is inverted, we should be obtaining hotter temperatures at the outlet of the primary, not Tp,in {Tp_in:.2f} > Tp,out {Tp_out:.2f}')

    if return_epsilon:
        return Tp_out, Ts_out, epsilon
    else:
        return Tp_out, Ts_out


def calculate_heat_transfer_effectiveness(Tp_in: float, Tp_out: float, Ts_in: float, Ts_out: float, qp: float,
                                          qs: float) -> float:
    """
    Equation (11–33) from [1]

    [1] Y. A. Çengel and A. J. Ghajar, Heat and mass transfer: fundamentals & applications, Fifth edition. New York, NY: McGraw Hill Education, 2015.

    Returns:
        eta: Heat transfer effectiveness

    """

    w_props_Tp_in = w_props(P=0.16, T=Tp_in + 273.15)
    w_props_Ts_in = w_props(P=0.16, T=Ts_in + 273.15)

    cp_Tp_in = w_props_Tp_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    cp_Ts_in = w_props_Ts_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]

    mp = qp / 3600 * w_props_Tp_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s
    ms = qs / 3600 * w_props_Ts_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s

    Cmin = np.min([mp * cp_Tp_in, ms * cp_Ts_in])

    # It could be calculated with any, just to disregard specific heat capacity
    if abs(Cmin - mp * cp_Tp_in) < 1e-6:  # Primary circuit is the one with the lowest heat capacity
        epsilon = (Tp_in - Tp_out) / (Tp_in - Ts_in)
    else:  # Secondary circuit is the one with the lowest heat capacity
        epsilon = (Ts_out - Ts_in) / (Tp_in - Ts_in)

    return epsilon


def evaluate_model(
    df: pd.DataFrame, sample_rate: int, model_params: ModelParameters,
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
    out_var_ids = ["Thx_p_out", "Thx_s_out"]
    N = len(out_var_ids)

    # Experimental (reference) outputs, used lated in performance metrics evaluation
    # out_ref = np.concatenate(df.iloc[idx_start:]["Thx_p_out"].values, df.iloc[idx_start:]["Thx_s_out"].values)
    if base_df is None:
        out_ref = np.concatenate([df[out_var_ids].values[idx_start:]], axis=1)
    else:
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
                out_p, out_s = heat_exchanger_model(
                    Tp_in=ds['Thx_p_in'], 
                    Ts_in=ds['Thx_s_in'], 
                    qp=ds['qhx_p'], 
                    qs=ds['qhx_s'], 
                    Tamb=ds['Tamb'], 
                    UA=model_params.UA, H=model_params.H,
                    water_props=None
                )
            elif alt_id == "constant-water-props":
                out_p, out_s = heat_exchanger_model(
                    Tp_in=ds['Thx_p_in'], 
                    Ts_in=ds['Thx_s_in'], 
                    qp=ds['qhx_p'], 
                    qs=ds['qhx_s'], 
                    Tamb=ds['Tamb'], 
                    UA=model_params.UA, H=model_params.H,
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
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.1f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")

    dfs: list[pd.DataFrame] = [
        pd.DataFrame(out, columns=out_var_ids, index=df.index[idx_start:])
        for out in outs_mod
    ]

    return dfs, stats
