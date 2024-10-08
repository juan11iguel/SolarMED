from typing import Literal, Any
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar
from scipy.optimize import fsolve
from scipy import signal
from solarmed_modeling.metrics import calculate_metrics

b, a = signal.butter(3, 0.005)
zi = signal.lfilter_zi(b, a)

supported_eval_alternatives: list[str] = ["standard", "no-delay", "constant-water-props"]
""" 
    - standard: Model with transport delay
    - no-delay: Model without transport delay
    - constant-water-props: Standard model but with constant water properties evaluated at some average temperature
"""

@dataclass
class ModelParameters:
    beta: float  # Gain coefficient (-)
    H: float  # Losses to the environment (W/m²)
    gamma: float  # Artificial parameters to account for flow variations within the whole solar field"
    

def calculate_total_pipe_length(q: float, n: int, sample_time:int = 10, equivalent_pipe_area:float = 7.85e-5) -> float:
    """
    Calculate the total pipe length that a flow travels in a certain number of samples.
    Args:
        q: Array of flow rates from current samples up to a minimum of n past samples. The order should be from newer to older [m³/h]
        n: Number of samples
        sample_time: Time between samples [s]
        equivalent_pipe_area: Cross-sectional area representative of the pipe (simplification to not consider each different pipe section) [m²]

    Returns:
        l: Total pipe length [m]
    """
    q_avg = sum(q[:n]) / n
    l = q_avg * sample_time * n / equivalent_pipe_area
    return l


def find_delay_samples(q: float, sample_time: int = 1, total_pipe_length: float = 5e7, equivalent_pipe_area: float = 7.85e-5, log: bool = True) -> int:
    """
    Find the number of samples that a flow takes to travel a certain distance in a pipe.
    Args:
        q: Array of flow rates. If less samples than the delay are provided, then the delay will be under-estimated.
        IMPORTANT!The order should be from oldest (first element) to newest (last element) [m³/h]
        sample_time: Time between samples [s]
        total_pipe_length: Total representative length of the pipe (simplification to not consider each different pipe section) [m]
        equivalent_pipe_area: Cross-sectional area representative of the pipe (simplification to not consider each different pipe section) [m²] #NOTE: Probably the current value is way to small for the actual equivalent pipe area since it's just the cross section of the collector tubes, while the connecting pipes are quite large and with a much larger cross-section.

    Notes:
        - The `l` was manually fitted by:
            - 1. Manually setting the `n` value in the model until visually the delay was close to the actual delay
            - 2. Calculating the `l` for each sample using that `n` value with `calculate_total_pipe_length`
            - 3. Taking the average of the `l` values

    Returns:
        n_samples: Number of samples that the flow takes to travel the distance

        References:
            [1] J. E. Normey-Rico, C. Bordons, M. Berenguel, and E. F. Camacho, “A Robust Adaptive Dead-Time Compensator with Application to A Solar Collector Field1,” IFAC Proceedings Volumes, vol. 31, no. 19, pp. 93–98, Jul. 1998, doi: 10.1016/S1474-6670(17)41134-7.

            [2] G. Ampuño, L. Roca, M. Berenguel, J. D. Gil, M. Pérez, and J. E. Normey-Rico, “Modeling and simulation of a solar field based on flat-plate collectors,” Solar Energy, vol. 170, pp. 369–378, Aug. 2018, doi: 10.1016/j.solener.2018.05.076.


    Returns:
        n: Number of samples that the flow takes to travel the distance

    """
    sum_q = 0
    n_samples = 0
    n = len(q) - 1
    while sum_q < total_pipe_length and n >= 0:
        sum_q += q[n] * sample_time / equivalent_pipe_area
        n -= 1
        n_samples += 1

    if n < 0 and log:
        logger.warning(
            f'Flow rate array is not long enough to estimate delay, or the total pipe length is too big. Returning {n_samples} samples == len(q) == {n_samples * sample_time} s')

    return n_samples


def solar_field_inverse_model2(
        Tout_ant: float, Tin: float | np.ndarray[float], Tout: float,
        I: float, Tamb: float,
        beta: float = 0.0975, gamma: float = 1.0, H: float = 2.2,
        sample_time=1,
        q_ant: np.ndarray[float] = None,
        nt=1, npar=7 * 5, ns=2, Lt=1.15 * 20, Acs: float = 7.85e-5,
) -> float:
    """
    Solar field inverse model. New approach using fsolve to solve `solar_field_model`.

    Args:
        Tout_ant: Solar field outlet temperature at previous time step [ºC]
        Tin: Solar field inlet temperature [ºC]
        q: Solar field volumetric flow rate [m³/h]
        I: Solar direct irradiance [W/m²]
        Tamb: Ambient temperature [ºC]
        q_ant (optional): Solar field volumetric flow rate at previous time step [m³/h], for dynamic estimation of delay between q and Tout. Not yet implemented.
        beta: Irradiance model parameter [m]
        H: Thermal losses coefficient [J/sºC]
        nt: Number of tubes in parallel per collector
        npar: Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments
        ns: Number of loops in series
        Lt: Solar field. Collector tube length [m
        Acs (float, optional): Flat plate collector tube cross-section area [m²]. Defaults to 7.85e-5
        sample_time:

    Returns:
        Tout: (float): Solar field outlet temperature [ºC]

    """

    qmin = 5
    qmax = 10  # TODO: Check

    # If there is no temperature difference, the flow rate cannot be estimated, it could be any
    if Tout - Tin.take(-1) < 0.5:
        return 0

    wrapped_model = lambda q: solar_field_model(Tout_ant=Tout_ant, Tin=Tin, q=q, I=I, Tamb=Tamb, beta=beta, gamma=gamma,
                                                H=H, sample_time=sample_time, consider_transport_delay=True, nt=nt,
                                                npar=npar, ns=ns, Lt=Lt, Acs=Acs) - Tout

    # Solve for q
    initial_guess = q_ant.take(-1)
    q = fsolve(wrapped_model, initial_guess)

    return q


def solar_field_inverse_model(
        Tout_ant: float, Tin: float | np.ndarray[float], Tout: float,
        I: float, Tamb: float,
        beta: float = 0.0975, gamma: float = 1.0, H: float = 2.2,
        sample_time=1, consider_transport_delay: bool = False,
        filter_signal: bool = True,
        q_ant: np.ndarray[float] = None,
        nt=1, npar=7 * 5, ns=2, Lt=1.15 * 20, Acs: float = 7.85e-5,
        f=0.8
) -> float:
    """

    Args:
        Tout_ant: Solar field outlet temperature at previous time step [ºC]
        Tin: Solar field inlet temperature [ºC]
        q: Solar field volumetric flow rate [m³/h]
        I: Solar direct irradiance [W/m²]
        Tamb: Ambient temperature [ºC]
        q_ant (optional): Solar field volumetric flow rate at previous time step [m³/h], for dynamic estimation of delay between q and Tout. Not yet implemented.
        beta: Irradiance model parameter [m]
        H: Thermal losses coefficient [J/sºC]
        nt: Number of tubes in parallel per collector
        npar: Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments
        ns: Number of loops in series
        Lt: Solar field. Collector tube length [m
        Acs (float, optional): Flat plate collector tube cross-section area [m²]. Defaults to 7.85e-5
        sample_time:

    Returns:
        Tout: (float): Solar field outlet temperature [ºC]

    """

    qmin = 5
    qmax = 10  # TODO: Check

    Leq = ns * Lt
    cf = npar * nt

    Tavg = (Tin.take(-1) + Tout) / 2  # ºC

    w_props_avg = w_props(P=0.16, T=Tavg + 273.15)  # P=1 bar  -> 0.1MPa, T=Tin C,
    rho = w_props_avg.rho  # [kg/m³]
    cp = w_props_avg.cp * 1e3  # [kJ/kg·K] -> [J/kg·K]

    K1 = beta / (rho * cp * Acs)  # m / (kg/m3 * J/kg·K) = K·m²/J
    K2 = H / (Leq * Acs * rho * cp)  # J/sK / (m · m² · kg/m3 · J/kg·K) = 1/s
    K3 = gamma / (Leq * Acs * cf) * (1 / 3600)  # 1/(m · m² · -) * (1 / 3600s) = h/(3600·m³·s)

    # If there is no temperature difference, the flow rate cannot be estimated
    if Tout - Tin.take(-1) < 0.5:
        return 0

    if consider_transport_delay:
        n = find_delay_samples(q_ant, sample_time=sample_time)
        # n = 500 / sample_time # Temporary value
    else:
        n = 1

    q = 1 / (K3 * (Tout - Tin.take(-n))) * (((Tout_ant - Tout) / sample_time) + (I * K1) + ((Tamb - Tavg) * K2))

    q = float(q)  # Chapuza

    # try:
    q = np.max([q, qmin])
    q = np.min([q, qmax])
    # except Exception as e:
    #     logger.error(f'Error: {e}, q: {q}, qmin: {qmin}, qmax: {qmax}, deltaT: {deltaT}, Tout_ant: {Tout_ant}, Tout: {Tout}, I: {I}, Tin: {Tin}, Tamb: {Tamb}')
    #     q = qmin

    if q_ant is not None and filter_signal:
        # q = signal.filtfilt(b, a, np.append(q_ant, q))[-1]
        # q, _ = signal.lfilter(b, a, np.append(q_ant, q), zi=zi*q_ant[0])
        # q = q.take(-1)
        # Weighted average of the last n samples
        q = f * q + (1 - f) * q_ant[-n:].mean()

    return q


def solar_field_model(
    Tout_ant: float,
    Tin: float | np.ndarray,
    q: float | np.ndarray,
    I: float,
    Tamb: float,
    beta: float = 0.0975,
    gamma: float = 1.0,
    H: float = 2.2,
    sample_time=1,
    consider_transport_delay: bool = False,
    water_props: w_props = None,
    nt=1,
    npar=7 * 5,
    ns=2,
    Lt=1.15 * 20,
    Acs: float = 7.85e-5,
    log: bool = False,
) -> float:
    """

    Args:
        Tout_ant: Solar field outlet temperature at previous time step [ºC]
        Tin: Solar field inlet temperature [ºC]
        q: Solar field volumetric flow rate [m³/h]
        I: Solar direct irradiance [W/m²]
        Tamb: Ambient temperature [ºC]
        q_ant (optional): Solar field volumetric flow rate at previous time step [m³/h], for dynamic estimation of delay between q and Tout.
        beta: Irradiance model parameter [m]
        H: Thermal losses coefficient [J/sºC]
        nt: Number of tubes in parallel per collector
        np: Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments
        ns: Number of loops in series
        Lt: Solar field. Collector tube length [m]
        Acs (float, optional): Flat plate collector tube cross-section area [m²]. Defaults to 7.85e-5
        sample_time:

    Returns:
        Tout: (float): Solar field outlet temperature [ºC]

    """

    Tin = np.array(Tin)
    q = np.array(q)

    Leq = ns * Lt
    cf = npar * nt

    if Tout_ant > 120:
        # Above 110ºC, the model is not valid
        return 9999

    Tavg = (Tin.take(-1) + Tout_ant) / 2  # ºC
    if water_props is None:
        water_props = w_props(P=0.2, T=Tavg + 273.15)  # P=2 bar  -> 0.1MPa, T=Tin C,
    rho = water_props.rho  # [kg/m³]
    cp = water_props.cp * 1e3  # [kJ/kg·K] -> [J/kg·K]

    K1 = beta / (rho * cp * Acs)  # m / (kg/m3 * J/kg·K) = K·m²/J
    K2 = H / (Leq * Acs * rho * cp)  # J/sK / (m · m² · kg/m3 · J/kg·K) = 1/s
    K3 = (
        gamma / (Leq * Acs * cf) * (1 / 3600)
    )  # 1/(m · m² · -) * (1 / 3600s) = h/(3600·m³·s)

    # deltaTout_m = m [m3/h * kg/m3*1h/3600s] * deltaT [K] * K3 [1/(m * * m2 * kg/m3 * -)]

    if q.take(-1) == 0:
        # Just thermal losses
        deltaTout = -K2 * (Tavg - Tamb)

        """
            A more thorough approach would include radiation and convection losses:
            T = T - (H * (T - Tamb) + eta * (T⁴-T⁴)) * sample_time
        """

    else:
        if consider_transport_delay:
            n = find_delay_samples(q, sample_time=sample_time, log=log)
            # n = 500 / sample_time # Temporary value
        else:
            n = 1

        deltaTout = (
            K1 * I - K3 * q.take(-1) * (Tout_ant - Tin.take(-n)) - K2 * (Tavg - Tamb)
        )

        # If the model predicts a lower temperature than the inlet, return the inlet temperature
        # i.e. the solar field can't cool the fluid below the inlet temperature
        # if Tout_ant + deltaTout * sample_time <= Tin.take(-1):
        #     deltaTout = (Tin.take(-1) - Tout_ant) / sample_time
        #     logger.warning(f'Solar field cant cooldown below inlet temperature. New {deltaTout:.4f}ºC')

    out = Tout_ant + deltaTout * sample_time

    return out

def evaluate_model(
    df: pd.DataFrame, sample_rate: int, model_params: ModelParameters,
    alternatives_to_eval: list[Literal["standard", "no-delay", "constant-water-props"]] = supported_eval_alternatives,
    log_iteration: bool = False,
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    """
    Evaluate the solar field model using different alternatives and calculate performance metrics.

    Args:
        df: DataFrame containing the input data for the model.
        sample_rate: Sampling rate in seconds.
        model_params: ModelParameters object containing the model parameters.
        alternatives_to_eval: List of alternatives to evaluate. Supported alternatives are "standard", "no-delay", and "constant-water-props".
        log_iteration: Boolean flag to log each iteration.

    Raises:
        ValueError: If an unsupported alternative is provided in alternatives_to_eval.

    Returns:
        tuple: A tuple containing a list of DataFrames with the model outputs and a list of dictionaries with the performance metrics.
    """
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval])

    idx_start = int(round(600 / sample_rate, 0))  # 600 s
    span = idx_start

    if span > idx_start:
        logger.warning(
            f"Span {span} cant be greater than idx_start {idx_start}. Setting span to idx_start"
        )
        span = idx_start

    # Experimental (reference) outputs, used lated in performance metrics evaluation
    out_ref = df.iloc[idx_start:]["Tsf_out"].values

    # Initialize particular variables for earch alternative that requires it
    water_props = None
    if "constant-water-props" in alternatives_to_eval:
        water_props: w_props = w_props(
            P=0.2, T=90 + 273.15
        )  # P=2 bar  -> 0.2MPa, T in K, average working solar field temperature
        
    if "no-delay" in alternatives_to_eval:
        logger.warning("The 'no-delay' alternative has hardcoded parameters. Make sure to adjust them if needed.")

    # Initialize result vectors
    # q_sf_mod   = np.zeros(len(df), dtype=float)
    outs_mod: list[np.ndarray[float]] = [np.zeros(len(df) - idx_start, dtype=float) for _ in alternatives_to_eval]
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
            
            if alt_id == "no-delay":
                out[j] = solar_field_model(
                    Tin=ds["Tsf_in"],
                    q=ds["qsf"],
                    I=ds["I"],
                    Tamb=ds["Tamb"],
                    Tout_ant=out[j - 1],
                    sample_time=sample_rate,
                    consider_transport_delay=False,
                    beta=1.1578e-2,
                    H=3.1260,
                    gamma=0.0471,
                )
            elif alt_id == "standard":
                out[j] = solar_field_model(
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
                    beta=model_params.beta,
                    H=model_params.H,
                    gamma=model_params.gamma,
                )
            elif alt_id == "constant-water-props":
                out[j] = solar_field_model(
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
                    beta=model_params.beta,
                    H=model_params.H,
                    gamma=model_params.gamma,
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
        
        
        elapsed_time = time.time() - start_time_alt
        
        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out, out_ref), 
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "model_parameters": model_params.__dict__,
            "sample_rate": sample_rate
        })
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.1f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")

        # l[i] = calculate_total_pipe_length(q=df.iloc[i:i-idx_start:-1]['qsf'].values, n=60, sample_time=10, equivalent_pipe_area=7.85e-5)

        # df['Tsf_l2_pred'] = Tsf_out_mod

        # Estimate pipe length
        # l[l>1].mean() # 6e7

        dfs: list[pd.DataFrame] = [
            pd.DataFrame(out, columns=["Tsf_out"], index=df.index[idx_start:])
            for out in outs_mod
        ]

    return dfs, stats
