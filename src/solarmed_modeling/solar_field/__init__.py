from dataclasses import dataclass
import numpy as np
from loguru import logger
from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar
import scipy
import warnings

# TODO: Refactor models to make use of ModelParameter dataclass directly
# TODO: Refactor models to replace fixed parameters with FixedModelParameters dataclass
# TODO: Deprecate older models

b, a = scipy.signal.butter(3, 0.005)
zi = scipy.signal.lfilter_zi(b, a)

supported_eval_alternatives: list[str] = ["standard", "no-delay", "constant-water-props"]
""" 
    - standard: Model with transport delay
    - no-delay: Model without transport delay
    - constant-water-props: Standard model but with constant water properties evaluated at some average temperature
"""

@dataclass
class ModelParameters:
    beta: float = 4.36396e-02  # Gain coefficient (-)
    H: float = 13.676448551722462  # Losses to the environment (W/m²)
    gamma: float = 0.1  # Artificial parameters to account for flow variations within the whole solar field"

@dataclass
class FixedModelParameters:
    nt: int = 1 # Number of tubes in parallel per collector
    npar: int = 7 * 5 # Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments
    ns: int = 2 # Number of loops in series
    Lt: float = 1.15 * 20 # Collector individual tube length [m]
    Acs: float = 7.85e-5 # Flat plate collector tube cross-section area [m²]. Defaults to 7.85e-5
    Tmax: float = 110  # Maximum temperature of the solar field [ºC]
    Tmin: float = 10 # Minimum temperature of the solar field [ºC]
    qsf_min: float = 0.372 # Minimum (operating) flow rate [m³/h]. Taken from [20240925-20240927 aquasol-librescada]
    qsf_max: float = 8.928 # Maximum flow rate [m³/h]. Taken from [20240925-20240927 aquasol-librescada]
    delay_span: int = 600 # Max. time to keep previous inputs to accound for their delayed influence on the output [seconds]


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
    Tout: float,
    q_ant: np.ndarray[float],
    Tout_ant: float,
    Tin: float | np.ndarray[float],
    I: float,
    Tamb: float,
    model_params: ModelParameters = ModelParameters(),
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    sample_time: int = 1,
    consider_transport_delay: bool = True,
    water_props: w_props = None,
    log: bool = False,

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
    
    # warnings.warn('This model is deprecated, use the ones available in the `solar_field_inverse` module')

    # If there is no temperature difference, the flow rate cannot be estimated, it could be any
    if Tout - Tin.take(-1) < 0.5:
        return 0
    
    if water_props is None:
        water_props = w_props(P=0.2, T=(Tin.take(-1) + Tout) / 2 + 273.15)

    def wrapped_model(q):
        return (
            solar_field_model(
                Tout_ant=Tout_ant,
                Tin=Tin,
                q=q,
                I=I,
                Tamb=Tamb,
                model_params=model_params,
                fixed_model_params=fixed_model_params,
                sample_time=sample_time,
                consider_transport_delay=consider_transport_delay,
                water_props=water_props,
                log=log,
            ) - Tout
        )


    # Solve for q
    # initial_guess = q_ant.take(-1)
    initial_guess = (fixed_model_params.qsf_min + fixed_model_params.qsf_max)/2
    q = scipy.optimize.fsolve(wrapped_model, initial_guess)

    return np.clip(q, fixed_model_params.qsf_min, fixed_model_params.qsf_max)

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
    
    warnings.warn('This model is deprecated, use the ones available in the `solar_field_inverse` module')

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
    Tin: float | np.ndarray[float],
    q: float | np.ndarray[float],
    I: float,
    Tamb: float,
    model_params: ModelParameters = ModelParameters(),
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    sample_time: int = 1,
    consider_transport_delay: bool = False,
    water_props: w_props = None,
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
    
    mp = model_params
    fmp = fixed_model_params

    Tin = np.array(Tin)
    q = np.array(q)

    Leq = fmp.ns * fmp.Lt
    cf = fmp.npar * fmp.nt

    Tavg = (Tin.take(-1) + Tout_ant) / 2  # ºC
    if water_props is None:
        water_props = w_props(P=0.2, T=Tavg + 273.15)  # P=2 bar  -> 0.1MPa, T=Tin C,
    rho = water_props.rho  # [kg/m³]
    cp = water_props.cp * 1e3  # [kJ/kg·K] -> [J/kg·K]

    K1 = mp.beta / (rho * cp * fmp.Acs)  # m / (kg/m3 * J/kg·K) = K·m²/J
    K2 = mp.H / (Leq * fmp.Acs * rho * cp)  # J/sK / (m · m² · kg/m3 · J/kg·K) = 1/s
    K3 = (
        mp.gamma / (Leq * fmp.Acs * cf) * (1 / 3600)
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
    out = np.min([out, fmp.Tmax]) # Upper limit
    out = np.max([out, fmp.Tmin]) # Lower limit

    return out