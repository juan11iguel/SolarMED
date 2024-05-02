import numpy
import numpy as np
from loguru import logger
from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar
from scipy.optimize import fsolve
from .data_validation import conHotTemperatureType
from pydantic import PositiveFloat, PositiveInt
from scipy import signal
from scipy.optimize import fsolve

b, a = signal.butter(3, 0.005)
zi = signal.lfilter_zi(b, a)

def calculate_total_pipe_length(q, n, sample_time=10, equivalent_pipe_area=7.85e-5):
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


def find_delay_samples(q, sample_time=1, total_pipe_length: float = 5e7, equivalent_pipe_area: float = 7.85e-5, log=True) -> int:
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
        Tout_ant: float, Tin: float | np.ndarray, Tout: float,
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
        Tout_ant: float, Tin: float | np.ndarray, Tout: float,
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
    q = numpy.max([q, qmin])
    q = numpy.min([q, qmax])
    # except Exception as e:
    #     logger.error(f'Error: {e}, q: {q}, qmin: {qmin}, qmax: {qmax}, deltaT: {deltaT}, Tout_ant: {Tout_ant}, Tout: {Tout}, I: {I}, Tin: {Tin}, Tamb: {Tamb}')
    #     q = qmin

    if q_ant is not None and filter_signal:
        # q = signal.filtfilt(b, a, numpy.append(q_ant, q))[-1]
        # q, _ = signal.lfilter(b, a, numpy.append(q_ant, q), zi=zi*q_ant[0])
        # q = q.take(-1)
        # Weighted average of the last n samples
        q = f * q + (1 - f) * q_ant[-n:].mean()

    return q


def solar_field_model(
        Tout_ant: float, Tin: float | numpy.ndarray, q: float | numpy.ndarray, I: float, Tamb: float,
        beta: float = 0.0975, gamma: float = 1.0, H: float = 2.2,
        sample_time=1, consider_transport_delay: bool = False,
        nt=1, npar=7 * 5, ns=2, Lt=1.15 * 20, Acs: float = 7.85e-5,
        log: bool = False
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

    Tin = numpy.array(Tin)
    q = numpy.array(q)

    Leq = ns * Lt
    cf = npar * nt

    if Tout_ant > 120:
        # Above 110ºC, the model is not valid
        return 9999

    Tavg = (Tin.take(-1) + Tout_ant) / 2  # ºC

    w_props_avg = w_props(P=0.16, T=Tavg + 273.15)  # P=1 bar  -> 0.1MPa, T=Tin C,
    rho = w_props_avg.rho  # [kg/m³]
    cp = w_props_avg.cp * 1e3  # [kJ/kg·K] -> [J/kg·K]

    K1 = beta / (rho * cp * Acs)  # m / (kg/m3 * J/kg·K) = K·m²/J
    K2 = H / (Leq * Acs * rho * cp)  # J/sK / (m · m² · kg/m3 · J/kg·K) = 1/s
    K3 = gamma / (Leq * Acs * cf) * (1 / 3600)  # 1/(m · m² · -) * (1 / 3600s) = h/(3600·m³·s)

    # deltaTout_m = m [m3/h * kg/m3*1h/3600s] * deltaT [K] * K3 [1/(m * * m2 * kg/m3 * -)]

    if q.take(-1) == 0:
        # Just thermal losses
        deltaTout = - K2 * (Tavg - Tamb)

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

        deltaTout = K1 * I - K2 * (Tavg - Tamb) - K3 * q.take(-1) * (Tout_ant - Tin.take(-n))

    return Tout_ant + deltaTout * sample_time

# def solar_field_model_temp(Tin, Q, I, Tamb, beta, H, nt=1, np=7 * 5, ns=2, Lt=1.15 * 20):  # , Acs=7.85e-5):
#     """Steady state model of a flat plate collector solar field
#        with any number of collectors in series and parallel.
#
#     [1] G. Ampuño, L. Roca, M. Berenguel, J. D. Gil, M. Pérez, and J. E. Normey-Rico,
#         “Modeling and simulation of a solar field based on flat-plate collectors,” Solar Energy,
#         vol. 170, pp. 369–378, Aug. 2018, doi: 10.1016/j.solener.2018.05.076.
#
#
#     Args:
#         I (float): Solar radiation [W/m2]
#         Tin (float): Inlet fluid temperature [ºC]
#         Tout (float): Current outlet fluid temperature [ºC]
#         Tout_ant (float): Prior outlet fluid temperature [ºC]
#         Tamb (float): Ambient temperature [ºC]
#         period (int): Time elapsed [s]
#         nt (int, optional): Number of tubes in parallel per collector. Defaults to 1.
#         np (int, optional): Number of collectors in parallel per loop. Defaults to 7.
#         ns (int, optional): Number of loops in series. Defaults to 2.
#         Lt (float, optional): Collector tube length [m]. Defaults to 97.
#         H (float, optional): Thermal losses coefficient [J·s^-1·C^-1]. Defaults to 2.2.
#         nt (int, optional): Number of tubes in parallel per collector. Defaults to 1.
#         np (int, optional): Number of collectors in parallel per loop. Defaults to 7.
#         ns (int, optional): Number of loops in series. Defaults to 2.
#         Lt (float, optional): Collector tube length [m]. Defaults to 97.
#         Acs (float, optional): Flat plate collector tube cross-section area [m²]. Defaults to 7.85e-5
#
#     Returns:
#         Q (float): Total volumetric solar field flow rate [m^3/s]
#         Pgen (float): Power generated [kWth]
#         SEC (float): Conversion factor / Specific energy consumption  [kWe/kWth]
#     """
#
#     def inner_function(Tout):
#
#         error = abs(Tin + (beta * I - H * ((Tin + Tout) / 2 - Tamb)) / (m * cp / cf * 1 / Leq) - Tout)  # ºC
#
#         return error
#
#     Leq = ns * Lt
#     cf = np * nt * 1  # Convertir m^3/h a kg/s y algo más
#     w_props_avg = w_props(P=0.16, T=Tin + 273.15)  # P=1 bar  -> 0.1MPa, T=Tin C,
#     cp = w_props_avg.cp * 1e3  # [kJ/kg·K] -> [J/kg·K]
#     rho = w_props_avg.rho  # [kg/m³]
#     m = Q * rho / 3600 / np  # m^3/h -> kg/s (entre número de captadores en paralelo)
#
#     # cp_Tamb = w_props(P=0.1, T=Tamb+273.15).cp # P=1 bar  -> 0.1MPa, T=Tamb C, cp [kJ/kg·K]
#     # m = Q/3600*w_props(P=0.1, T=Tin+273.15).rho # rho [kg/m³] # Convertir m^3/s a kg/s
#
#     if Q < 0:
#         Tout = Tin
#         logger.warning('Negative or null temperature gradient. Flow set to 0')
#     else:
#         # Temporary removed: - H*(Tavg-Tamb)
#         # bounds = (Tin, 110)
#         Tout0 = Tin + (beta * I - H * (Tin - Tamb)) / (m * cp / cf * 1 / Leq)  # ºC
#         Tout = fsolve(inner_function, Tout0)[0]  # ºC bounds=bounds
#
#     return Tout