from dataclasses import dataclass
import numpy as np
from loguru import logger
from iapws import IAPWS97 as w_props
import scipy
from solarmed_modeling.solar_field import ModelParameters, solar_field_model
from solarmed_modeling.solar_field import FixedModelParameters as FixedModelParametersDirect 

supported_eval_alternatives: list[str] = ["fsolve-direct-model", ]
""" 
    - fsolve-direct-model: Direct model with transport delay where flow is obtained by numerical methods
    - solve-inverse-model: [not-implemented] Solve for flow by clearing the flow in the model equations
    - solve-inverse-model-filter: [not-implemented] Solve for flow by clearing the flow in the model equations and apply a filter to the result
"""

@dataclass
class FixedModelParameters(FixedModelParametersDirect):
    Qmax: float = 8.928 # Maximum flow rate [m³/h]. Taken from [20240925-20240927 aquasol-librescada]
    Qmin: float = 0.372 # Minimum (operating) flow rate [m³/h]. Taken from [20240925-20240927 aquasol-librescada]
    max_flow_change: float = 0.15 #  Maximum flow rate change between inverse model evaluation [m³/h]
    min_temp_diff: float = 6.0  # Minimum temperature difference between inlet and outlet of solar field to actually evaluate inverse model [ºC or K]


def solar_field_inverse_model(
        Tout_ant: float, 
        Tin: float | np.ndarray[float], 
        Tout: float,
        q_ant: np.ndarray[float],
        I: float, 
        Tamb: float,
        model_params: ModelParameters,
        water_props: w_props = None,
        sample_time: int = 1,
        fixed_model_params: FixedModelParameters = None,
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
    fmp = fixed_model_params if fixed_model_params is not None else FixedModelParameters()

    # If there is no temperature difference, the flow rate cannot be estimated
    if Tout - Tin.take(-1) < fmp.min_temp_diff:
        # Assume the same flow rate as the previous time step
        return q_ant[-1]

    wrapped_model: callable = lambda q: solar_field_model(
        Tout_ant=Tout_ant, Tin=Tin, q=q, I=I, Tamb=Tamb, 
        model_params=model_params,
        fixed_model_params=fixed_model_params, 
        sample_time=sample_time, 
        consider_transport_delay=True, water_props=water_props) - Tout

    # Solve for q
    initial_guess = q_ant.take(-1)
    initial_guess = np.min([initial_guess, fmp.Qmax])
    initial_guess = np.max([initial_guess, 0])
    
    upper_bound = np.min([initial_guess*(1+fmp.max_flow_change), fmp.Qmax])
    lower_bound = np.max([0, initial_guess*(1-fmp.max_flow_change)])
    bounds = (np.min([lower_bound, upper_bound-0.1]), # To make sure lower bound is lower than upper bound
              np.max([upper_bound, fmp.Qmin])) # To make sure upper bound is always greater than zero
    
    sol = scipy.optimize.least_squares(wrapped_model, initial_guess, bounds=bounds)

    return sol.x[0]