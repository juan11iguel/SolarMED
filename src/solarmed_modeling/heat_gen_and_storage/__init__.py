from typing import Literal
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
import scipy
from optimparallel import minimize_parallel
from iapws import IAPWS97 as w_props

from solarmed_modeling.data_validation import conHotTemperatureType_upper_limit as Tmax
from solarmed_modeling.data_validation import conHotTemperatureType_lower_limit as Tmin
from solarmed_modeling.solar_field import solar_field_model, ModelParameters as SfModParams
from solarmed_modeling.heat_exchanger import heat_exchanger_model, ModelParameters as HexModParams 
from solarmed_modeling.thermal_storage import thermal_storage_two_tanks_model, ModelParameters as TsModParams

supported_eval_alternatives: list[str] = ["standard", ]
""" 
    - standard: Combined component models where Ts,out and Tts,c,b are solved 
    simulatenously using `scipy.optimize.least_squares`, water properties are 
    kept constant throught the evaluation.
"""

@dataclass
class ModelParameters:
    sf: SfModParams   = field(default_factory=lambda: SfModParams())
    ts: TsModParams   = field(default_factory=lambda: TsModParams())
    hex: HexModParams = field(default_factory=lambda: HexModParams())


def heat_generation_and_storage_subproblem(
    qsf: np.ndarray[float], Tsf_in_ant: np.ndarray[float], Tsf_out_ant: float,
    qts_src: float, qts_dis: float,
    Tts_b_in: float, Tts_h: np.ndarray[float], Tts_c: np.ndarray[float], 
    Tamb: float, I: float,  
    model_params: ModelParameters,
    sample_time: int,
    water_props: tuple[w_props, w_props] = None,
    problem_type: Literal["1p2x", "2p1x"] = "1p2x",
    solver: Literal["scipy", "optimparallel"] = "optimparallel",
    solver_method: str = "lm",
) -> tuple[float, float, float, np.ndarray[float], np.ndarray[float]]:
    """
    Solves the heat generation and storage subproblem for a solar field and thermal storage system.
    Parameters:
    -----------
    qsf : float
        Solar field flow rate (m³/h).
    Tsf_in_ant : np.ndarray[float]
        Previous solar field inlet temperatures (ºC).
    Tsf_out_ant : float
        Previous solar field outlet temperature (ºC).
    qts_src : float
        Thermal storage charge flow rate (m³/h).
    qts_dis : float
        Thermal storage discharge flow rate (m³/h).
    Tts_b_in : float
        Bottom tank inlet temperature (ºC).
    Tts_h : np.ndarray[float]
        Previous hot tank temperatures (ºC).
    Tts_c : np.ndarray[float]
        Previous cold tank temperatures (ºC).
    Tamb : float
        Ambient temperature (ºC).
    I : float
        Solar irradiance (W/m²).
    model_params : ModelParameters
        Model parameters for the system.
    # component_models : list[ComponentModel]
    #     list containing one element with the model function and a parameters 
    #     dataclass for each of the subsystem components, in the following order:
    #     0: heat generation, 1: heat exchange, 3: heat storage
    sample_time : int
        Sample time (seconds).
    water_props : tuple[w_props, w_props], optional
        Water properties for the hot and cold tanks.
    problem_type : Literal["1p2x", "2p1x"], optional
        Type of problem to solve. Default is "1p2x" (1 problem, 2 variables).
    solver : Literal["scipy", "optimparallel"], optional
        Solver to use for finding the unknown variables. Default is "scipy".
    Returns:
    --------
    tuple[float, float, float, np.ndarray[float], np.ndarray[float]]
        Tuple containing estimations for the solar field inlet temperature, solar field outlet temperature,
        thermal storage tank inlet temperature, hot tank temperatures, and cold tank temperatures.
    """    

    def inner_function(x, return_states: bool = False):
        """
        Variables that end with an underscore are the ones calculated in the 
        inner function, to avoid overwriting the outer scope variables.
        """
        
        if len(x) == 2:
            Tsf_out = x[0]
            Tts_c_b = x[1]
        elif len(x) == 1:
            # Bottom tank temperature is not considered to change
            Tsf_out = x[0]
            Tts_c_b = Tts_c_b_orig
        else:
            raise ValueError("Invalid number of decision variables")

        # Heat exchanger of solar field - thermal storage
        qhx_p = qsf[-1]
        qhx_s = qts_src
        if qhx_p < 0.1 or qhx_s < 0.1:
            # HEX not exchanging heat
            if qhx_p < 0.1 and not qhx_s < 0.1:
                # Primary circuit not operating, HEX temps probably similar to only inlet
                Tsf_in_, Tts_t_in_ = Tts_c_b, Tts_c_b
            elif not qhx_p < 0.1 and qhx_s < 0.1:
                # Secondary circuit not operating, HEX temps probably similar to only inlet
                Tsf_in_, Tts_t_in_ = Tsf_out, Tsf_out
            else:
                # Both circuits not operating
                Tsf_in_, Tts_t_in_ = Tsf_in_ant[-1], Tsf_in_ant[-1]
        else:
            Tsf_in_, Tts_t_in_ = heat_exchanger_model(
                Tp_in=Tsf_out,  # Solar field outlet temperature (decision variable, ºC)
                Ts_in=Tts_c_b,  # Cold tank bottom temperature (ºC)
                qp=qsf[-1],  # Solar field flow rate (m³/h)
                qs=qts_src,  # Thermal storage charge flow rate (decision variable, m³/h)
                Tamb=Tamb,
                
                UA=model_params.hex.UA,
                H=model_params.hex.H,
                water_props=water_props,
            )

        # Solar field
        Tsf_in_ = np.append(Tsf_in_ant, Tsf_in_)

        Tsf_out_ = solar_field_model(
            Tin=Tsf_in_,
            q=qsf,
            I=I,
            Tamb=Tamb,
            Tout_ant=Tsf_out_ant,
            
            beta=model_params.sf.beta,
            H=model_params.sf.H,
            gamma=model_params.sf.gamma,
            water_props = water_props[0],
            sample_time=sample_time,
            consider_transport_delay=True,
        )

        # Thermal storage
        Tts_h_, Tts_c_ = thermal_storage_two_tanks_model(
            Ti_ant_h=Tts_h, Ti_ant_c=Tts_c,  # [ºC], [ºC]
            Tt_in=Tts_t_in_,  # ºC
            Tb_in=Tts_b_in,  # ºC
            Tamb=Tamb,  # ºC
            qsrc=qts_src,  # m³/h
            qdis=qts_dis,  # m³/h
            
            UA_h=model_params.ts.UA_h,  # W/K
            UA_c=model_params.ts.UA_c,  # W/K
            Vi_h=model_params.ts.V_h,  # m³
            Vi_c=model_params.ts.V_c,  # m³
            water_props=water_props,
            ts=sample_time, # seg 
            Tmin=Tmin  # ºC
        )
        Tts_c_b_ = Tts_c_[-1]

        if return_states:
            return Tsf_in_[-1], Tsf_out_, Tts_t_in_, Tts_h_, Tts_c_
        elif len(x) == 2:
            return np.array( [abs(Tsf_out - Tsf_out_), abs(Tts_c_b - Tts_c_b_) ])
        elif len(x) == 1:
            return [abs(Tsf_out - Tsf_out_)]
        else:
            raise ValueError("Invalid number of decision variables")
    # End of inner function ---------------------------------------------------
           

    if problem_type != "1p2x":
        raise NotImplementedError("Currently, only `1p2x` alternative is implemented")
    
    if water_props is None:
        # Initialize from input values
        water_props = (
            w_props(P=0.2, T=Tts_h[0] + 273.15),
            w_props(P=0.2, T=Tts_c[-1] + 273.15)
        )
    
    Tts_c_b_orig: float | None = None
    if problem_type == "1p2x":
        pass
    elif problem_type == "2p1x":
        Tts_c_b_orig = float(Tts_c[-1]) # To have an inmutable value
    else:
        raise ValueError("Invalid problem type")
    
    # Cap solar field outlet temperature
    if Tsf_out_ant > Tmax:
        Tsf_out_ant = Tmax
    if Tsf_out_ant < Tmin:
        Tsf_out_ant = Tmin
        
    initial_guess = [Tsf_out_ant, Tts_c[-1]]
    bounds = ((Tmin, Tmin), (Tmax, Tmax)) if solver_method != "lm" else None 

    if solver == "scipy":
        outputs = scipy.optimize.least_squares(fun=inner_function, x0=initial_guess, bounds=bounds, xtol=1e-1, ftol=1e-1, method=solver_method)
    elif solver == "optimparallel":
        outputs = minimize_parallel(fun=inner_function, x0=initial_guess, bounds=bounds, tol=1e-1)
    else:
        raise ValueError(f"Invalid solver {solver}, options are: 'scipy', 'optimparallel'")
    
    # Cap solar field outlet temperature
    if outputs.x[0] > Tmax:
        outputs.x[0] = Tmax
    
    # With the system of equations solved, calculate the outputs
    return inner_function(outputs.x, return_states=True)