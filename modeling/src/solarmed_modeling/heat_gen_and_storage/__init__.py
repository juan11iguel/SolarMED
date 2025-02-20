from typing import Literal
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
import scipy
from optimparallel import minimize_parallel
from iapws import IAPWS97 as w_props

# from solarmed_modeling.data_validation import conHotTemperatureType_upper_limit as Tmax
# from solarmed_modeling.data_validation import conHotTemperatureType_lower_limit as Tmin
from solarmed_modeling.solar_field import (solar_field_with_q_validation_model, 
                                           ModelParameters as SfModParams, 
                                           FixedModelParameters as SfFixedModParams)
from solarmed_modeling.heat_exchanger import (heat_exchanger_model,
                                              ModelParameters as HexModParams)
from solarmed_modeling.thermal_storage import (thermal_storage_two_tanks_model, 
                                               ModelParameters as TsModParams,
                                               FixedModelParameters as TsFixedModParams)

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
    
@dataclass
class FixedModelParameters:
    sf: SfFixedModParams = field(default_factory=lambda: SfFixedModParams())
    ts: TsFixedModParams = field(default_factory=lambda: TsFixedModParams())


def heat_generation_and_storage_subproblem(
    qsf: float, qsf_ant: np.ndarray[float], Tsf_in_ant: np.ndarray[float], Tsf_out_ant: float,
    qts_src: float, qts_dis: float,
    Tts_b_in: float, Tts_h: np.ndarray[float], Tts_c: np.ndarray[float], 
    Tamb: float, I: float,  
    model_params: ModelParameters,
    sample_time: int,
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
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

    def inner_function(x, return_states: bool = False, enforce_limits_sf: bool = False) ->  np.ndarray | tuple[float, float, float, np.ndarray[float], np.ndarray[float]]:
        """
        Variables that end with an underscore are the ones calculated in the 
        inner function, to avoid overwriting the outer scope variables.
        """
        
        if len(x) == 3:
            Tsf_out = x[0]
            Tts_c_b = x[1]
            qsf_ = x[2]
        elif len(x) == 1:
            # Bottom tank temperature is not considered to change
            Tsf_out = x[0]
            qsf_ = qsf
            Tts_c_b = Tts_c_b_orig
        else:
            raise ValueError("Invalid number of decision variables")

        # Heat exchanger of solar field - thermal storage
        qhx_p = qsf_
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
                qp=qhx_p,  # Solar field flow rate (m³/h)
                qs=qts_src,  # Thermal storage charge flow rate (decision variable, m³/h)
                Tamb=Tamb,
                
                model_params=model_params.hex,
                water_props=water_props,
                Tmin=fixed_model_params.sf.Tmin,
            )

        # Solar field
        Tsf_out_, qsf_ = solar_field_with_q_validation_model(
            Tin=np.append(Tsf_in_ant, Tsf_in_),
            q=qsf_,
            q_ant=qsf_ant,
            I=I,
            Tamb=Tamb,
            Tout_ant=Tsf_out_ant,
            
            model_params=model_params.sf,
            fixed_model_params=fixed_model_params.sf,
            water_props = water_props[0],
            sample_time=sample_time,
            consider_transport_delay=True,
            enforce_final_limits=enforce_limits_sf,
        )

        # Thermal storage
        Tts_h_, Tts_c_ = thermal_storage_two_tanks_model(
            Ti_ant_h=Tts_h, Ti_ant_c=Tts_c,  # [ºC], [ºC]
            Tt_in=Tts_t_in_,  # ºC
            Tb_in=Tts_b_in,  # ºC
            Tamb=Tamb,  # ºC
            qsrc=qts_src,  # m³/h
            qdis=qts_dis,  # m³/h
            
            model_params=model_params.ts,
            fixed_model_params=fixed_model_params.ts,
            water_props=water_props,
            sample_time=sample_time, # seg 
            # Tmin=Tmin  # ºC
        )
        Tts_c_b_ = Tts_c_[-1]

        if return_states:
            return Tsf_in_, Tsf_out_, Tts_t_in_, Tts_h_, Tts_c_, qsf_
        elif len(x) == 3:
            return np.array([ abs(Tsf_out - Tsf_out_), abs(Tts_c_b - Tts_c_b_), abs(qsf-qsf_) ])
        elif len(x) == 1:
            return [abs(Tsf_out - Tsf_out_)]
        else:
            raise ValueError("Invalid number of decision variables")
    # End of inner function ---------------------------------------------------
           
    Tmin: float = fixed_model_params.sf.Tmin
    Tmax: float = fixed_model_params.sf.Tmax
    qsf_max: float = fixed_model_params.sf.qsf_max

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
    Tsf_out_ant = max(min(Tsf_out_ant, Tmax), Tmin)
    initial_guess = [Tsf_out_ant, Tts_c[-1], qsf]
    bounds = ((Tmin, Tmin, qsf*0.9), (Tmax, Tmax, qsf_max)) if solver_method != "lm" else None 

    if solver == "scipy":
        outputs = scipy.optimize.least_squares(fun=inner_function, x0=initial_guess, bounds=bounds, xtol=1e-1, ftol=1e-1, method=solver_method)
    elif solver == "optimparallel":
        outputs = minimize_parallel(fun=inner_function, x0=initial_guess, bounds=bounds, tol=1e-1)
    else:
        raise ValueError(f"Invalid solver {solver}, options are: 'scipy', 'optimparallel'")
    
    # Cap solar field outlet temperature
    outputs.x[1] = min(Tmax, outputs.x[1])
    
    # With the system of equations solved, calculate the outputs
    return inner_function(outputs.x, return_states=True, enforce_limits_sf=True)