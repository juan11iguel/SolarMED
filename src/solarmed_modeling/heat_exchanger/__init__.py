from typing import Literal
from dataclasses import dataclass
import math
import numpy as np
from loguru import logger
from iapws import IAPWS97 as w_props

supported_eval_alternatives: list[str] = ["standard", "constant-water-props"]
""" 
    - standard: Counter-flow heat exchanger steady state model based on the effectiveness-NTU method
    - constant-water-props: Standard model but with constant water properties evaluated at some average temperature
"""

@dataclass
class ModelParameters:
    UA: float = 13536.596 # Heat transfer coefficient (W/K)
    H: float  = 0.0 # Losses to the environment (W/m²)
   

def heat_exchanger_model(
    Tp_in: float, Ts_in: float, qp: float, qs: float, Tamb: float, 
    model_params: ModelParameters,
    hex_type: Literal['counter_flow',] = 'counter_flow',
    water_props: tuple[w_props, w_props] = None, 
    return_epsilon: bool = False, 
    epsilon: float = None,
    Tmin: float = 5,
    log: bool = True, 
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

    # HEX not really exchanging heat, bypass model
    if qp < 0.1 or qs < 0.1:
        Tp_out = Tp_in - model_params.H * (Tp_in - Tamb)
        Ts_out = Ts_in - model_params.H * (Ts_in - Tamb)

        Tp_out = np.max([Tp_out, Tmin])
        Ts_out = np.max([Ts_out, Tmin])

        if return_epsilon:
            return Tp_out, Ts_out, 0
        else:
            return Tp_out, Ts_out

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
        NTU = model_params.UA / Cmin
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

    Tp_out = np.max([Tp_out, Tmin])
    Ts_out = np.max([Ts_out, Tmin])

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
