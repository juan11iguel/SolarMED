from loguru import logger
import numpy as np
from iapws import IAPWS97 as w_props
import math
from typing import Literal

def estimate_flow_secondary(Tp_in: float, Ts_in: float, qp: float, Tp_out: float, Ts_out: float) -> float:
    if np.abs(Ts_out - Ts_in) < 1:
        return np.nan

    try:
        w_props_Tp_in = w_props(P=0.16, T=(Tp_in + Tp_out) / 2 + 273.15)
        w_props_Ts_in = w_props(P=0.16, T=(Ts_in + Ts_out) / 2 + 273.15)
    except NotImplementedError:
        logger.warning(f'Invalid temperature input values: Tp_in={Tp_in}, Ts_in={Ts_in}, Tp_out={Tp_out}, Ts_out={Ts_out} (ºC), returning NaN')
        return np.nan

    cp_Tp_in = w_props_Tp_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    cp_Ts_in = w_props_Ts_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]

    qs = qp * (cp_Tp_in * (Tp_in - Tp_out)) / (cp_Ts_in * (Ts_out - Ts_in))

    return np.max([qs, 0])


def heat_exchanger_model(Tp_in: float, Ts_in: float, qp: float, qs: float, Tamb: float, UA: float = 13536.596, H: float = 0,
                         log: bool = True, hex_type: Literal['counter_flow',] = 'counter_flow',
                         return_epsilon: bool = False, epsilon: float = None):  # eta_p, eta_s):

    """Counter-flow heat exchanger steady state model.

    Based on the effectiveness-NTU method [2] - Chapter Heat exchangers 11-5:

    ΔTa: Temperature difference between primary circuit inlet and secondary circuit outlet
    ΔTb: Temperature difference between primary circuit outlet and secondary circuit inlet

    `p` references the primary circuit, usually the hot side, unless the heat exchanger is inverted.
    `s` references the secondary circuit, usually the cold side, unless the heat exchanger is inverted.
    `Qdot` is the heat transfer rate
    `C` is the capacity ratio, defined as the ratio of the heat capacities of the two fluids, C = Cmin/Cmax

    To avoid confussion, whichever the heat exchange direction is, the hotter side will be referenced as `h` and the colder side as `c`.

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
    - It has been assumed that the rate of change for the temperature of both fluids is proportional to the temperature difference; this assumption is valid for fluids with a constant specific heat, which is a good description of fluids changing temperature over a relatively small range. However, if the specific heat changes, the LMTD approach will no longer be accurate.
    - A particular case for the LMTD are condensers and reboilers, where the latent heat associated to phase change is a special case of the hypothesis. For a condenser, the hot fluid inlet temperature is then equivalent to the hot fluid exit temperature.
    - It has also been assumed that the heat transfer coefficient (U) is constant, and not a function of temperature. If this is not the case, the LMTD approach will again be less valid
    - The LMTD is a steady-state concept, and cannot be used in dynamic analyses. In particular, if the LMTD were to be applied on a transient in which, for a brief time, the temperature difference had different signs on the two sides of the exchanger, the argument to the logarithm function would be negative, which is not allowable.
    - No phase change during heat transfer
    - Changes in kinetic energy and potential energy are neglected

    [1] W. M. Kays and A. L. London, Compact heat exchangers: A summary of basic heat transfer and flow friction design data. McGraw-Hill, 1958. [Online]. Available: https://books.google.com.br/books?id=-tpSAAAAMAAJ

    [2] Y. A. Çengel and A. J. Ghajar, Heat and mass transfer: fundamentals & applications, Fifth edition. New York, NY: McGraw Hill Education, 2015.

    Args:
        Tp_in (float): Primary circuit inlet temperature [C]
        Ts_in (float): Secondary circuit inlet temperature [C]
        qp (float): Primary circuit volumetric flow rate [m^3/h]
        qs (float): Secondary circuit volumetric flow rate [m^3/h]
        UA (float, optional): Heat transfer coefficient multiplied by the exchange surface area [W·ºC^-1]. Defaults to 28000.

    Returns:
        Tp_out: Primary circuit outlet temperature [C]
        Ts_out: Secondary circuit outlet temperature [C]
    """

    # TODO: Add option to simplify model by using constant water properties

    if hex_type != 'counter_flow':
        raise ValueError('Only counter-flow heat exchangers are supported')

    inverted_hex = False

    w_props_Tp_in = w_props(P=0.16, T=Tp_in + 273.15)
    w_props_Ts_in = w_props(P=0.16, T=Ts_in + 273.15)
    cp_Tp_in = w_props_Tp_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    cp_Ts_in = w_props_Ts_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]

    mp = qp / 3600 * w_props_Tp_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s
    ms = qs / 3600 * w_props_Ts_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s

    Cp = mp * cp_Tp_in
    Cs = ms * cp_Ts_in
    Cmin = np.min([Cp, Cs])
    Cmax = np.max([Cp, Cs])

    # mcp_min = min([mp*cp_Tp_in, ms*cp_Ts_in])
    # mcp_max = max([mp*cp_Tp_in, ms*cp_Ts_in])

    # theta = UA*(1/mcp_max-1/mcp_min)

    # eta_p = (1-math.e**theta)/( 1-math.e**theta*(mcp_min/mcp_max) )
    # eta_s = mp*cp_Tp_in/(ms*cp_Ts_in)

    # Tp_out = Tp_in - eta_p*(mcp_min)/(mp*cp_Tp_in)*(Tp_in-Ts_in) - H*(Tp_in-Tamb) # ºC
    # Ts_out = Ts_in + eta_s*(Tp_in-Tp_out) # ºC

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

        if log: logger.warning('Inverted operation in heat exchanger')

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

# def heat_exchanger_model_simple(Tp_in_C, Ts_in_C, qp_m3h, qs_m3h, eta=0.85):

#     """Simplified Heat exhanger steady state model.

#     Args:
#         Tp_in (float): Primary (heat source) circuit inlet temperature [ºC]
#         Ts_in (float): Secondary (heat sink) circuit inlet temperature [ºC]
#         qp (float): Primary circuit volumetric flow rate [m³/h]
#         qs (float): Secondary circuit volumetric flow rate [m³/h]
#         eta (float, optional): Exchange efficiency parameter. Defaults to 0.85.

#     Returns:
#         Tp_out: Primary circuit outlet temperature [C]
#         Ts_out: Secondary circuit outlet temperature [C]
#     """

#     Cp_p = w_props(P=0.1, T=Tp_in_C+273.15).Cp # KJ/kg·K
#     Cp_s = w_props(P=0.1, T=Ts_in_c+273.15).Cp

#     Tp_out_C = Tp_in_C - eta* (qp_m3h*Cp_p)/(qs_m3h*Cp_s) * (Tp_in_C-Ts_in_C)
#     Ts_out_C = Ts_in_C + eta* (qp_m3h*Cp_p)/(qs_m3h*Cp_s) * (Tp_in_C-Tp_out_C)

#     return Tp_out_C, Ts_out_C