from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar
import math
from loguru import logger
import numpy as np

def heat_exchanger_model(Tp_in, Ts_in, Qp, Qs, Tamb, UA=28000, H=0):  # eta_p, eta_s):
    """Heat exhanger steady state model.

    [1] G. Ampuño, L. Roca, M. Berenguel, J. D. Gil, M. Pérez, and J. E. Normey-Rico,
    “Modeling and simulation of a solar field based on flat-plate collectors,” Solar Energy,
    vol. 170, pp. 369–378, Aug. 2018, doi: 10.1016/j.solener.2018.05.076.

    Args:
        Tp_in (float): Primary circuit inlet temperature [C]
        Ts_in (float): Secondary circuit inlet temperature [C]
        Qp (float): Primary circuit volumetric flow rate [m^3/h]
        Qs (float): Secondary circuit volumetric flow rate [m^3/h]
        UA (float, optional): Heat transfer coefficient multiplied by the exchange surface area [W·ºC^-1]. Defaults to 28000.

    Returns:
        Tp_out: Primary circuit outlet temperature [C]
        Ts_out: Secondary circuit outlet temperature [C]
        Pgen: Power supplied by the primary circuit [kWth]
        Pabs: Power absorbed by the secondary circuit [kWth]
    """

    # TODO: Add losses to the environment (H)
    # TODO: Add option to simplify model by using constant water properties
    # TODO: Allow inversion of the heat exchanger?

    w_props_Tp_in = w_props(P=0.16, T=Tp_in + 273.15)
    w_props_Ts_in = w_props(P=0.16, T=Ts_in + 273.15)
    cp_Tp_in = w_props_Tp_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
    cp_Ts_in = w_props_Ts_in.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]

    mp = Qp / 3600 * w_props_Tp_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s
    ms = Qs / 3600 * w_props_Ts_in.rho  # rho [kg/m³] # Convertir m^3/s a kg/s

    # mcp_min = min([mp*cp_Tp_in, ms*cp_Ts_in])
    # mcp_max = max([mp*cp_Tp_in, ms*cp_Ts_in])

    # theta = UA*(1/mcp_max-1/mcp_min)

    # eta_p = (1-math.e**theta)/( 1-math.e**theta*(mcp_min/mcp_max) )
    # eta_s = mp*cp_Tp_in/(ms*cp_Ts_in)

    # Tp_out = Tp_in - eta_p*(mcp_min)/(mp*cp_Tp_in)*(Tp_in-Ts_in) - H*(Tp_in-Tamb) # ºC
    # Ts_out = Ts_in + eta_s*(Tp_in-Tp_out) # ºC

    if Qp == 0:
        Tp_out = Tamb
        if Qs == 0:
            Ts_out = Tamb
        else:
            Ts_out = Ts_in

        return Tp_out, Ts_out

    if Qs == 0:
        Ts_out = Tamb
        if Qp == 0:
            Tp_out = Tamb
        else:
            Tp_out = Tp_in

        return Tp_out, Ts_out

    delta_T1 = Tp_in - Ts_in
    # Direction of exchange should be inverted, temporary just bypass hex
    if delta_T1 < 0:
        logger.warning('Inverted operation in heat exchanger, bypassing')
        Tp_out = Tp_in
        Ts_out = Ts_in

        return Tp_out, Ts_out

    delta_T2 = delta_T1 - (ms / mp) * delta_T1

    # if abs(delta_T1 - delta_T2) < 0.5:
    #     logger.warning('delta_T1 equal to delta_T2 in hex')
    #     Tp_out = Tp_in
    #     Ts_out = Ts_in

    #     return Tp_out, Ts_out

    delta_T_lm = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)

    # Calculate the heat transfer rate (Q)
    Q = UA * delta_T_lm

    # Calculate the outlet temperatures
    Tp_out = Tp_in - (Q / (mp * cp_Tp_in)) - H * (Tp_in - Tamb)
    Ts_out = Ts_in + (Q / (ms * cp_Ts_in))

    if Ts_out > Tp_out:
        # logger.warning('Unfesible result')
        Ts_out = Tp_out - 0.5

        # cp_Tp_out = w_props(P=0.1, T=Tp_out+273.15).cp # P=0.1 bar->0.1 MPa, T=Tp_out C, cp [kJ/kg·K]
        # cp_Ts_out = w_props(P=0.1, T=Ts_out+273.15).cp # P=0.1 bar->0.1 MPa, T=Ts_out C, cp [kJ/kg·K]

        # Pgen = mp*(cp_Tp_in+cp_Tp_out)/2*(Tp_in-Tp_out)/1000 # kWth
        # Pabs = ms*(cp_Ts_in+cp_Ts_out)/2*(Ts_out-Ts_in)/1000 # kWth

    return Tp_out, Ts_out


def calculate_heat_transfer_effectiveness(Tp_in: float, Tp_out: float, Ts_in: float, Ts_out: float, qp: float, qs: float) -> float:
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

    Cmin = np.min([mp*cp_Tp_in, ms*cp_Ts_in])

    # It could be calculated with any, just to disregard specific heat capacity
    if abs(Cmin - mp*cp_Tp_in) < 1e-6:  # Primary circuit is the one with the lowest heat capacity
        eta = (Tp_in - Tp_out) / (Tp_in - Ts_in)
    else: # Secondary circuit is the one with the lowest heat capacity
        eta = (Ts_out - Ts_in) / (Tp_in - Ts_in)

    return eta

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