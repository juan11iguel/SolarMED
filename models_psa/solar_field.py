from loguru import logger
from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar
from scipy.optimize import fsolve

def solar_field_model(Tin, Tout, I, Tamb, beta, H, nt=1, np=7 * 5, ns=2, Lt=1.15 * 20):
    """Steady state model of a flat plate collector solar field
       with any number of collectors in series and parallel.

    [1] G. Ampuño, L. Roca, M. Berenguel, J. D. Gil, M. Pérez, and J. E. Normey-Rico,
        “Modeling and simulation of a solar field based on flat-plate collectors,” Solar Energy,
        vol. 170, pp. 369–378, Aug. 2018, doi: 10.1016/j.solener.2018.05.076.


    Args:
        I (float): Solar radiation [W/m2]
        Tin (float): Inlet fluid temperature [ºC]
        Tout (float): Current outlet fluid temperature [ºC]
        Tout_ant (float): Prior outlet fluid temperature [ºC]
        Tamb (float): Ambient temperature [ºC]
        period (int): Time elapsed [s]
        beta (float, optional): Irradiance model parameter [m]. Defaults to 0.0975.
        H (float, optional): Thermal losses coefficient [J·s^-1·C^-1]. Defaults to 2.2.
        nt (int, optional): Number of tubes in parallel per collector. Defaults to 1.
        np (int, optional): Number of collectors in parallel per loop. Defaults to 7.
        ns (int, optional): Number of loops in series. Defaults to 2.
        Lt (float, optional): Collector tube length [m]. Defaults to 97.
        Acs (float, optional): Flat plate collector tube cross-section area [m²]. Defaults to 7.85e-5

    Returns:
        Q (float): Total volumetric solar field flow rate [m^3/s]
        Pgen (float): Power generated [kWth]
        SEC (float): Conversion factor / Specific energy consumption  [kWe/kWth]
    """

    Leq = ns * Lt
    cf = np * nt * 1  # Convertir m^3/h a kg/s y algo más
    Tavg = (Tin + Tout) / 2  # ºC

    w_props_avg = w_props(P=0.16, T=Tavg + 273.15)  # P=1 bar  -> 0.1MPa, T=Tin C,
    cp_avg = w_props_avg.cp * 1e3  # [kJ/kg·K]
    rho_avg = w_props_avg.rho  # [kg/m³]

    # cp_Tamb = w_props(P=0.1, T=Tamb+273.15).cp # P=1 bar  -> 0.1MPa, T=Tamb C, cp [kJ/kg·K]
    # m = Q/3600*w_props(P=0.1, T=Tin+273.15).rho # rho [kg/m³] # Convertir m^3/s a kg/s

    if Tout - Tin < 0.5:
        m = 0
        logger.warning('Negative or null temperature gradient. Flow set to 0')
    else:
        # Temporary removed: - H*(Tavg-Tamb)
        m = (Leq * beta * I - H * (Tavg - Tamb)) / (cp_avg / cf * (Tout - Tin))
        # m = (-rho*cp_avg*Acs*(Tout-Tout_ant)/period - beta*I + H/Leq*(Tavg-Tamb))/\
        #     (cf/cp_avg*Leq*(Tout_ant-Tin)) # kg/s, seguro? comprobar unidades

    # Tout = I * (beta)/( 1/Leq*(H/2+cp_Tin/cf*m) ) + \
    #        Tin * ( m-(H*cf)/(2*cp_Tin) )/( m+(H*cf)/(2*cp_Tin) ) + \
    #        Tamb * ( 2 )/( 1+(2*cp_Tamb)/(cf*H)*m ) # ºC

    # cp_Tout = w_props(P=0.1, T=Tout+273.15).cp # P=1 bar  -> 0.1MPa, T=Tout C, cp [KJ/kg·K]

    # Pth = m * cp_avg * (Tout-Tin) / 1000 # kWth

    Q = m * 3600 / rho_avg  # kg/s -> m^3/h

    Q = Q * 4 * np  # 4 loops

    # if Q>20: Q=20
    # if Q<0: Q=0

    return Q


def solar_field_model_temp(Tin, Q, I, Tamb, beta, H, nt=1, np=7 * 5, ns=2, Lt=1.15 * 20):  # , Acs=7.85e-5):
    """Steady state model of a flat plate collector solar field
       with any number of collectors in series and parallel.

    [1] G. Ampuño, L. Roca, M. Berenguel, J. D. Gil, M. Pérez, and J. E. Normey-Rico,
        “Modeling and simulation of a solar field based on flat-plate collectors,” Solar Energy,
        vol. 170, pp. 369–378, Aug. 2018, doi: 10.1016/j.solener.2018.05.076.


    Args:
        I (float): Solar radiation [W/m2]
        Tin (float): Inlet fluid temperature [ºC]
        Tout (float): Current outlet fluid temperature [ºC]
        Tout_ant (float): Prior outlet fluid temperature [ºC]
        Tamb (float): Ambient temperature [ºC]
        period (int): Time elapsed [s]
        nt (int, optional): Number of tubes in parallel per collector. Defaults to 1.
        np (int, optional): Number of collectors in parallel per loop. Defaults to 7.
        ns (int, optional): Number of loops in series. Defaults to 2.
        Lt (float, optional): Collector tube length [m]. Defaults to 97.
        H (float, optional): Thermal losses coefficient [J·s^-1·C^-1]. Defaults to 2.2.
        nt (int, optional): Number of tubes in parallel per collector. Defaults to 1.
        np (int, optional): Number of collectors in parallel per loop. Defaults to 7.
        ns (int, optional): Number of loops in series. Defaults to 2.
        Lt (float, optional): Collector tube length [m]. Defaults to 97.
        Acs (float, optional): Flat plate collector tube cross-section area [m²]. Defaults to 7.85e-5

    Returns:
        Q (float): Total volumetric solar field flow rate [m^3/s]
        Pgen (float): Power generated [kWth]
        SEC (float): Conversion factor / Specific energy consumption  [kWe/kWth]
    """

    def inner_function(Tout):

        error = abs(Tin + (beta * I - H * ((Tin + Tout) / 2 - Tamb)) / (m * cp / cf * 1 / Leq) - Tout)  # ºC

        return error

    Leq = ns * Lt
    cf = np * nt * 1  # Convertir m^3/h a kg/s y algo más
    w_props_avg = w_props(P=0.16, T=Tin + 273.15)  # P=1 bar  -> 0.1MPa, T=Tin C,
    cp = w_props_avg.cp * 1e3  # [kJ/kg·K] -> [J/kg·K]
    rho = w_props_avg.rho  # [kg/m³]
    m = Q * rho / 3600 / np  # m^3/h -> kg/s (entre número de captadores en paralelo)

    # cp_Tamb = w_props(P=0.1, T=Tamb+273.15).cp # P=1 bar  -> 0.1MPa, T=Tamb C, cp [kJ/kg·K]
    # m = Q/3600*w_props(P=0.1, T=Tin+273.15).rho # rho [kg/m³] # Convertir m^3/s a kg/s

    if Q < 0:
        Tout = Tin
        logger.warning('Negative or null temperature gradient. Flow set to 0')
    else:
        # Temporary removed: - H*(Tavg-Tamb)
        # bounds = (Tin, 110)
        Tout0 = Tin + (beta * I - H * (Tin - Tamb)) / (m * cp / cf * 1 / Leq)  # ºC
        Tout = fsolve(inner_function, Tout0)[0]  # ºC bounds=bounds

    return Tout