import json
# test
import scipy
from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar
import math
from dataclasses import dataclass, field, asdict

import numpy as np
from model_calibrations.eletrical_consumption_fit import evaluate_fit
import logging
from utils.curve_fitting import polynomial_interpolation
from typing import Union, List
dot = np.multiply
from utils.validation import validate_input_types
import py_validate as pv # https://github.com/gfyoung/py-validate

# MATLAB MED model
import MED_model
import matlab

"""

PENDIENTE:
    - [ ] Ajuste de curvas de consumo eléctrico de las bombas
    - [ ] Implementar ecuaciones válvula tres vías
    - [ ] Implementar ecuaciones intercambiador
    - [ ] Actualizar modelo de campo solar para que calcule Msf en lugar de Tsf_out
    - [x] Meter modelo de válvula de tres vías
    - [ ] Actualizar modelo de tanques para que calcule temperatura actual en base a temperatura
          anterior, ahora mismo sólo es válida cuando Pin=Pout
    - [ ] Incluir modelo de MED de MATLAB
    - [ ] Versión nueva de modelo de MED directamente en Python
    - [ ] Incluir curvas de consumo de las bombas de MATLAB
    
    En solar_MED:
        - [ ] Método para actualizar parámetros (solar_MED.update_parameters())
        - [ ] Un millón de cosas más
        
    - [ ] Validar modelo de campo solar con datos experimentales
    - [ ] Validar modelo de tanques con datos experimentales
    - [ ] Validar modelo de intercambiador con datos experimentales
        
        
    Modelos dinámicos para control de bajo nivel:
    - [ ] Campo solar
    - [ ] Tanques?
    - [ ] Intercambiador?
        
"""

# Initialize logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


def solar_field_model(I, Tin, Tout, Tout_ant, Tamb, period, beta=0.0975, H=2.2, nt=50, np=7, ns=2, Lt=1.94, Acs=7.85e-5):
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
    
    w_props_avg = w_props(P=0.1, T=(Tin+Tout+Tout_ant)/3+273.15) # P=1 bar  -> 0.1MPa, T=Tin C, 
    cp_avg = w_props_avg.cp # [kJ/kg·K]
    rho = w_props_avg.rho # [kg/m³]
    
    # cp_Tamb = w_props(P=0.1, T=Tamb+273.15).cp # P=1 bar  -> 0.1MPa, T=Tamb C, cp [kJ/kg·K]
    # m = Q/3600*w_props(P=0.1, T=Tin+273.15).rho # rho [kg/m³] # Convertir m^3/s a kg/s
    
    Leq = ns*Lt
    cf = np*nt*1 # Convertir m^3/h a kg/s y algo más
    Tavg = (Tin+Tout+Tout_ant)/3
    
    m = (-rho*cp_avg*Acs*(Tout-Tout_ant)/period - beta*I + H/Leq*(Tavg-Tamb))/\
        (cf/cp_avg*Leq*(Tout_ant-Tin)) # kg/s, seguro? comprobar unidades
    
    # Tout = I * (beta)/( 1/Leq*(H/2+cp_Tin/cf*m) ) + \
    #        Tin * ( m-(H*cf)/(2*cp_Tin) )/( m+(H*cf)/(2*cp_Tin) ) + \
    #        Tamb * ( 2 )/( 1+(2*cp_Tamb)/(cf*H)*m ) # ºC
           
    # cp_Tout = w_props(P=0.1, T=Tout+273.15).cp # P=1 bar  -> 0.1MPa, T=Tout C, cp [KJ/kg·K]
           
    Pgen = m * cp_avg * (Tout-Tin) / 1000 # kWth
    
    Q = m*CONVERSION
    SEC = None
    # SEC = pump_electrical_consumption(Q, pump='solar_field') / Pgen # kWe/kWth
    
    return Q, Pgen, SEC

def heat_exchanger_model(Tp_in, Ts_in, Qp, Qs, UA=28000):
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
    
    w_props_Tp_in = w_props(P=0.1, T=Tp_in+273.15)
    w_props_Ts_in = w_props(P=0.1, T=Ts_in+273.15)
    cp_Tp_in = w_props_Tp_in.cp # P=0.1 bar->0.1 MPa, T=Tp_in C, cp [KJ/kg·K]
    cp_Ts_in = w_props_Ts_in.cp # P=0.1 bar->0.1 MPa, T=Ts_in C, cp [KJ/kg·K]
    mp = Qp/3600*w_props_Tp_in.rho # rho [kg/m³] # Convertir m^3/s a kg/s
    ms = Qs/3600*w_props_Ts_in.rho # rho [kg/m³] # Convertir m^3/s a kg/s
    
    mcp_min = min([mp*cp_Tp_in, ms*cp_Ts_in])
    mcp_max = max([mp*cp_Tp_in, ms*cp_Ts_in])
    
    theta = UA*(1/mcp_max-1/mcp_min)
    
    eta_p = (1-math.e**theta)/( 1-math.e**theta*(mcp_min/mcp_max) )
    eta_s = mp*cp_Tp_in/(ms*cp_Ts_in) 
    
    Tp_out = Tp_in - eta_p*(mcp_min)/(mp*cp_Tp_in)*(Tp_in-Ts_in) # ºC
    Ts_out = Ts_in + eta_s*(Tp_in-Tp_out) # ºC
    
    cp_Tp_out = w_props(P=0.1, T=Tp_out+273.15).cp # P=0.1 bar->0.1 MPa, T=Tp_out C, cp [kJ/kg·K]
    cp_Ts_out = w_props(P=0.1, T=Ts_out+273.15).cp # P=0.1 bar->0.1 MPa, T=Ts_out C, cp [kJ/kg·K]
    
    Pgen = mp*(cp_Tp_in+cp_Tp_out)/2*(Tp_in-Tp_out)/1000 # kWth
    Pabs = ms*(cp_Ts_in+cp_Ts_out)/2*(Ts_out-Ts_in)/1000 # kWth
    
    return Tp_out, Ts_out, Pgen, Pabs

def three_way_valve_model(Mdis, Tsrc, Tdis_in, Tdis_out):
    
    """Three way valve steady state model.
    
    Args:
        Mdis (float): Discharge flow rate [m^3/h or any]
        Tdis_in (float): Discharge / load inlet temperature [ºC]
        Tdis_out (float): Discharge / load outlet temperature [ºC]
        Tsrc (float): Source temperature [ºC]

    Returns:
        Msrc: Source flow rate [Same units as Mdis]
        R: Ratio of mixing, se puede usar, tras ajustar una curva de apertura de válvula,
           para implementar un controlador con feedforward
    """
    
    Tsrc     = Tsrc + 273.15
    Tdis_in  = Tdis_in + 273.15
    Tdis_out = Tdis_out + 273.15
    
    cp_src = w_props(P=0.1, T=Tsrc).cp # P=0.1 bar->0.1 MPa, T [K], cp [kJ/kg·K]
    cp_dis_in = w_props(P=0.1, T=Tdis_in).cp # P=0.1 bar->0.1 MPa, T [K], cp [kJ/kg·K]
    cp_dis_out = w_props(P=0.1, T=Tdis_out).cp # P=0.1 bar->0.1 MPa, T [K], cp [kJ/kg·K]
    
    # Msrc = Mdis * ( Tdis_in*cp_dis_in  - Tdis_out*cp_dis_out ) / ( Tsrc*cp_src - Tdis_out*cp_dis_out )
    R = (Tdis_in*cp_dis_in - Tsrc*cp_src) / (Tdis_out*cp_dis_out - Tsrc*cp_src)

    # Saturation
    if R>1: R=1
    elif R<0: R=0
        
    Msrc = Mdis*(1-R)
    
    return Msrc, R
    
    
def MED(Ms, Ts_in, Mf, Tcw_in, Tcw_out, t_operated):
    
    # return Mprod, Ts_out, Mcw
    
    pass


def thermal_storage_model(Ti_ant:np.array, Tt_in, Tb_in, Tamb, msrc, mdis, 
                          UA:np.array([0.00677724, 0.0039658 , 0.02872586, 0.00688542]), 
                          V_i:np.array([2.97217468, 1.71277001, 9.4345576 , 3.7807375 ]), 
                          N=4, ts=60, Tmin=60, V=30, calculate_energy=False):
    
    """ Thermal storage steady state model

    Args:
        Ti_ant (List[Float]): List of previous temperatures in storage [ºC]
        Tt_in (float): Inlet temperature to top of the tank after heat source [ºC]
        Tb_in (float): Inlet temperature to bottom of the tank after load [ºC]
        msrc (float): Flow rate from heat source [kg/s]
        mdis (float): Flow rate to energy sink [kg/s]
        Tmin (float, optional): Useful temperature limit [ºC]. Defaults to 60.
        Tamb (float): Ambient temperature [ºC]
        UA (List[Float]): Losses to the environment, it depends on the total outer surface
            of the tanks and the heat transfer coefficient [W/K].
        V_i (List[Float]): Volume of each control volume [m³]
        V (float, optional): Total volume of the tank(s) [m³]. Defaults to 30.
        ts (int, optional): Sample rate [sec]. Defaults to 60.
        N (int, optional): Number of control volumes. Defaults to 4.
        calculate_energy (bool, optional): Whether or not to calculate and return
            energy stored above Tmin. Defaults to False.

    Returns:
        Ti: List of temperatures at each control volume [List of ºC]
        energy: Only if calculate_energy == True. Useful energy stored in the 
            tank (reference Tmin) [kWh]
    """
    
    def model_function(x):
    
        # Ti = x+273.15 # K
        Ti = x
        
        if any(Ti < Tmin_) or any(Ti > Tmax_):
            # Return large error values if temperature limits are violated
            return [1e6] * N
        # if np.sum(V_i) > 1.1*V or np.sum(V_i) < 0.9*V:
        #     # Return large error values if total volume limits are violated
        #     return [1e6] * N            
            
        eqs = [None for _ in range(N)]
            
        try:
            w_props_i = [w_props(P=0.1, T=ti) for ti in Ti]
        except NotImplementedError:
            print(f'Attempted inputs: {Ti}')
            
            raise
            
        cp_i  = [w.cp for w in w_props_i]  # [KJ/kg·K]
        rho_i = [w.rho for w in w_props_i] # [kg/m³]
        
        # Volumen i
        for i in range(1, N-1):
            eqs[i] = (  - rho_i[i]*V_i[i]*cp_i[i]*(Ti[i]-Ti_ant[i])/ts +   # Cambio de temperatura en el volumen
                        msrc*cp_i[i-1]*Ti[i-1] - mdis*cp_i[i]*Ti[i] +      # Recirculación con volumen superior
                        - msrc*cp_i[i]*Ti[i] + mdis*cp_i[i+1]*Ti[i+1] +    # Recirculación con volumen inferior
                        - UA[i]*(Ti[i]-Tamb) )                             # Pérdidas al ambiente
                            
        # Volumen superior
        eqs[0] = (  - rho_i[0]*V_i[0]*cp_i[0]*(Ti[0]-Ti_ant[0])/ts +  # Cambio de temperatura en el volumen
                    msrc*cp_Ttin*Tt_in - mdis*cp_i[0]*Ti[0] +         # Aporte externo
                    - msrc*cp_i[0]*Ti[0] + mdis*cp_i[1]*Ti[1] +       # Recirculación con volumen inferior
                    - UA[0]*(Ti[0]-Tamb) )                            # Pérdidas al ambiente
                    
        # Volumen inferior
        eqs[-1] = ( - rho_i[-1]*V_i[-1]*cp_i[-1]*(Ti[-1]-Ti_ant[-1])/ts + # Cambio de temperatura en el volumen
                    mdis*cp_Tbin*Tb_in - msrc*cp_i[-1]*Ti[-1] +           # Aporte externo
                    + msrc*cp_i[-2]*Ti[-2] - mdis*cp_i[-1]*Ti[-1] +       # Recirculación con volumen superior
                    - UA[-1]*(Ti[-1]-Tamb) )                              # Pérdidas al ambiente
        
        return eqs
    
    Tmin_ = 273.15 # K
    Tmax_ = 623.15 # K
    
    # Initial checks
    if len(Ti_ant) != N:
        raise Exception('Ti_ant must have the same length as N')
        
    if len(V_i) != N:
        raise Exception('Vi must have the same length as N')
    
    # if np.any( np.diff(Ti_ant) ) > 0:
    #     raise Exception('Values of previous temperatures profile needs to be monotonically decreasing')
    
    # Check temperature is within limits
    # if any(Ti_ant > 120):
    #     raise ValueError(f'Temperature must be below {120} ºC')
    
    # Initialize variables
    Tt_in = Tt_in + 273.15 # K
    Tb_in = Tb_in + 273.15 # K
    Tamb  = Tamb  + 273.15 # K
    Ti_ant = Ti_ant + 273.15 # K
    
    w_props_Ttin = w_props(P=0.1, T=Tt_in)
    w_props_Tbin = w_props(P=0.1, T=Tb_in)
    
    cp_Ttin = w_props_Ttin.cp # P=1 bar->0.1 MPa, T=Tin C, cp [kJ/kg·K]
    cp_Tbin = w_props_Tbin.cp # P=1 bar->0.1 MPa, T=Tin C, cp [kJ/kg·K]
    
    # V_i = V/N # Volumen de cada volumen de control
    
    initial_guess = Ti_ant
    Ti = scipy.optimize.fsolve(model_function, initial_guess)
    
    
    # Tt = ( Tamb*( UA**2+UA*cp_Tbin*(msrc+2*mdis) ) + Tt_in*(msrc*cp_Ttin*(UA+cp_Tbin*(msrc+mdis))) + Tb_in*(mdis**2*cp_Tbin**2) )/ \
    #      ( UA**2+UA*(msrc+mdis)*(cp_Tbin+cp_Ttin)+(msrc+mdis)**2*cp_Ttin*cp_Tbin - msrc*mdis*cp_Ttin*cp_Tbin ) # ºC
         
    # Tb = (Tt*msrc*cp_Ttin + Tb_in*mdis*cp_Tbin + Tamb*UA)/(UA + (msrc+mdis)*cp_Tbin) # ºC
    
    if calculate_energy:
        return Ti-273.15, calculate_stored_energy(Ti-273.15, V_i, Tmin)
    
    else:
        return Ti-273.15
    
def thermal_storage_model_single_tank(Ti_ant:np.ndarray, 
                                      Tt_in: Union[float, List[float]], 
                                      Tb_in: Union[float, List[float]], 
                                      Tamb: float, 
                                      mt_in: Union[float, List[float]], 
                                      mb_in: Union[float, List[float]], 
                                      mt_out: float, 
                                      mb_out: float, 
                                      UA:np.ndarray=np.array([0.00677724, 0.0039658 , 0.02872586]), 
                                      V_i:np.ndarray=np.array([2.97217468, 1.71277001, 9.4345576]), 
                                      N=3, ts=60, Tmin=60, V=15, calculate_energy=False):
    
    """ Thermal storage steady state model

    Args:
        Ti_ant (List[Float]): List of previous temperatures in storage [ºC]
        Tt_in (float): Inlet temperature to top of the tank after heat source [ºC]
        Tb_in (float): Inlet temperature to bottom of the tank after load [ºC]
        msrc (float): Flow rate from heat source [kg/s]
        mdis (float): Flow rate to energy sink [kg/s]
        Tmin (float, optional): Useful temperature limit [ºC]. Defaults to 60.
        Tamb (float): Ambient temperature [ºC]
        UA (List[Float]): Losses to the environment, it depends on the total outer surface
            of the tanks and the heat transfer coefficient [W/K].
        V_i (List[Float]): Volume of each control volume [m³]
        V (float, optional): Total volume of the tank(s) [m³]. Defaults to 30.
        ts (int, optional): Sample rate [sec]. Defaults to 60.
        N (int, optional): Number of control volumes. Defaults to 4.
        calculate_energy (bool, optional): Whether or not to calculate and return
            energy stored above Tmin. Defaults to False.

    Returns:
        Ti: List of temperatures at each control volume [List of ºC]
        energy: Only if calculate_energy == True. Useful energy stored in the 
            tank (reference Tmin) [kWh]
    """
    
    def model_function(x):
    
        # Ti = x+273.15 # K
        Ti = x
        
        if any(Ti < Tmin_) or any(Ti > Tmax_):
            # Return large error values if temperature limits are violated
            return [1e6] * N
        # if np.sum(V_i) > 1.1*V or np.sum(V_i) < 0.9*V:
        #     # Return large error values if total volume limits are violated
        #     return [1e6] * N            
            
        eqs = [None for _ in range(N)]
            
        try:
            w_props_i = [w_props(P=0.1, T=ti) for ti in Ti]
        except NotImplementedError:
            print(f'Attempted inputs: {Ti}')
            
            raise
            
        cp_i  = [w.cp for w in w_props_i]  # [KJ/kg·K]
        rho_i = [w.rho for w in w_props_i] # [kg/m³]
        
        # Volumen i
        for i in range(1, N-1):
            eqs[i] = (  - rho_i[i]*V_i[i]*cp_i[i]*(Ti[i]-Ti_ant[i])/ts +           # Cambio de temperatura en el volumen
                        np.sum(mt_in)*cp_i[i-1]*Ti[i-1] - mt_out*cp_i[i]*Ti[i] +   # Recirculación con volumen superior
                        - mb_out*cp_i[i]*Ti[i] + np.sum(mb_in)*cp_i[i+1]*Ti[i+1] + # Recirculación con volumen inferior
                        - UA[i]*(Ti[i]-Tamb) )                                     # Pérdidas al ambiente
                            
        # Volumen superior
        eqs[0] = (  - rho_i[0]*V_i[0]*cp_i[0]*(Ti[0]-Ti_ant[0])/ts +                  # Cambio de temperatura en el volumen
                    np.sum( dot(dot(mt_in,cp_Ttin), Tt_in) ) - mt_out*cp_i[0]*Ti[0] + # Aporte externo
                    - np.sum(mt_in)*cp_i[0]*Ti[0] + mt_out*cp_i[1]*Ti[1] +            # Recirculación con volumen inferior
                    - UA[0]*(Ti[0]-Tamb) )                                            # Pérdidas al ambiente
                    
        # Volumen inferior
        eqs[-1] = ( - rho_i[-1]*V_i[-1]*cp_i[-1]*(Ti[-1]-Ti_ant[-1])/ts +              # Cambio de temperatura en el volumen
                    np.sum( dot(dot(mb_in,cp_Tbin),Tb_in) ) - mb_out*cp_i[-1]*Ti[-1] + # Aporte externo
                    + mb_out*cp_i[-2]*Ti[-2] - np.sum(mb_in)*cp_i[-1]*Ti[-1] +         # Recirculación con volumen superior
                    - UA[-1]*(Ti[-1]-Tamb) )                                           # Pérdidas al ambiente
        
        return eqs
    
    Tmin_ = 273.15 # K
    Tmax_ = 623.15 # K
    
    # Initial checks    
    if len(Ti_ant) != N:
        raise Exception('Ti_ant must have the same length as N')
        
    if len(V_i) != N:
        raise Exception('Vi must have the same length as N')
        
    if isinstance(Tt_in, list):
        if len(Tt_in) != len(mt_in):
            raise Exception('Tt_in must have the same length as mt_in')
            
    if isinstance(Tb_in, list):
        if len(Tb_in) != len(mb_in):
            raise Exception('Tb_in must have the same length as mb_in')
    
    # if np.any( np.diff(Ti_ant) ) > 0:
    #     raise Exception('Values of previous temperatures profile needs to be monotonically decreasing')
    
    # Check temperature is within limits
    # if any(Ti_ant > 120):
    #     raise ValueError(f'Temperature must be below {120} ºC')
    
    # Initialize variables
    Tamb  = Tamb  + 273.15 # K
    Ti_ant = Ti_ant + 273.15 # K
    
    Tt_in = Tt_in if isinstance(Tt_in, list) else [Tt_in] # Make sure it's a list
    mt_in = mt_in if isinstance(mt_in, list) else [mt_in] # Make sure it's a list
    Tt_in = [t+273.15 for t in Tt_in] # K
    cp_Ttin = [w_props(P=0.1, T=t).cp for t in Tt_in] # P=1 bar-> 0.1 MPa, T=Tin C, cp [kJ/kg·K]
    
    Tb_in = Tb_in if isinstance(Tb_in, list) else [Tb_in] # Make sure it's a list
    mb_in = mb_in if isinstance(mb_in, list) else [mb_in] # Make sure it's a list
    Tb_in = [t+273.15 for t in Tb_in] # K
    cp_Tbin = [w_props(P=0.1, T=t).cp for t in Tb_in] # P=1 bar-> 0.1 MPa, T=Tin C, cp [kJ/kg·K]
    
    # V_i = V/N # Volumen de cada volumen de control
    
    initial_guess = Ti_ant
    Ti = scipy.optimize.fsolve(model_function, initial_guess)
    
    
    # Tt = ( Tamb*( UA**2+UA*cp_Tbin*(msrc+2*mdis) ) + Tt_in*(msrc*cp_Ttin*(UA+cp_Tbin*(msrc+mdis))) + Tb_in*(mdis**2*cp_Tbin**2) )/ \
    #      ( UA**2+UA*(msrc+mdis)*(cp_Tbin+cp_Ttin)+(msrc+mdis)**2*cp_Ttin*cp_Tbin - msrc*mdis*cp_Ttin*cp_Tbin ) # ºC
         
    # Tb = (Tt*msrc*cp_Ttin + Tb_in*mdis*cp_Tbin + Tamb*UA)/(UA + (msrc+mdis)*cp_Tbin) # ºC
    
    if calculate_energy:
        return Ti-273.15, calculate_stored_energy(Ti-273.15, V_i, Tmin)
    
    else:
        return Ti-273.15

def thermal_storage_model_two_tanks(Ti_ant_h:np.ndarray, Ti_ant_c:np.ndarray, 
                                    Tt_in: Union[float, List[float]], 
                                    Tb_in: Union[float, List[float]], 
                                    Tamb: float, 
                                    msrc: float, 
                                    mdis: float, 
                                    UA_h:np.array([0.00677724, 0.0039658 , 0.02872586]), 
                                    UA_c:np.array([0.00677724, 0.0039658 , 0.02872586]),
                                    Vi_h:np.array([2.97217468, 1.71277001, 9.4345576]), 
                                    Vi_c:np.array([2.97217468, 1.71277001, 9.4345576]),
                                    ts=60, Tmin=60, V=30, unified_output=False,
                                    calculate_energy=False):
    
    if mdis-msrc > 0:
        # Recirculation from cold to hot
        Ti_c = thermal_storage_model_single_tank(
                                     Ti_ant_c, Tt_in=0, Tb_in=Tb_in, Tamb=Tamb, 
                                     mt_in=0, mb_in=mdis, mt_out=mdis-msrc, mb_out=msrc,
                                     UA=UA_c, V_i=Vi_c, 
                                     N=3, ts=ts, calculate_energy=False) # ºC!!
        
        Ti_h = thermal_storage_model_single_tank(
                                     Ti_ant_h, Tt_in=Tt_in, Tb_in=Ti_c[-1], Tamb=Tamb, 
                                     mt_in=msrc, mb_in=mdis-msrc, mt_out=mdis, mb_out=0, 
                                     UA=UA_h, V_i=Vi_h, 
                                     N=3, ts=ts, calculate_energy=False) # ºC!!
    else:
        # Recirculation from hot to cold
        Ti_h = thermal_storage_model_single_tank(
                                     Ti_ant_h, Tt_in=Tt_in, Tb_in=0, Tamb=Tamb, 
                                     mt_in=msrc, mb_in=0, mt_out=mdis, mb_out=msrc-mdis, 
                                     UA=UA_h, V_i=Vi_h, 
                                     N=3, ts=ts, calculate_energy=False) # ºC!!

        Ti_c = thermal_storage_model_single_tank(
                                     Ti_ant_c, Tt_in=Ti_h[-1], Tb_in=Tb_in, Tamb=Tamb, 
                                     mt_in=msrc-mdis, mb_in=mdis, mt_out=0, mb_out=msrc,
                                     UA=UA_c, V_i=Vi_c, 
                                     N=3, ts=ts, calculate_energy=False) # ºC!!
        
    if calculate_energy:
        E_avail_h = calculate_stored_energy(Ti_h, Vi_h, Tmin)
        E_avail_c = calculate_stored_energy(Ti_c, Vi_c, Tmin)
        return Ti_h, Ti_c, E_avail_h, E_avail_c
    
    else:
        return Ti_h, Ti_c
    
    
@dataclass
class med_storage_model:
    """
    Model of the Multi-effect distillation plant and the thermal storage system.
    
    It includes several functions:
        -  
    """
    
    # States (need to be initialized)
    ## Thermal storage
    Tts_h: List[float] # Hot tank temperature profile (ºC)
    Tts_c: List[float] # Cold tank temperature profile (ºC)
    
    
    # Parameters
    ts: float = 60 # Sample rate (seg)
    # cost_e:float = None # Cost of electricty (€/kWhe)
    # cost_w = None # Sale price of water (€/m³)
    curve_fits_path: str = 'datos/curve_fits.json' # Path to the file with the curve fits
    
    ## MED
    # Pumps to calculate SEEC_med, must be in the same order as in step method
    med_pumps = ["brine_electrical_consumption", "feedwater_electrical_consumption", 
                 "prod_electrical_consumption", "cooling_electrical_consumption", 
                 "hotwater_electrical_consumption"]
    
    ## Thermal storage
    UAts_h: List[float] = field(default_factory=lambda: [0.0069818 , 0.00584034, 0.03041486]) # Heat losses to the environment from the hot tank (W/K)
    UAts_c: List[float] = field(default_factory=lambda: [0.01396848, 0.0001    , 0.02286885]) # Heat losses to the environment from the cold tank (W/K)
    Vts_h: List[float]  = field(default_factory=lambda: [5.94771006, 4.87661781, 2.19737023]) # Volume of each control volume of the hot tank (m³)
    Vts_c: List[float]  = field(default_factory=lambda: [5.33410037, 7.56470594, 0.90547187]) # Volume of each control volume of the cold tank (m³)
    
    
    # Decision variables
    ## MED
    # Tmed_s_in: float  = None # MED hot water inlet temperature (ºC)
    # Tmed_c_out: float = None # MED condenser outlet temperature (ºC)
    # mmed_s: float = None # MED hot water flow rate (m³/h)
    # mmed_f: float = None # MED feedwater flow rate (m³/h)
    
    ## Thermal storage
    # mts_src: float # Thermal storage heat source flow rate (m³/h)
    
    # Environment
    # Tamb: float = None # Ambient temperature (ºC)
    # # I: float # Solar irradiance (W/m2)
    # Tmed_c_in: float = None # Default 20 # Seawater temperature (ºC)
    # wmed_f: float = None # Default 35 # Seawater / MED feedwater salinity (g/kg)
    
    # # Outputs
    # ## Thermal storage
    # Tts_h_t: float  = None # Temperature of the top of the hot tank (ºC)
    # Tts_t_in: float = None # Temperature of the heating fluid to top of hot tank (ºC)
    # mts_dis: float  = None # Thermal storage discharge flow rate (m³/h)
    
    # ## MED
    # mmed_c: float = None # MED condenser flow rate (m³/h)
    # Tmed_s_out: float = None # MED hot water outlet temperature (ºC)
    # STEC_med: float = None # MED specific thermal energy consumption (kWh/m³)
    # SEEC_med: float = None # MED specific electrical energy consumption (kWh/m³)
    # mmed_d: float = None # MED distillate flow rate (m³/h)
    # mmed_b: float = None # MED brine flow rate (m³/h)
    # Emed_e: float = None # MED electrical power consumption (kW)
    
    # ## Three-way valve
    # R_3wv: float = None # Three-way valve mix ratio (-)
    
    # Limits
    ## Decision variables
    ### MED
    Tmed_s_in_max: float = 90 # ºC, maximum temperature of the hot water inlet, changes dynamically with Tts_h_t
    Tmed_s_in_min: float = 60 # ºC, minimum temperature of the hot water inlet, operational limit
    
    # Tmed_c_out_max: float = 50 # ºC, maximum temperature of the condenser outlet, depends on Tmed_c_in, Mmed_c_min, Mmed_d and Pmed_c
    # Tmed_c_out_min: float = 12 # ºC, maximum temperature of the condenser outlet, depends on Tmed_c_in, Mmed_c_min, Mmed_d and Pmed_c
    
    mmed_s_max: float = 14.8*3.6 # m³/h, maximum hot water flow rate
    mmed_s_min: float = 5.56*3.6 # m³/h, minimum hot water flow rate
    
    mmed_f_max: float = 9 # m³/h, maximum feedwater flow rate
    mmed_f_min: float = 5 # m³/h, minimum feedwater flow rate
    
    mmed_d_max: float = 3.2 # m³/h, maximum distillate flow rate
    mmed_d_min: float = 1.2 # m³/h, minimum distillate flow rate
    
    mmed_b_max: float = 6   # m³/h, maximum brine flow rate
    mmed_b_min: float = 1.2 # m³/h, minimum brine flow rate
    
    mmed_c_max: float = 21 # m³/h, maximum condenser flow rate
    mmed_c_min: float = 8  # m³/h, minimum condenser flow rate
    
    ### Thermal storage
    mts_src_max: float = 8.48 # m³/h, maximum thermal st. heat source / heat ex. secondary flow rate
    mts_src_min: float = 0    # m³/h, (ZONA MUERTA: 1.81) minimum thermal st. heat source / heat ex. secondary flow rate    
    
    ## Environment
    Tamb_max: float = 50 # ºC, maximum ambient temperature
    Tamb_min: float = -15 # ºC, minimum ambient temperature
    Tmed_c_in_max: float = 28 # ºC, maximum temperature of the condenser inlet cooling water / seawater
    Tmed_c_in_min: float = 10 # ºC, minimum temperature of the condenser inlet cooling water / seawater
    wmed_f_min: float = 30 # g/kg, minimum salinity of the seawater / MED feedwater
    wmed_f_max: float = 90 # g/kg, maximum salinity of the seawater / MED feedwater
    Imin: float = 0 # W/m2, minimum solar irradiance
    Imax: float = 2000 # W/m2, maximum solar irradiance
    
    ## Outputs / others
    ### MED
    mmed_c_min: float = 10 # m³/h, minimum condenser flow rate
    mmed_c_max: float = 21 # m³/h, maximum condenser flow rate

    def __setattr__(self, name, value):
        """Input validation. Check inputs are within the allowed range.
        """
        
        # Keep dataclass default input validation
        super().__setattr__(name, value)
        
        # MED
        if name == "Tmed_s_in":
            if (value < self.Tmed_s_in_min or 
                value > self.Tmed_s_in_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Tmed_s_in_min}, {self.Tmed_s_in_max}] ({value})")
        
        # elif name == "Tmed_s_out":
        #     if (value < self.Tmed_s_out_min or 
        #         value > self.Tmed_s_out_max or
        #         not isinstance(value, (int,float))):
        #         raise ValueError(f"Value of {name} must be a number within: [{self.Tmed_s_out_min}, {self.Tmed_s_out_max}] ({value})")
        
        elif name == "mmed_s":
            if (value < self.mmed_s_min or 
                value > self.mmed_s_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.mmed_s_min}, {self.mmed_s_max}] ({value})")
        
        elif name == "mmed_f":
            if (value < self.mmed_f_min or 
                value > self.mmed_f_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.mmed_f_min}, {self.mmed_f_max}] ({value})")
        
        elif name == "mmed_c":
            if (value < self.mmed_c_min or 
                value > self.mmed_c_max or
                value > self.mmed_f or
                not isinstance(value, (int,float))):
                raise ValueError(f"""Value of {name} must be a number within: [{self.mmed_c_min}, {self.mmed_c_max}] ({value}) 
                                 and greater than mmed_f ({self.mmed_f})""")
        
        # Thermal storage
        elif name == "mts_src":
            if (value < self.mts_src_min or 
                value > self.mts_src_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.mts_src_min}, {self.mts_src_max}] ({value})")
        
        elif name == "Tts_h":
            if np.any( np.array(value) < 0):
                raise ValueError(f"Elements of {name} must be greater than zero. It's freezing out here! ({value})")
            if hasattr(self, 'Tmed_s_in'):
                if value[0] < self.Tmed_s_in:
                    raise ValueError(f"Value of {name}_t ({value[0]}) must be greater than Tmed,s,in ({self.Tmed_s_in})")
           
            # Make sure it's a numpy array
            if not isinstance(self.Tts_h, np.ndarray):
                self.Tts_h = np.array(self.Tts_h)
        
        elif name == "Tts_c":
            if np.any( np.array(value) < 0):
                raise ValueError(f"Elements of {name} must be greater than zero. It's freezing out here! ({value})")
            
            # Make sure it's a numpy array
            if not isinstance(self.Tts_c, np.ndarray):
                self.Tts_c = np.array(self.Tts_c)
                
        elif name in ["UAts_h", "UAts_c"]:
            # To check whether all elements in list are floats
            if ( isinstance(value, list) or isinstance(value, np.ndarray) ):
                
                if not set(map(type, value)) == {float}:
                    raise TypeError(f'All elements of {name} must be floats')
                
                if np.any( np.array(value) < 0) or np.any( np.array(value) > 1):
                    raise ValueError(f"Elements of {name} must be a number within: [{0}, {1}] ({value})")
            else:
                 raise TypeError(f'{name} must be either a list of floats or a numpy array')
        
        # Environment
        elif name == "Tmed_c_in":
            if (value < self.Tmed_c_in_min or 
                value > self.Tmed_c_in_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Tmed_c_in_min}, {self.Tmed_c_in_max}] ({value})")
        
        elif name == "Tamb":
            if (value < self.Tamb_min or 
                value > self.Tamb_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Tamb_min}, {self.Tamb_max}] ({value})")
        
        elif name == "HR":
            if (value < 0 or 
                value > 100 or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{0}, {100}] ({value})")
        
        elif name == "wmed_f":
            if (value < self.wmed_f_min or 
                value > self.wmed_f_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.wmed_f_min}, {self.wmed_f_max}] ({value})")
        
        elif name == "I":
            if (value < self.Imin or 
                value > self.Imax or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Imin}, {self.Imax}] ({value})")
        
        
        
        # If the value is within the allowed range, set the attribute
        super().__setattr__(name, value)
    
    def __post_init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Filter matplotlib logging
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        
        # Initalize MATLAB MED model
        self.MED_model = MED_model.initialize()
        self.logger.info('MATLAB MED model initialized')
        
        # Load electrical consumption fit curves
        try:
            with open(self.curve_fits_path, 'r') as file:
                    self.fit_config = json.load(file)
            self.logger.debug(f'Curve fits file loaded from {self.curve_fits_path}')
        except FileNotFoundError:
            self.logger.error(f'Curve fits file not found in {self.curve_fits_path}')
            raise
            
        self.logger.debug('Initialization completed')
         
        
        
    # def __post_init__(self):
        
    #     # Update limits that depend on other variables
    #     self.update_limits(step_done=False)
    # def check_limits(self):
    #     """
    #     Check if the current values are within the allowed range for each variable
    #     """
    #     # Iterate over all the attributes of the class and set them to their 
    #     # own value to trigger __setattr__ validation
    #     for attr in self.__dict__.keys():
    #         setattr(self, attr, getattr(self, attr))
                
    # def update_limits(self):
    #     """ Method to update the limits that depend on other variables 
    #     (dynamic restrictions)
        
    #     """
        
    #     if self.step_done:
    #         # Update limits that depend on other variables and can only be 
    #         # calculated after the step
            
    #         pass
    #     else:
    #         # Update limits that depend on other variables and can be 
    #         # calculated before the step
    #         pass
        
    #     # Check limits
    #     self.check_limits()
    
    # @pv.validate_inputs(a="number", mmed_s="number", mmed_f="number", Tmed_s_in="number", Tmed_c_out="number", 
    #                     Tmed_c_in="number", wmed_f="number", mts_src="number", Tamb="number")
    def step(self, mmed_s:float, mmed_f:float, Tmed_s_in:float, Tmed_c_out:float, 
             Tmed_c_in:float, wmed_f:float, mts_src:float, Tts_t_in:float, Tamb:float):
        
        """Calculate model outputs given current environment variables and decision variables

            Inputs:
                - Decision variables
                    + mmed_s (m³/h): Heat source flow rate 
                    + mmed_f (m³/h): Feed water flow rate
                    + Tmed_s_in (ºC): Heat source inlet temperature
                    + Tmed_c_out (ºC): Cooling water outlet temperature
                     
                - Environment variables
                    + Tmed_c_in (ºC): Seawater temperature
                    + wmed_f (g/kg): Seawater salinity
                    + Tamb (ºC): Ambient temperature
        Returns:
            _type_: _description_
        """
        
        # Update limits that depend on other variables and can be calculated before the step
        # self.step_done = False; self.update_limits()
        
        # Update class decision and environment variables values
        self.mmed_s = mmed_s
        self.mmed_f = mmed_f
        self.Tmed_s_in = Tmed_s_in
        self.Tmed_c_out = Tmed_c_out
        self.Tmed_c_in = Tmed_c_in
        self.wmed_f = wmed_f
        self.mts_src = mts_src
        self.Tts_t_in = Tts_t_in
        self.Tamb = Tamb
        
        # Make sure thermal storage state is a numpy array
        self.Tts_h = np.array(self.Tts_h)
        self.Tts_c = np.array(self.Tts_c)
        
        # MED
        MsIn = matlab.double([mmed_s/3.6], size=(1, 1)) # m³/h -> L/s
        TsinIn = matlab.double([Tmed_s_in], size=(1, 1))
        MfIn = matlab.double([mmed_f], size=(1, 1))
        TcwoutIn = matlab.double([Tmed_c_out], size=(1, 1))
        TcwinIn = matlab.double([Tmed_c_in], size=(1, 1))
        op_timeIn = matlab.double([0], size=(1, 1))
        # wf=wmed_f # El modelo sólo es válido para una salinidad así que ni siquiera 
        # se considera como parámetro de entrada
        self.mmed_d, self.Tmed_s_out, self.mmed_c, _, _ = self.MED_model.MED_model(MsIn,     # m³/h
                                                                                TsinIn,   # ºC
                                                                                MfIn,     # m³/h
                                                                                TcwoutIn, # ºC
                                                                                TcwinIn,  # ºC
                                                                                op_timeIn,# hours
                                                                                nargout=5 )
        
        ## Brine flow rate
        self.mmed_b = self.mmed_f - self.mmed_d # m³/h
        
        ## MED electrical consumption
        Emed_e = 0
        for flow, pump in zip([self.mmed_b, self.mmed_f, self.mmed_d, 
                               self.mmed_c, self.mmed_s], self.med_pumps):
            Emed_e = Emed_e + self.electrical_consumption(flow, self.fit_config[pump]) # kWhe
        
        self.Emed_e = Emed_e
        self.SEEC_med = Emed_e / self.mmed_d # kWhe/m³
        
        ## MED STEC
        w_props_s = w_props(P=0.1, T=(Tmed_s_in + self.Tmed_s_out)/2+273.15)
        cp_s = w_props_s.cp # kJ/kg·K
        rho_s = w_props_s.rho # kg/m³
        # rho_d = w_props(P=0.1, T=Tmed_c_out+273.15) # kg/m³
        mmed_s_kgs = mmed_s * rho_s / 3600 # kg/s
        
        self.STEC_med = mmed_s_kgs * (Tmed_s_in - self.Tmed_s_out) * cp_s / self.mmed_d # kWhth/m³
        
        # Three-way valve
        self.mts_dis, self.R_3wv = three_way_valve_model(Mdis=mmed_s, Tsrc=self.Tts_h[0], 
                                                         Tdis_in=Tmed_s_in, Tdis_out=self.Tmed_s_out)
        
        # Thermal storage
        self.Tts_h, self.Tts_c = thermal_storage_model_two_tanks(Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c, # [ºC], [ºC]
                                                                 Tt_in = self.Tts_t_in,                    # ºC
                                                                 Tb_in = self.Tmed_s_out,                  # ºC 
                                                                 Tamb = Tamb,                              # ºC
                                                                 msrc = mts_src,                           # m³/h
                                                                 mdis = self.mts_dis,                      # m³/h
                                                                 UA_h = self.UAts_h,                       # W/K
                                                                 UA_c = self.UAts_c,                       # W/K
                                                                 Vi_h = self.Vts_h,                        # m³
                                                                 Vi_c = self.Vts_c,                        # m³
                                                                 ts=self.ts, Tmin=self.Tmed_s_in_min)      # seg, ºC
        
        # Update limits that depend on other variables and can only be calculated after the step
        # self.step_done=True; self.update_limits()
        
        # Store inputs as properties to have them available in get_properties
        # Done last to make sure that any model call is not using a class property
        # instead of method input
        # mmed_s, mmed_f, Tmed_s_in, Tmed_c_out, Tmed_c_in, wmed_f, mts_src, Tamb
        self.mmed_s = mmed_s
        self.mmed_f = mmed_f
        self.Tmed_s_in = Tmed_s_in
        self.Tmed_c_out = Tmed_c_out
        self.Tmed_c_in = Tmed_c_in
        self.wmed_f = wmed_f
        self.mts_src = mts_src
        self.Tamb = Tamb
                    
        return self.mmed_d, self.STEC_med, self.SEEC_med
        
    def electrical_consumption(self, Q, fit_config):
        """Returns the electrical consumption (kWe) of a pump given its 
           flow rate in m^3/h.

        Args:
            Q (float): Volumentric flow rate [m³/h]
            
        Retunrs:
            power (float): Electrical power consumption [kWe]
        """
        
        power = evaluate_fit(
            x=Q, fit_type=fit_config['best_fit'], params=fit_config['params'][ fit_config['best_fit'] ]
        )
        
        return power
    
    def calculate_fixed_costs(self, ):
        # TODO
        # costs_fixed = cost_med + cost_ts + cost_sf
        return 0
    
    
    def calculate_cost(self, cost_w=None, cost_e=None):
        
        self.cost_w = cost_w if cost_w else self.cost_w # €/m³
        self.cost_e = cost_e if cost_e else self.cost_e # €/kWhe
        
        # Operational costs (€/m³h)
        self.cost_op = cost_e * ( self.SEEC_med + self.STEC_med*self.SEC_sf )
        
        # Fixed costs (€/h)
        if not hasattr(self, 'cost_fixed'):
            self.cost_fixed = self.calculate_fixed_costs() 
        
        self.cost = ( (self.cost_e-self.cost_op)*self.mmed_d + self.cost_fixed ) * self.ts/3600
        
        return self.cost

    def get_properties(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        output = vars(self).copy()
        
        # Filter some properties
        output.pop('fit_config') if 'fit_config' in output else None
        output.pop('curve_fits_path') if 'curve_fits_path' in output else None
        output.pop('MED_model') if 'MED_model' in output else None
        output.pop('logger') if 'logger' in output else None
        
        return output

    def terminate(self):
        """
        

        Returns
        -------
        None.

        """
        
        self.MED_model.terminate()
    
# @dataclass
# class solarMED:
#     """
#     Model of the complete system for case study 3, this includes:
#         - Solar flat-plate collector field
#         - Heat exchanger
#         - Thermal storage
#         - Three-way valve
#         - Multi-effect distillation plant
        
        
#     """
    
#     # Inputs
    
#     # MED
#     # Tmed_s_in_sp: float # MED hot water inlet temperature (ºC)
#     # Tmed_cw_out_sp: float # MED condenser outlet temperature (ºC)
#     # Mmed_s_sp: float # MED hot water flow rate (m^3/h)
#     # Mmed_f_sp: float # MED feed water flow rate (m^3/h)
    
#     # # Solar field
#     # Tsf_out_sp: float # Solar field outlet temperature (ºC)
    
#     # # Thermal storage
#     # Mtk_src_sp: float # Thermal storage heating flow rate (m^3/h)
    
#     # Environment variables
#     # Xf: float # Water salinity (g/kg)
#     # Tcwin: float # Seawater inlet temperature (ºC)
#     # Tamb: float # Ambient temperature (ºC)
#     # I: float # Solar radiation (W/m^2)
    
#     # Model outputs
#     # - Solar field
#     Tsf_in: float # Solar field inlet temperature (ºC)
    
#     # - Thermal storage
#     Tts_t_in: float # Inlet temperature to top of the tank after heat source (ºC)
#     Tts_b_out: float # Outlet temperature from bottom of the tank to heat source (ºC)
    
    
#     # Parameters
#     ts: int = 600 # Sample rate (s)
#     Ts_min: float = 50 # Minimum temperature of the hot water inlet temperature (ºC)
    
#     # - Costs
#     Ce: float # Electricity cost (€/kWh_e)
#     thermal_self_production: bool = True # Whether thermal energy is by a self-owned solar field or provided externally
#     Cw: float # Water sale price (€/m^3)
    
#     # - Fixed costs
#     cost_investment_MED: float = 0 # Investment cost of the MED (€)
#     cost_investment_solar_field: float = 0 # Investment cost of the MED (€)
#     cost_investment_storage: float = 0 # Investment cost of the MED (€)
#     # amortizacion, etc
    
#     # - Solar field
#     SF_beta: float = 0.0975 # Irradiance model parameter (m) # Pendiente de cambiar
#     SF_H: float = 2.2 # Thermal losses coefficient for the loop (W/ºC)
#     SF_nt: int = 1 # Number of parallel tubes per collector
#     SF_np: int = 7 # Number of parallel collectors in each loop
#     SF_ns: int = 2 # Number of serial connections of collector rows
#     SF_Lt: float = 1.94 # Length of the collector inner tube (m)
    
#     # - Thermal storage
#     TS_UA: float = 28000 # Heat transfer coefficient of the thermal storage (W/ºC)
#     TS_V: float = 30 # Total volume of the tank(s) [m^3] 
    
#     # - MED
#     med_pumps = ['brine_electrical_consumption', 'feedwater_electrical_consumption', 'prod_electrical_consumption', 'cooling_electrical_consumption', 'heatsource_electrical_consumption']
    
#     # curve_fits: dict # Dictionary with the curve fits for the electrical consumption of the pumps
#     curve_fits_path: str = 'curve_fits.json' # Path to the file with the curve fits for the electrical consumption of the pumps
    
#     # Outputs
#     # Mmed_d: float # MED distillate flow rate (m^3/h)
#     # SEEC_MED: float # Specific electric energy consumption of the MED (kWh_e/m^3)
#     # STEC_MED: float # Specific thermal energy consumption of the MED (kWh_th/m^3)
#     # SEC_SF: float # Specific electric energy consumption of the solar field (kWh_e/kWh_th)
    
#     # def __post_init__(self):
#     #     # Check initial values for attributes are within allowed limits
#     #     pass
    
#     # def __setattr__(self, name, value):
#     #     # Check values for attributes every time they are updated are within allowed limits
#     #     if name == 'value' and value < 5:
#     #         raise ValueError('Value must be greater than 5')
#     #     super().__setattr__(name, value)
    
#     def __init__(self):
        
#         # Load curve fits for the electrical consumption of the pumps
#         try:
#             with open(self.curve_fits_path, 'r') as file:
#                     self.curve_fits = json.load(file)
#         except FileNotFoundError:
#             logger.error(f'Curve fits file not found in {self.curve_fits_path}')
#             raise
        
                
#     def electrical_consumption(Q, fit_config=None):
#         """Returns the electrical consumption (kWe) of a pump given the flow rate in m^3/h.

#         Args:
#             Q (float): Volumentric flow rate [m^3/h]
#         """
        
#         power = evaluate_fit( x=Q, fit_type=fit_config['best_fit'], params=fit_config['params'][fit_config['best_fit']] )
        
#         return power
        
    
#     def calculate_fixed_costs(self):
#         pass
    
#     def get_thermal_energy_cost(self):
#         pass
        
#         # return self.cost_investment_MED
    
#     def calculate_cost(self, Mmed_d, SEEC_MED, STEC_MED, SEC_SF):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
        
#         if self.thermal_self_production:
#             Cop = self.Ce*( SEEC_MED + SEC_SF*STEC_MED )
#         else:
#             Cop = self.Ce*SEEC_MED + self.get_thermal_energy_cost()*STEC_MED
            
#         Cfixed = self.calculate_fixed_costs()
        
#         return (self.Cw - Cop)*Mmed_d - Cfixed
    
    
#     def update(self, I, Tcwin, Tamb, Xf, Tmed_s_in, Tmed_cw_out, Mmed_s, Mmed_f, Tsf_out, Mts_src):
#         """Calculates the outputs of the system based on the current state of the environment
#         variables, the new setpoints and the last state of the system.

#         Args:
#             I (_type_): _description_
#             Tcwin (_type_): _description_
#             Tamb (_type_): _description_
#             Xf (_type_): _description_
#             Tmed_s_in (_type_): _description_
#             Tmed_cw_out (_type_): _description_
#             Mmed_s (_type_): _description_
#             Mmed_f (_type_): _description_
#             Tsf_out (_type_): _description_
#             Mtk_src (_type_): _description_
            
#         Returns:
#             _type_: _description_
#         """
#         # Check inputs are within allowed limits
#         # Solar irradiance
        
#         # Cooling water inlet temperature
        
#         # Ambient temperature
        
#         # Water salinity
        
#         # MED hot water inlet temperature
        
#         # MED condenser outlet temperature
        
#         # MED hot water flow rate
        
#         # MED feed water flow rate
        
#         # Solar field outlet temperature
        
#         # Thermal storage heating flow rate
        
        
#         # Solar field
#         Msf = solar_field(I, self.Tsf_in, Tsf_out, Tamb, beta=self.SF_beta, H=self.SF_H, 
#                           nt=self.SF_nt, np=self.SF_np, ns=self.SF_ns, Lt=self.SF_Lt)
        
#         # Heat exchanger
#         self.Tsf_in, self.Tts_t_in, Phx_in, Phx_out = heat_exchanger(Tp_in=Tsf_out, Ts_in=self.Tts_b_out, 
#                                                                      Qp=Msf, Qs=Mts_src, UA=self.HX_UA)
        
#         SEC_SF = Phx_in / self.electrical_consumption(Msf, fit_config=self.curve_fits['solarfield_electrical_consumption'])
                
#         # Thermal storage
#         # Por actualizar
#         self.Tts_t, self.Tts_b, self.Ets_net = thermal_storage(self.Tts_t_in, self.Tmed_s_out, Mts_src, self.M3wv, 
#                                                                self.Ts_min, Tamb, UA=self.TS_UA, V=self.TS_V)
        
#         # Three-way valve
#         self.M3wv = three_way_valve(Mmed_s, Tmed_s_in, self.Tmed_s_out, self.Tts_t)
        
#         # MED
#         Mmed_d, self.Tmed_s_out, Mmed_c, STEC_MED = MED(Ms=Tmed_s_in, Ts_in=Tmed_s_in, Mf=Mmed_f, Tcw_out=Tmed_cw_out)
#         Mmed_b = Mmed_f - Mmed_d
        
#         med_power_e = 0
#         for flow, pump in zip([Mmed_b, Mmed_f, Mmed_d, Mmed_c, Mmed_s], self.med_pumps):
#             med_power_e = med_power_e + self.electrical_consumption(flow, fit_config=self.curve_fits[pump])
        
#         SEEC_MED = med_power_e / Mmed_d

#         return Mmed_d, SEEC_MED, STEC_MED, SEC_SF
