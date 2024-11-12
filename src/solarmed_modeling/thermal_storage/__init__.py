from dataclasses import dataclass
import numpy as np
from scipy.optimize import fsolve
from iapws import IAPWS97 as w_props
from loguru import logger

from solarmed_modeling.curve_fitting.curves import sigmoid_interpolation

dot = np.multiply
ts_pressure: float = 0.16  # MPa

supported_eval_alternatives: list[str] = ["standard", "constant-water-props"]
""" 
    - standard: Base model based on energy and mass balances
    - constant-water-props: Standard model but with constant water properties evaluated at some average temperature
"""

@dataclass
class ModelParameters:
    UA_h: tuple[float] | np.ndarray[float] = (0.0069818 , 0.00584034, 0.03041486) # Heat losses to the environment from the hot tank (W/K)
    V_h: tuple[float] | np.ndarray[float]  = (5.94771006, 4.87661781, 2.19737023) # Volume of each control volume of the hot tank (m³)
    UA_c: tuple[float] | np.ndarray[float] = (0.01396848, 0.0001    , 0.02286885) # Heat losses to the environment from the cold tank (W/K)
    V_c: tuple[float] | np.ndarray[float]  = (5.33410037, 7.56470594, 0.90547187) # Volume of each control volume of the cold tank (m³)
    
@dataclass
class FixedModelParameters:
    qts_src_min: float = 0.95  # Minimum flow rate [m³/h]
    qts_src_max: float = 20    # Maximum flow rate [m³/h]
    Tmin: float = 10 # Minimum temperature [ºC]
    Tmax: float = 120 # Maximum temperature [ºC]
    

def calculate_stored_energy(Ti: np.ndarray[float], V_i: np.ndarray[float], Tmin: float) -> float:
    # T in ºC, V_i in m³ 
    # Perform polynomial interpolation for temperatures
    interp_volumes = np.linspace(np.min(V_i), np.max(V_i), num=100)  # Generate interpolated volume points
    interp_temperatures = sigmoid_interpolation(V_i, Ti, interp_volumes)
    
    # Calculate the energy stored above Tmin
    # Estimate specific heat capacity (cp) and density (rho)
    cp  = np.array([w_props(T=T+273.15, P=ts_pressure).cp if T>=Tmin else 0 for T in interp_temperatures])
    rho = np.array([w_props(T=T+273.15, P=ts_pressure).rho if T>=Tmin else 0 for T in interp_temperatures])
    
    # Calculate the temperature differences above Tmin
    temperature_diff = np.maximum(0, interp_temperatures - Tmin)
    
    # Calculate the energy
    energy = np.sum(interp_volumes * rho * cp * temperature_diff)/3600 # m³·kg/m³·KJ/KgK·K == kWh

    return energy

def thermal_storage_model_single_tank(
        Ti_ant: np.ndarray[float],
        Tt_in: float | list[float],
        Tb_in: float | list[float],
        Tamb: float,
        mt_in: float | list[float],
        mb_in: float | list[float],
        mt_out: float,
        mb_out: float,
        UA: np.ndarray,
        V_i: np.ndarray,
        ts, N:int = 3, Tmin:float = 60, calculate_energy=False,
        water_props: w_props = None,
) -> np.ndarray[float] | tuple[np.ndarray[float], float]:

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
        water_props (w_props, optional): Water properties for the tank. Defaults to None.

    Returns:
        Ti: List of temperatures at each control volume [List of ºC]
        energy: Only if calculate_energy == True. Useful energy stored in the
            tank (reference Tmin) [kWh]
    """
    # TODO: Once available in iapws and water props are not provided, use w_props.from_list to get water properties

    def model_function(x):

        # Ti = x+273.15 # K
        Ti = x

        # if any(Ti < Tmin_) or any(Ti > Tmax_):
        #     # Return large error values if temperature limits are violated
        #     return [1e6] * N
        # if np.sum(V_i) > 1.1*V or np.sum(V_i) < 0.9*V:
        #     # Return large error values if total volume limits are violated
        #     return [1e6] * N

        eqs = [None for _ in range(N)]

        if water_props is not None:
            cp_i = [water_props.cp] * N
            rho_i = [water_props.rho] * N
        else:
            try:
                w_props_i = [w_props(P=ts_pressure, T=ti) for ti in Ti]
                cp_i = [w.cp for w in w_props_i]  # [KJ/kg·K]
                rho_i = [w.rho for w in w_props_i]  # [kg/m³]
            except NotImplementedError:
                print(f'Attempted inputs: {Ti}')
                raise

        # Volumen i
        for i in range(1, N - 1):
            eqs[i] = (- rho_i[i] * V_i[i] * cp_i[i] * (Ti[i] - Ti_ant[i]) / ts +  # Temperature change within the volume
                      np.sum(mt_in) * cp_i[i - 1] * Ti[i - 1] - mt_out * cp_i[i] * Ti[i] +  # Recirculation with upper volumne
                      - mb_out * cp_i[i] * Ti[i] + np.sum(mb_in) * cp_i[i + 1] * Ti[i+1] +  # Recirculation with lower volume
                      - UA[i] * (Ti[i] - Tamb))  # Losses to the environment

        # Volumen superior
        eqs[0] = (- rho_i[0] * V_i[0] * cp_i[0] * (Ti[0] - Ti_ant[0]) / ts +  # Temperature change within the volume
                  np.sum(dot(dot(mt_in, cp_Ttin), Tt_in)) - mt_out * cp_i[0] * Ti[0] +  # External input
                  - np.sum(mt_in) * cp_i[0] * Ti[0] + mt_out * cp_i[1] * Ti[1] +  # Recirculation with lower volume
                  - UA[0] * (Ti[0] - Tamb))  # Losses to the environment

        # Volumen inferior
        eqs[-1] = (- rho_i[-1] * V_i[-1] * cp_i[-1] * (
                    Ti[-1] - Ti_ant[-1]) / ts +  # Temperature change within the volume
                   np.sum(dot(dot(mb_in, cp_Tbin), Tb_in)) - mb_out * cp_i[-1] * Ti[-1] +  # External input
                   + mb_out * cp_i[-2] * Ti[-2] - np.sum(mb_in) * cp_i[-1] * Ti[-1] +  # Recirculation with upper volumne
                   - UA[-1] * (Ti[-1] - Tamb))  # Losses to the environment

        return eqs

    # Tmin_ = 273.15  # K
    # Tmax_ = 623.15  # K

    # Initial checks
    # if len(Ti_ant) != N:
    #     raise Exception('Ti_ant must have the same length as N')

    # if len(V_i) != N:
    #     raise Exception('Vi must have the same length as N')

    # if isinstance(Tt_in, list):
    #     if len(Tt_in) != len(mt_in):
    #         raise Exception('Tt_in must have the same length as mt_in')

    # if isinstance(Tb_in, list):
    #     if len(Tb_in) != len(mb_in):
    #         raise Exception('Tb_in must have the same length as mb_in')

    # if np.any( np.diff(Ti_ant) ) > 0:
    #     raise Exception('Values of previous temperatures profile needs to be monotonically decreasing')

    # Check temperature is within limits
    # if any(Ti_ant > 120):
    #     raise ValueError(f'Temperature must be below {120} ºC')

    # Initialize variables
    
    Tamb = Tamb + 273.15  # K
    Ti_ant = Ti_ant + 273.15  # K

    # Make sure variables are lists
    Tt_in = Tt_in if isinstance(Tt_in, list) else [Tt_in]  
    Tb_in = Tb_in if isinstance(Tb_in, list) else [Tb_in]  # Make sure it's a list
    Tt_in = [t + 273.15 for t in Tt_in]  # C -> K
    Tb_in = [t + 273.15 for t in Tb_in]  # C -> K
    
    mt_in = mt_in if isinstance(mt_in, list) else [mt_in]  # Make sure it's a list
    mb_in = mb_in if isinstance(mb_in, list) else [mb_in]  # Make sure it's a list
    
    # P=1 bar-> 0.1 MPa, T=Tin C, cp [kJ/kg·K]
    if water_props is not None:
        cp_Ttin = [water_props.cp] * len(Tt_in)
        cp_Tbin = [water_props.cp] * len(Tb_in)
    else: 
        cp_Ttin = [w_props(P=ts_pressure, T=t).cp for t in Tt_in]
        cp_Tbin = [w_props(P=ts_pressure, T=t).cp for t in Tb_in]

    # V_i = V/N # Volumen de cada volumen de control

    initial_guess = Ti_ant

    # debug_result = model_function(initial_guess)[0]

    # if debug_result < -1000 or abs(debug_result-28.425790075780405) < 1e-1:
    #     Ti = initial_guess
    #     w_props_i = [w_props(P=ts_pressure, T=ti) for ti in Ti]
    #     cp_i = [w.cp for w in w_props_i]  # [KJ/kg·K]
    #     rho_i = [w.rho for w in w_props_i]  # [kg/m³]
    #
    #     temp_change = - rho_i[0] * V_i[0] * cp_i[0] * (Ti[0] - Ti_ant[0]) / ts
    #     external_input = np.sum(dot(dot(mt_in, cp_Ttin), Tt_in)) - mt_out * cp_i[0] * Ti[0]
    #     inner_circ = - np.sum(mt_in) * cp_i[0] * Ti[0] + mt_out * cp_i[1] * Ti[1]
    #     env_losses = - UA[0] * (Ti[0] - Tamb)
    #
    #     print(f"{debug_result:.2f}: {external_input:.2f} + {inner_circ:.2f} + {env_losses:.2f}") # Print the upper volume equation evaluation result

    Ti: np.ndarray[float] = fsolve(model_function, initial_guess) - 273.15  # K -> ºC
    Ti = np.maximum(Ti, Tmin) # Saturate in lower limit

    # Tt = ( Tamb*( UA**2+UA*cp_Tbin*(msrc+2*mdis) ) + Tt_in*(msrc*cp_Ttin*(UA+cp_Tbin*(msrc+mdis))) + Tb_in*(mdis**2*cp_Tbin**2) )/ \
    #      ( UA**2+UA*(msrc+mdis)*(cp_Tbin+cp_Ttin)+(msrc+mdis)**2*cp_Ttin*cp_Tbin - msrc*mdis*cp_Ttin*cp_Tbin ) # ºC

    # Tb = (Tt*msrc*cp_Ttin + Tb_in*mdis*cp_Tbin + Tamb*UA)/(UA + (msrc+mdis)*cp_Tbin) # ºC

    if calculate_energy:
        return Ti, calculate_stored_energy(Ti, V_i, Tmin)

    return Ti

def thermal_storage_two_tanks_model(
    Ti_ant_h: np.ndarray[float], Ti_ant_c: np.ndarray[float],
    Tt_in: float | list[float],
    Tb_in: float | list[float],
    Tamb: float,
    qsrc: float,
    qdis: float,
    model_params: ModelParameters = ModelParameters(),
    fixed_model_params: FixedModelParameters = FixedModelParameters(),
    sample_time: int = 300,
    water_props: tuple[w_props, w_props] = None,
    calculate_energy: bool = False
) -> tuple[np.ndarray[float], np.ndarray[float]] | tuple[np.ndarray[float], np.ndarray[float], float, float]:
    """
    Thermal storage steady state model

    Args:
        Ti_ant_h (List[Float]): List of previous temperatures in hot storage [ºC]
        Ti_ant_c (List[Float]): List of previous temperatures in cold storage [ºC]
        Tt_in (float): Inlet temperature to top of the tank after heat source [ºC]
        Tb_in (float): Inlet temperature to bottom of the tank after load [ºC]
        qsrc (float): Flow rate from heat source [m³/h]
        qdis (float): Flow rate to energy sink [m³/h]
        Tmin (float, optional): Useful temperature limit [ºC]. Defaults to 60.
        Tamb (float): Ambient temperature [ºC]
        UA_h (List[Float]): Losses to the environment, it depends on the total outer surface
            of the tanks and the heat transfer coefficient [W/K].
        UA_c (List[Float]): Losses to the environment, it depends on the total outer surface
            of the tanks and the heat transfer coefficient [W/K].
        Vi_h (List[Float]): Volume of each control volume in hot tank [m³]
        Vi_c (List[Float]): Volume of each control volume in cold tank [m³]
        V (float, optional): Total volume of the tank(s) [m³]. Defaults to 30.
        ts (int, optional): Sample rate [sec]. Defaults to 60.
        calculate_energy (bool, optional): Whether or not to calculate and return
            energy stored above Tmin. Defaults to False.
        water_props (tuple[w_props, w_props], optional): Tuple of water properties for the hot and cold tanks. Defaults to None.

    Returns:
        Ti_h: List of temperatures at each control volume in hot tank [List of ºC]
        Ti_c: List of temperatures at each control volume in cold tank [List of ºC]
        energy_h: Only if calculate_energy == True. Useful energy stored in the
            hot tank (above reference Tmin) [kWh]
        energy_c: Only if calculate_energy == True. Useful energy stored in the
            cold tank (above reference Tmin) [kWh]

    """
    mp = model_params
    fmp = fixed_model_params
    
    Nc: int = len(Ti_ant_c) # Number of cold tank discrete volumes
    Nh: int = len(Ti_ant_h) # Number of hot tank discrete volumes
    
    w_props_h, w_props_c = water_props if water_props is not None else (w_props(P=ts_pressure, T=Tt_in+273.15), 
                                                                        w_props(P=ts_pressure, T=Tb_in+273.15))
    
    # Convert qdis and qsrc from m³/h to kg/s
    msrc = qsrc * w_props_h.rho / 3600  # m³/h -> kg/s
    mdis = qdis * w_props_c.rho / 3600  # m³/h -> kg/s

    if mdis > msrc:
        # Recirculation from cold to hot
        # print('from cold to hot!')
        Ti_c = thermal_storage_model_single_tank(
            Ti_ant_c, Tt_in=0, Tb_in=Tb_in, Tamb=Tamb,
            mt_in=0, mb_in=mdis, mt_out=mdis - msrc, mb_out=msrc,
            UA=mp.UA_c, V_i=mp.V_c, N=Nc, ts=sample_time, calculate_energy=False,
            water_props=w_props_c, Tmin=fmp.Tmin
        )  # ºC

        Ti_h = thermal_storage_model_single_tank(
            Ti_ant_h, Tt_in=Tt_in, Tb_in=Ti_c[-1], Tamb=Tamb,
            mt_in=msrc, mb_in=mdis - msrc, mt_out=mdis, mb_out=0,
            UA=mp.UA_h, V_i=mp.V_h, N=Nh, ts=sample_time, calculate_energy=False,
            water_props=w_props_h, Tmin=fmp.Tmin
        )  # ºC!

    else:
        # Recirculation from hot to cold
        # print('from hot to cold!')
        Ti_h = thermal_storage_model_single_tank(
            Ti_ant_h, Tt_in=Tt_in, Tb_in=0, Tamb=Tamb,
            mt_in=msrc, mb_in=0, mt_out=mdis, mb_out=msrc - mdis,
            UA=mp.UA_h, V_i=mp.V_h, N=Nh, ts=sample_time, calculate_energy=False,
            water_props=w_props_h, Tmin=fmp.Tmin
        )  # ºC!!

        Ti_c = thermal_storage_model_single_tank(
            Ti_ant_c, Tt_in=Ti_h[-1], Tb_in=Tb_in, Tamb=Tamb,
            mt_in=msrc - mdis, mb_in=mdis, mt_out=0, mb_out=msrc,
            UA=mp.UA_c, V_i=mp.V_c, N=Nc, ts=sample_time, calculate_energy=False,
            water_props=w_props_c, Tmin=fmp.Tmin
        )  # ºC!!!


    if calculate_energy:    
        E_avail_h = calculate_stored_energy(Ti_h, mp.V_h, fmp.Tmin)
        E_avail_c = calculate_stored_energy(Ti_c, mp.V_c, fmp.Tmin)

        return Ti_h, Ti_c, E_avail_h, E_avail_c

    return Ti_h, Ti_c