from .curve_fitting.curves import sigmoid_interpolation
from iapws import IAPWS97 as w_props
import numpy as np
from scipy.optimize import fsolve

dot = np.multiply

def calculate_stored_energy(Ti, V_i, Tmin):
    # T in ºC, V_i in m³ 
    # Perform polynomial interpolation for temperatures
    interp_volumes = np.linspace(np.min(V_i), np.max(V_i), num=100)  # Generate interpolated volume points
    interp_temperatures = sigmoid_interpolation(V_i, Ti, interp_volumes)
    
    # Calculate the energy stored above Tmin
    # Estimate specific heat capacity (cp) and density (rho)
    cp  = np.array([w_props(T=T+273.15, P=0.1).cp if T>=Tmin else 0 for T in interp_temperatures])
    rho = np.array([w_props(T=T+273.15, P=0.1).rho if T>=Tmin else 0 for T in interp_temperatures])
    
    # Calculate the temperature differences above Tmin
    temperature_diff = np.maximum(0, interp_temperatures - Tmin)
    
    # Calculate the energy
    energy = np.sum(interp_volumes * rho * cp * temperature_diff)/3600 # m³·kg/m³·KJ/KgK·K == kWh

    return energy


def thermal_storage_model_single_tank(
        Ti_ant: np.ndarray,
        Tt_in: float | list[float],
        Tb_in: float | list[float],
        Tamb: float,
        mt_in: float | list[float],
        mb_in: float | list[float],
        mt_out: float,
        mb_out: float,
        UA: np.ndarray,
        V_i: np.ndarray,
        ts, N:int = 3, Tmin:float = 60, calculate_energy=False
):

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
            w_props_i = [w_props(P=0.16, T=ti) for ti in Ti]
        except NotImplementedError:
            print(f'Attempted inputs: {Ti}')

            raise

        cp_i = [w.cp for w in w_props_i]  # [KJ/kg·K]
        rho_i = [w.rho for w in w_props_i]  # [kg/m³]

        # Volumen i
        for i in range(1, N - 1):
            eqs[i] = (- rho_i[i] * V_i[i] * cp_i[i] * (Ti[i] - Ti_ant[i]) / ts +  # Cambio de temperatura en el volumen
                      np.sum(mt_in) * cp_i[i - 1] * Ti[i - 1] - mt_out * cp_i[i] * Ti[
                          i] +  # Recirculación con volumen superior
                      - mb_out * cp_i[i] * Ti[i] + np.sum(mb_in) * cp_i[i + 1] * Ti[
                          i + 1] +  # Recirculación con volumen inferior
                      - UA[i] * (Ti[i] - Tamb))  # Pérdidas al ambiente

        # Volumen superior
        eqs[0] = (- rho_i[0] * V_i[0] * cp_i[0] * (Ti[0] - Ti_ant[0]) / ts +  # Cambio de temperatura en el volumen
                  np.sum(dot(dot(mt_in, cp_Ttin), Tt_in)) - mt_out * cp_i[0] * Ti[0] +  # Aporte externo
                  - np.sum(mt_in) * cp_i[0] * Ti[0] + mt_out * cp_i[1] * Ti[1] +  # Recirculación con volumen inferior
                  - UA[0] * (Ti[0] - Tamb))  # Pérdidas al ambiente

        # Volumen inferior
        eqs[-1] = (- rho_i[-1] * V_i[-1] * cp_i[-1] * (
                    Ti[-1] - Ti_ant[-1]) / ts +  # Cambio de temperatura en el volumen
                   np.sum(dot(dot(mb_in, cp_Tbin), Tb_in)) - mb_out * cp_i[-1] * Ti[-1] +  # Aporte externo
                   + mb_out * cp_i[-2] * Ti[-2] - np.sum(mb_in) * cp_i[-1] * Ti[
                       -1] +  # Recirculación con volumen superior
                   - UA[-1] * (Ti[-1] - Tamb))  # Pérdidas al ambiente

        return eqs

    Tmin_ = 273.15  # K
    Tmax_ = 623.15  # K

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
    Tamb = Tamb + 273.15  # K
    Ti_ant = Ti_ant + 273.15  # K

    Tt_in = Tt_in if isinstance(Tt_in, list) else [Tt_in]  # Make sure it's a list
    mt_in = mt_in if isinstance(mt_in, list) else [mt_in]  # Make sure it's a list
    Tt_in = [t + 273.15 for t in Tt_in]  # K
    cp_Ttin = [w_props(P=0.1, T=t).cp for t in Tt_in]  # P=1 bar-> 0.1 MPa, T=Tin C, cp [kJ/kg·K]

    Tb_in = Tb_in if isinstance(Tb_in, list) else [Tb_in]  # Make sure it's a list
    mb_in = mb_in if isinstance(mb_in, list) else [mb_in]  # Make sure it's a list
    Tb_in = [t + 273.15 for t in Tb_in]  # K
    cp_Tbin = [w_props(P=0.1, T=t).cp for t in Tb_in]  # P=1 bar-> 0.1 MPa, T=Tin C, cp [kJ/kg·K]

    # V_i = V/N # Volumen de cada volumen de control

    initial_guess = Ti_ant
    Ti: np.ndarray = fsolve(model_function, initial_guess)

    # Tt = ( Tamb*( UA**2+UA*cp_Tbin*(msrc+2*mdis) ) + Tt_in*(msrc*cp_Ttin*(UA+cp_Tbin*(msrc+mdis))) + Tb_in*(mdis**2*cp_Tbin**2) )/ \
    #      ( UA**2+UA*(msrc+mdis)*(cp_Tbin+cp_Ttin)+(msrc+mdis)**2*cp_Ttin*cp_Tbin - msrc*mdis*cp_Ttin*cp_Tbin ) # ºC

    # Tb = (Tt*msrc*cp_Ttin + Tb_in*mdis*cp_Tbin + Tamb*UA)/(UA + (msrc+mdis)*cp_Tbin) # ºC

    if calculate_energy:
        return Ti - 273.15, calculate_stored_energy(Ti - 273.15, V_i, Tmin)

    else:
        return Ti - 273.15


def thermal_storage_two_tanks_model(
    Ti_ant_h: np.ndarray, Ti_ant_c: np.ndarray,
    Tt_in: float | list[float],
    Tb_in: float | list[float],
    Tamb: float,
    qsrc: float,
    qdis: float,
    UA_h: np.ndarray,
    UA_c: np.ndarray,
    Vi_c: np.ndarray,
    Vi_h: np.ndarray,
    ts: int, Tmin: float = 60, V=30, unified_output=False,
    calculate_energy=False
):
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

    Returns:
        Ti_h: List of temperatures at each control volume in hot tank [List of ºC]
        Ti_c: List of temperatures at each control volume in cold tank [List of ºC]
        energy_h: Only if calculate_energy == True. Useful energy stored in the
            hot tank (above reference Tmin) [kWh]
        energy_c: Only if calculate_energy == True. Useful energy stored in the
            cold tank (above reference Tmin) [kWh]

    """

    # Convert qdis and qsrc from m³/h to kg/s
    msrc = qsrc * w_props(P=0.16, T=Tt_in+273.15).rho / 3600  # m³/h -> kg/s
    mdis = qdis * w_props(P=0.16, T=Tb_in+273.15).rho / 3600  # m³/h -> kg/s

    if mdis > msrc:
        # print('from cold to hot!')
        # Recirculation from cold to hot
        Ti_c = thermal_storage_model_single_tank(
            Ti_ant_c, Tt_in=0, Tb_in=Tb_in, Tamb=Tamb,
            mt_in=0, mb_in=mdis, mt_out=mdis - msrc, mb_out=msrc,
            UA=UA_c, V_i=Vi_c, N=3, ts=ts, calculate_energy=False
        )  # ºC

        Ti_h = thermal_storage_model_single_tank(
            Ti_ant_h, Tt_in=Tt_in, Tb_in=Ti_c[-1], Tamb=Tamb,
            mt_in=msrc, mb_in=mdis - msrc, mt_out=mdis, mb_out=0,
            UA=UA_h, V_i=Vi_h, N=3, ts=ts, calculate_energy=False
        )  # ºC!!

    else:
        # print('from hot to cold!')
        # Recirculation from hot to cold
        Ti_h = thermal_storage_model_single_tank(
            Ti_ant_h, Tt_in=Tt_in, Tb_in=0, Tamb=Tamb,
            mt_in=msrc, mb_in=0, mt_out=mdis, mb_out=msrc - mdis,
            UA=UA_h, V_i=Vi_h, N=3, ts=ts, calculate_energy=False
        )  # ºC!!

        Ti_c = thermal_storage_model_single_tank(
            Ti_ant_c, Tt_in=Ti_h[-1], Tb_in=Tb_in, Tamb=Tamb,
            mt_in=msrc - mdis, mb_in=mdis, mt_out=0, mb_out=msrc,
            UA=UA_c, V_i=Vi_c, N=3, ts=ts, calculate_energy=False
        )  # ºC!!

    if calculate_energy:
        E_avail_h = calculate_stored_energy(Ti_h, Vi_h, Tmin)
        E_avail_c = calculate_stored_energy(Ti_c, Vi_c, Tmin)

        return Ti_h, Ti_c, E_avail_h, E_avail_c

    else:
        return Ti_h, Ti_c


def thermal_storage_model(Ti_ant: np.array, Tt_in, Tb_in, Tamb, msrc, mdis,
                          UA: np.ndarray,
                          V_i: np.ndarray,
                          N:int, ts:int, Tmin:float=60, calculate_energy=False):
    """ DEPRECATED! Thermal storage steady state model

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

        cp_i = [w.cp for w in w_props_i]  # [KJ/kg·K]
        rho_i = [w.rho for w in w_props_i]  # [kg/m³]

        # Volumen i
        for i in range(1, N - 1):
            eqs[i] = (- rho_i[i] * V_i[i] * cp_i[i] * (Ti[i] - Ti_ant[i]) / ts +  # Cambio de temperatura en el volumen
                      msrc * cp_i[i - 1] * Ti[i - 1] - mdis * cp_i[i] * Ti[i] +  # Recirculación con volumen superior
                      - msrc * cp_i[i] * Ti[i] + mdis * cp_i[i + 1] * Ti[i + 1] +  # Recirculación con volumen inferior
                      - UA[i] * (Ti[i] - Tamb))  # Pérdidas al ambiente

        # Volumen superior
        eqs[0] = (- rho_i[0] * V_i[0] * cp_i[0] * (Ti[0] - Ti_ant[0]) / ts +  # Cambio de temperatura en el volumen
                  msrc * cp_Ttin * Tt_in - mdis * cp_i[0] * Ti[0] +  # Aporte externo
                  - msrc * cp_i[0] * Ti[0] + mdis * cp_i[1] * Ti[1] +  # Recirculación con volumen inferior
                  - UA[0] * (Ti[0] - Tamb))  # Pérdidas al ambiente

        # Volumen inferior
        eqs[-1] = (- rho_i[-1] * V_i[-1] * cp_i[-1] * (
                Ti[-1] - Ti_ant[-1]) / ts +  # Cambio de temperatura en el volumen
                   mdis * cp_Tbin * Tb_in - msrc * cp_i[-1] * Ti[-1] +  # Aporte externo
                   + msrc * cp_i[-2] * Ti[-2] - mdis * cp_i[-1] * Ti[-1] +  # Recirculación con volumen superior
                   - UA[-1] * (Ti[-1] - Tamb))  # Pérdidas al ambiente

        return eqs

    Tmin_ = 273.15  # K
    Tmax_ = 623.15  # K

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
    Tt_in = Tt_in + 273.15  # K
    Tb_in = Tb_in + 273.15  # K
    Tamb = Tamb + 273.15  # K
    Ti_ant = Ti_ant + 273.15  # K

    w_props_Ttin = w_props(P=0.1, T=Tt_in)
    w_props_Tbin = w_props(P=0.1, T=Tb_in)

    cp_Ttin = w_props_Ttin.cp  # P=1 bar->0.1 MPa, T=Tin C, cp [kJ/kg·K]
    cp_Tbin = w_props_Tbin.cp  # P=1 bar->0.1 MPa, T=Tin C, cp [kJ/kg·K]

    # V_i = V/N # Volumen de cada volumen de control

    initial_guess = Ti_ant
    Ti = scipy.optimize.fsolve(model_function, initial_guess)

    # Tt = ( Tamb*( UA**2+UA*cp_Tbin*(msrc+2*mdis) ) + Tt_in*(msrc*cp_Ttin*(UA+cp_Tbin*(msrc+mdis))) + Tb_in*(mdis**2*cp_Tbin**2) )/ \
    #      ( UA**2+UA*(msrc+mdis)*(cp_Tbin+cp_Ttin)+(msrc+mdis)**2*cp_Ttin*cp_Tbin - msrc*mdis*cp_Ttin*cp_Tbin ) # ºC

    # Tb = (Tt*msrc*cp_Ttin + Tb_in*mdis*cp_Tbin + Tamb*UA)/(UA + (msrc+mdis)*cp_Tbin) # ºC

    if calculate_energy:
        return Ti - 273.15, calculate_stored_energy(Ti - 273.15, V_i, Tmin)

    else:
        return Ti - 273.15