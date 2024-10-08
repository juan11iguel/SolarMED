from dataclasses import dataclass
from typing import Literal
import time
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from iapws import IAPWS97 as w_props
from loguru import logger

from solarmed_modeling.curve_fitting.curves import sigmoid_interpolation
from solarmed_modeling.metrics import calculate_metrics


dot = np.multiply
ts_pressure: float = 0.16  # MPa

supported_eval_alternatives: list[str] = ["standard", "constant-water-props"]
""" 
    - standard: Base model based on energy and mass balances
    - constant-water-props: Standard model but with constant water properties evaluated at some average temperature
"""

@dataclass
class ModelParameters:
    UA_h: list[float] | np.ndarray[float]  # Heat losses to the environment from the hot tank (W/K)
    V_h: list[float] | np.ndarray[float]  # Volume of each control volume of the hot tank (m³)
    UA_c: list[float] | np.ndarray[float]  # Heat losses to the environment from the cold tank (W/K)
    V_c: list[float] | np.ndarray[float]  # Volume of each control volume of the cold tank (m³)


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

    Ti: np.ndarray = fsolve(model_function, initial_guess) - 273.15  # K -> ºC

    # Tt = ( Tamb*( UA**2+UA*cp_Tbin*(msrc+2*mdis) ) + Tt_in*(msrc*cp_Ttin*(UA+cp_Tbin*(msrc+mdis))) + Tb_in*(mdis**2*cp_Tbin**2) )/ \
    #      ( UA**2+UA*(msrc+mdis)*(cp_Tbin+cp_Ttin)+(msrc+mdis)**2*cp_Ttin*cp_Tbin - msrc*mdis*cp_Ttin*cp_Tbin ) # ºC

    # Tb = (Tt*msrc*cp_Ttin + Tb_in*mdis*cp_Tbin + Tamb*UA)/(UA + (msrc+mdis)*cp_Tbin) # ºC

    if calculate_energy:
        return Ti, calculate_stored_energy(Ti, V_i, Tmin)

    return Ti

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
    ts: int, Tmin: float = 60, V=30,
    water_props: tuple[w_props, w_props] = None,
    calculate_energy=False
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
            UA=UA_c, V_i=Vi_c, N=3, ts=ts, calculate_energy=False,
            water_props=w_props_c
        )  # ºC

        Ti_h = thermal_storage_model_single_tank(
            Ti_ant_h, Tt_in=Tt_in, Tb_in=Ti_c[-1], Tamb=Tamb,
            mt_in=msrc, mb_in=mdis - msrc, mt_out=mdis, mb_out=0,
            UA=UA_h, V_i=Vi_h, N=3, ts=ts, calculate_energy=False,
            water_props=w_props_h
        )  # ºC!!

    else:
        # Recirculation from hot to cold
        # print('from hot to cold!')
        Ti_h = thermal_storage_model_single_tank(
            Ti_ant_h, Tt_in=Tt_in, Tb_in=0, Tamb=Tamb,
            mt_in=msrc, mb_in=0, mt_out=mdis, mb_out=msrc - mdis,
            UA=UA_h, V_i=Vi_h, N=3, ts=ts, calculate_energy=False,
            water_props=w_props_h
        )  # ºC!!

        Ti_c = thermal_storage_model_single_tank(
            Ti_ant_c, Tt_in=Ti_h[-1], Tb_in=Tb_in, Tamb=Tamb,
            mt_in=msrc - mdis, mb_in=mdis, mt_out=0, mb_out=msrc,
            UA=UA_c, V_i=Vi_c, N=3, ts=ts, calculate_energy=False,
            water_props=w_props_c
        )  # ºC!!


    if calculate_energy:    
        E_avail_h = calculate_stored_energy(Ti_h, Vi_h, Tmin)
        E_avail_c = calculate_stored_energy(Ti_c, Vi_c, Tmin)

        return Ti_h, Ti_c, E_avail_h, E_avail_c

    return Ti_h, Ti_c


# TODO: Copied from solar field. Adapt for thermal storage
def evaluate_model(
    df: pd.DataFrame, sample_rate: int, model_params: ModelParameters,
    alternatives_to_eval: list[Literal["standard", "constant-water-props"]] = supported_eval_alternatives,
    log_iteration: bool = False,
    Th_labels: list[str] = ['Tts_h_t', 'Tts_h_m', 'Tts_h_b'],
    Tc_labels: list[str] = ['Tts_c_t', 'Tts_c_m', 'Tts_c_b']
) -> tuple[list[pd.DataFrame], list[dict[str, str | dict[str, float]]]]:
    
    """
    Evaluate the thermal storage model using different alternatives and calculate performance metrics.

    Args:
        df: DataFrame containing the input data for the model.
        sample_rate: Sampling rate in seconds.
        model_params: ModelParameters object containing the model parameters.
        alternatives_to_eval: List of alternatives to evaluate. Supported alternatives are "standard", and "constant-water-props".
        log_iteration: Boolean flag to log each iteration.

    Raises:
        ValueError: If an unsupported alternative is provided in alternatives_to_eval.

    Returns:
        tuple: A tuple containing a list of DataFrames with the model outputs and a list of dictionaries with the performance metrics.
    """
    
    assert all([alt in supported_eval_alternatives for alt in alternatives_to_eval])

    idx_start = 0
    N = len(Th_labels)


    # Experimental (reference) outputs, used later in performance metrics evaluation
    out_ref = np.concatenate((df[Th_labels].values[idx_start:], df[Tc_labels].values[idx_start:]), axis=1)
    # out_ref = np.concatenate((df[Th_labels].values[idx_start+1:], df[Tc_labels].values[idx_start+1:]), axis=1)

    # Initialize particular variables for earch alternative that requires it
    water_props = None
    if "constant-water-props" in alternatives_to_eval:
        water_props: tuple[w_props, w_props] = (
            w_props(P=0.2, T=90 + 273.15), # P=2 bar  -> 0.2MPa, T in K, average working temperature of hot tank
            w_props(P=0.2, T=65 + 273.15)  # P=2 bar  -> 0.2MPa, T in K, average working temperature of cold tank
        )

    # Initialize result vectors
    outs_mod: list[np.ndarray[float]] = [np.zeros((len(df) - idx_start, N*2), dtype=float) for _ in alternatives_to_eval]

    stats = []

    for alt_idx, alt_id in enumerate(alternatives_to_eval):
        out = outs_mod[alt_idx]
        out[0] = np.array( [df.iloc[idx_start][T] for T in Th_labels + Tc_labels] )
        
        logger.info(f"Starting evaluation of alternative {alt_id}. Sample rate = {sample_rate} s")
        
        # Evaluate model
        start_time_alt = time.time()
        for i in range(idx_start + 1, len(df)):
            ds = df.iloc[i]
            j = i - idx_start
            start_time = time.time()
            
            if alt_id == "standard":
                out_h, out_c = thermal_storage_two_tanks_model(
                    Ti_ant_h=out[j-1][:N], Ti_ant_c=out[j-1][N:], # ºC, ºC
                    Tt_in=ds["Tts_h_in"],  # ºC
                    Tb_in=ds["Tts_c_in"],  # ºC
                    Tamb=ds["Tamb"],  # ºC

                    qsrc=ds["qts_src"],  # m³/h
                    qdis=ds["qts_dis"],  # m³/h

                    UA_h=model_params.UA_h,  # W/K
                    UA_c=model_params.UA_c,  # W/K
                    Vi_h=model_params.V_h,  # m³
                    Vi_c=model_params.V_c,  # m³
                    ts=sample_rate, Tmin=60,  # seg, ºC
                    water_props=None,
                )
            elif alt_id == "constant-water-props":
                out_h, out_c = thermal_storage_two_tanks_model(
                    Ti_ant_h=out[j-1][:N], Ti_ant_c=out[j-1][N:], # ºC, ºC
                    Tt_in=ds["Tts_h_in"],  # ºC
                    Tb_in=ds["Tts_c_in"],  # ºC
                    Tamb=ds["Tamb"],  # ºC

                    qsrc=ds["qts_src"],  # m³/h
                    qdis=ds["qts_dis"],  # m³/h

                    UA_h=model_params.UA_h,  # W/K
                    UA_c=model_params.UA_c,  # W/K
                    Vi_h=model_params.V_h,  # m³
                    Vi_c=model_params.V_c,  # m³
                    ts=sample_rate, Tmin=60,  # seg, ºC
                    water_props=water_props,
                )
            else:
                raise ValueError(
                    f"Unsupported alternative {alt_id}, options are: {supported_eval_alternatives}"
                )
                
            out[j] = np.concatenate((out_h, out_c), axis=0)
            elapsed_time = time.time() - start_time


            if log_iteration:
                logger.info(
                    f"[{alt_id}] Iteration {i} / {len(df)}. Elapsed time: {elapsed_time:.5f} s. Error: {abs(out[j]-out_ref[j]):.2f}"
                )
        
        elapsed_time = time.time() - start_time_alt
                
        # Calculate performance metrics
        stats.append({
            "test_id": df.index[0].strftime("%Y%m%d"),
            "alternative": alt_id,
            "metrics": calculate_metrics(out, out_ref), 
            "elapsed_time": elapsed_time,
            "average_elapsed_time": elapsed_time / (len(df) - idx_start),
            "model_parameters": model_params.__dict__,
            "sample_rate": sample_rate
        })
        
        logger.info(f"Finished evaluation of alternative {alt_id}. Elapsed time: {elapsed_time:.3f} s, MAE: {stats[-1]['metrics']['MAE']:.2f} ºC")

        # l[i] = calculate_total_pipe_length(q=df.iloc[i:i-idx_start:-1]['qsf'].values, n=60, sample_time=10, equivalent_pipe_area=7.85e-5)

        # df['Tsf_l2_pred'] = Tsf_out_mod

        # Estimate pipe length
        # l[l>1].mean() # 6e7

        dfs: list[pd.DataFrame] = [
            pd.DataFrame(out, columns=Th_labels + Tc_labels, index=df.index[idx_start:])
            for out in outs_mod
        ]

    return dfs, stats
