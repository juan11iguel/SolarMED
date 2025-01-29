#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:47:44 2023

@author: patomareao
"""

import numpy as np
import pandas as pd
from iapws import IAPWS97 as w_props
import os
from utils.constants import vars_info
from utils import filter_nan, get_Q_from_3wv_model

# Select working directory
os.chdir(os.path.join(os.getenv("HOME"), "Nextcloud/Juanmi_MED_PSA/EURECAT/solarMED_modeling"))

#%% Functions

#%% Test thermal storage model

from models_psa import thermal_storage_model    
from parameters_fit import calculate_iae, calculate_ise, calculate_itae
from visualization.calibrations import (plot_model_result_thermal_storage, 
                                   plot_energy_thermal_storage,
                                   plot_energy_evolution_thermal_storage)

save_figure = True
figure_path = '/home/patomareao/Nextcloud/Juanmi_MED_PSA/EURECAT/Modelos/attachments'
figure_name = 'result_model_ts'


var_names = {v["signal_id"]: k for k, v in vars_info.items() if 'ts' in k or not '_' in k}

Tin_labels = ["Tts_h_t", "Tts_h_m", "Tts_c_t", "Tts_c_m"]
data_label_prefix = "datos_tanques"

# Parameters
N = 4 # Number of volumes
V = 30 # Total volume of the storage, m³
Tmin = 60 # Minimum useful temperature, °C
ts = 60*5 # Sample rate (seg)
sample_rate_str = '5Min'

# Model parameters
UA  = np.array([0.0068, 0.004, 0.0287, 0.0069])
V_i = np.array([2.9722, 1.7128, 9.4346, 3.7807])

for datos_name in ["20230621", "20230414"]:
    
    # Import data
    data = pd.read_csv(f'datos/{data_label_prefix}_{datos_name}.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
    
    data = data.rename(columns=var_names)
    data = data.resample(sample_rate_str, on='time').mean()
    
    # Remove rows with NaN values in place and generate a warning
    # data = filter_nan(data)

    # Inputs
    # Flow meter of Qdis is broken, using the output of the three-way valve 
    # model as an alternative
    data_Qdis = get_Q_from_3wv_model(datos_name, sample_rate_str=sample_rate_str)
    
    # Merge both dataframes so they are synced in time
    data.rename(columns={'m_ts_dis': 'm_ts_dis_sensor'}, inplace=True) # Rename the invalid signal
    data = pd.merge(data, data_Qdis, left_index=True, right_index=True, how='outer')
        
    # Remove rows with NaN values in place and generate a warning
    data = filter_nan(data)

    # Since we are using the first output as starting point, start from the second value
    Ti_ant = np.array( [data[T][0] for T in Tin_labels] )
    Tt_in = data.Tts_t_in.values[1:]
    Tb_in = data.Tts_b_in.values[1:] if 'Tts_b_in' in data.columns else np.zeros(len(data)-1)
    
    Tamb = data.Tamb.values[1:]
    Qsrc = data.m_ts_src.values[1:] if 'm_ts_src' in data.columns else np.zeros(len(data)-1)
    Qdis = data.m_ts_dis.values[1:] if 'm_ts_dis' in data.columns else np.zeros(len(data)-1)
    
    
    msrc = np.zeros(len(data)-1, dtype=float); mdis = np.zeros(len(data)-1, dtype=float)
    for idx in range(1, len(data)-1):
        msrc[idx] = Qsrc[idx]/60*w_props(P=0.1, T=Tt_in[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/min a kg/s
        mdis[idx] = Qdis[idx]*w_props(P=0.1, T=data.Tts_h_t[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/s a kg/s
    
    # Experimental outputs
    Ti_ref = data[Tin_labels].values[1:]
    
    # Initialize result vectors
    Ti_mod   = np.zeros((len(data)-1, N), dtype=float)
    energies = np.zeros(len(data)-1, dtype=float)
    Ti_ant = Ti_ref[0]
    
    # Evaluate model
    for idx in range(len(data)-1):
        Ti_mod[idx], energies[idx] = thermal_storage_model(Ti_ant, Tt_in[idx], 
                                                           Tb_in[idx], Tamb[idx], 
                                                           msrc[idx], mdis[idx], 
                                                           UA, V_i, N=N, ts=ts,
                                                           calculate_energy=True)
        Ti_ant = Ti_mod[idx]
        
    # Calculate performance metrics
    iae  = calculate_iae(Ti_mod, Ti_ref)
    ise  = calculate_ise(Ti_mod, Ti_ref)
    itae = calculate_itae(Ti_mod, Ti_ref)
    
    # Visualize result
    plot_model_result_thermal_storage(N, Tin_labels, data, Ti_mod, UA, V_i, itae, iae, ise,
                                      save_figure=save_figure, figure_path=figure_path, 
                                      figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')
    # plot_energy_thermal_storage(V_i, Ti_mod[400], energies[400], Tmin,
    #                             save_figure=save_figure, figure_path=figure_path, 
    #                             figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')
    
# plot_energy_evolution_thermal_storage(V_i, Ti_mod, energies, Tmin,
#                                       save_figure=save_figure, figure_path=figure_path, 
#                                       figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')

#%% Test thermal storage model V2

from models_psa import thermal_storage_model_two_tanks    
from parameters_fit import calculate_iae, calculate_ise, calculate_itae
from visualization.calibrations import (plot_model_result_thermal_storage, 
                                   plot_energy_thermal_storage,
                                   plot_energy_evolution_thermal_storage)

save_figure = True
figure_path = '/home/patomareao/Nextcloud/Juanmi_MED_PSA/EURECAT/Modelos/attachments'
figure_name = 'result_model_ts_V2'


var_names = {v["signal_id"]: k for k, v in vars_info.items() if 'ts' in k or not '_' in k}

Tin_labels_h = ["Tts_h_t", "Tts_h_m", "Tts_h_b"]
Tin_labels_c = [ "Tts_c_t", "Tts_c_m", "Tts_c_b"]
Tin_labels = Tin_labels_h + Tin_labels_c
data_label_prefix = "datos_tanques"

# Parameters
Tmin = 60 # Minimum useful temperature, °C
N = 3 # Number of volumes
V = 15 # Volume of an individual tank, m³
Vt = 15*2 # Total volume of the storage system (V tank·N tanks)
ts = 60*5 # Sample rate (seg)
sample_rate_str = '5Min'

# Model parameters
UA_h = np.array( [0.00561055, 0.00225925, 0.04767485] )
UA_c = np.array( [0.01019435, 0.00299455, 0.11281388] )
Vi_h = np.array( [2.44754599, 4.86137431, 2.4105236 ] )
Vi_c = np.array( [4.50502171,  1.33711331, 10.      ] )

for datos_name in ["20230621", "20230414"]:
    
    # Import data
    data = pd.read_csv(f'datos/{data_label_prefix}_{datos_name}.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
    
    data = data.rename(columns=var_names)
    data = data.resample(sample_rate_str, on='time').mean()
    
    # Remove rows with NaN values in place and generate a warning
    # data = filter_nan(data)

    # Inputs
    # Flow meter of Qdis is broken, using the output of the three-way valve 
    # model as an alternative
    data_Qdis = get_Q_from_3wv_model(datos_name, sample_rate_str=sample_rate_str)
    
    # Merge both dataframes so they are synced in time
    data.rename(columns={'m_ts_dis': 'm_ts_dis_sensor'}, inplace=True) # Rename the invalid signal
    data = pd.merge(data, data_Qdis, left_index=True, right_index=True, how='outer')
        
    # Remove rows with NaN values in place and generate a warning
    data = filter_nan(data)

    # Since we are using the first output as starting point, start from the second value
    Ti_ant = np.array( [data[T][0] for T in Tin_labels] )
    Tt_in = data.Tts_t_in.values[1:]
    Tb_in = data.Tts_b_in.values[1:] if 'Tts_b_in' in data.columns else np.zeros(len(data)-1)
    
    Tamb = data.Tamb.values[1:]
    Qsrc = data.m_ts_src.values[1:] if 'm_ts_src' in data.columns else np.zeros(len(data)-1)
    Qdis = data.m_ts_dis.values[1:] if 'm_ts_dis' in data.columns else np.zeros(len(data)-1)
    
    
    msrc = np.zeros(len(data)-1, dtype=float); mdis = np.zeros(len(data)-1, dtype=float)
    for idx in range(1, len(data)-1):
        msrc[idx] = Qsrc[idx]/60*w_props(P=0.1, T=Tt_in[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/min a kg/s
        mdis[idx] = Qdis[idx]*w_props(P=0.1, T=data.Tts_h_t[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/s a kg/s
    
    # Experimental outputs
    Ti_ref = data[Tin_labels].values[1:]
    
    # Initial outputs
    Ti_ant_h = np.array( [data[T][0] for T in Tin_labels_h] )
    Ti_ant_c = np.array( [data[T][0] for T in Tin_labels_c] )
    
    # Initialize result vectors
    Ti_h_mod   = np.zeros((len(data)-1, N), dtype=float)
    Ti_c_mod   = np.zeros((len(data)-1, N), dtype=float)


    # Evaluate model
    for idx in range(len(data)-1):
        Ti_h_mod[idx], Ti_c_mod[idx] = thermal_storage_model_two_tanks(
                                            Ti_ant_h=Ti_ant_h, Ti_ant_c=Ti_ant_c, 
                                            Tt_in=Tt_in[idx], 
                                            Tb_in= Tb_in[idx], 
                                            Tamb= Tamb[idx], 
                                            msrc= msrc[idx], 
                                            mdis= mdis[idx], 
                                            UA_h=UA_h, UA_c=UA_c,
                                            Vi_h=Vi_h, Vi_c=Vi_c,
                                            ts=ts, Tmin=Tmin, V=Vt, calculate_energy=False)
        Ti_ant_h = Ti_h_mod[idx]
        Ti_ant_c = Ti_c_mod[idx]
            
    # Calculate performance metrics
    Ti_mod = np.concatenate((Ti_h_mod, Ti_c_mod), axis=1)
    iae  = calculate_iae(Ti_mod, Ti_ref)
    ise  = calculate_ise(Ti_mod, Ti_ref)
    itae = calculate_itae(Ti_mod, Ti_ref)
    
    # Visualize result
    plot_model_result_thermal_storage(N*2, Tin_labels, data, np.concatenate((Ti_h_mod,Ti_c_mod), axis=1), 
                                      np.concatenate((UA_h, UA_c)), np.concatenate((Vi_h, Vi_c)), 
                                      itae, iae, ise, 
                                      save_figure=save_figure, figure_path=figure_path, 
                                      figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')
    
        # plot_energy_thermal_storage(V_i, Ti_mod[400], energies[400], Tmin,
        #                             save_figure=save_figure, figure_path=figure_path, 
        #                             figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')
        
    # plot_energy_evolution_thermal_storage(V_i, Ti_mod, energies, Tmin,
    #                                       save_figure=save_figure, figure_path=figure_path, 
    #                                       figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')


#%% Test three way valve model

from models_psa import three_way_valve_model    
from parameters_fit import calculate_iae, calculate_ise, calculate_itae
from visualization.calibrations import plot_model_result_three_way_valve

save_figure = True
figure_path = '/home/patomareao/Nextcloud/Juanmi_MED_PSA/EURECAT/Modelos/attachments'
figure_name = 'result_model_3wv'
# datos_name = 'datos_valvula_20230621'
data_label_prefix = "datos_valvula"

var_names = {v["signal_id"]: k for k, v in vars_info.items() if '3wv' in k or not '_' in k}


for datos_name in ["20230621", "20230414"]:

    data = pd.read_csv(f'datos/{data_label_prefix}_{datos_name}.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
    
    data = data.rename(columns=var_names)
    data = data.resample('1Min', on='time').mean()
    
    # Remove rows with NaN values in place and generate a warning
    data = filter_nan(data)
    
    # Parameters
    
    # Inputs
    Mdis = data.m_3wv_dis.values
    Tsrc = data.T_3wv_src2.values
    T_dis_in  = data.T_3wv_dis_in.values
    T_dis_out = data.T_3wv_dis_out.values
    
    # Experimental outputs
    Msrc_ref  = data.m_3wv_src.values
    R_ref = data.R_3wv.values
    
    # Initialize result vectors
    Msrc_mod = np.zeros(len(data), dtype=float)
    R_mod    = np.zeros(len(data), dtype=float)
    
    # Evaluate model
    for idx in range(len(data)):
        Msrc_mod[idx], R_mod[idx] = three_way_valve_model(Mdis[idx], Tsrc[idx], T_dis_in[idx], T_dis_out[idx])
        
    # Calculate performance metrics
    # Como  señal de Msrc_ref no es válida, de momento lo calculo con la posición 
    # de la válvula experimental
    # iae  = calculate_iae(Msrc_mod, Msrc_ref)
    # ise  = calculate_ise(Msrc_mod, Msrc_ref)
    # itae = calculate_itae(Msrc_mod, Msrc_ref)
    
    iae  = calculate_iae((1-R_mod)*100, R_ref)
    ise  = calculate_ise((1-R_mod)*100, R_ref)
    itae = calculate_itae((1-R_mod)*100, R_ref)
    
    # Visualize result
    plot_model_result_three_way_valve(data, Msrc_mod, R_mod, itae, iae, ise,
                                      save_figure=save_figure, figure_path=figure_path, 
                                      figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')

