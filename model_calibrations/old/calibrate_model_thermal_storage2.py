#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:53:51 2023

@author: patomareao
"""

#%% Imports and initial configuration
# from models import thermal_storage
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.pyplot import close
from iapws import IAPWS97 as w_props
# from results_visualization import (plot_model_result_thermal_storage, 
#                                    plot_energy_thermal_storage)
import sys
sys.path.insert(0, "..")

from utils.constants import var_labels, var_names, colors
from utils.constants import vars_info
from utils import filter_nan, get_Q_from_3wv_model

from pathlib import Path

# import matplotlib
# matplotlib.use('TkAgg')

sns.set_theme()
myFmt = mdates.DateFormatter('%H:%M')
plot_colors = sns.color_palette()

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats = ['%y',  # ticks are mostly years
                     '%b',       # ticks are mostly months
                     '%d',       # ticks are mostly days
                     '%H:%M',    # hrs
                     '%H:%M',    # min
                     '%S.%f', ]  # secs
# these are mostly just the level above...
formatter.zero_formats = [''] + formatter.formats[:-1]
# ...except for ticks that are mostly hours, then it is nice to have
# month-day:
formatter.zero_formats[3] = '%d-%b'

formatter.offset_formats = ['',
                            '%Y',
                            '%b %Y',
                            '%d %b %Y',
                            '%d %b %Y',
                            '%d %b %Y %H:%M', ]

Tin_labels_h = ["Tts_h_t", "Tts_h_m", "Tts_h_b"]
Tin_labels_c = [ "Tts_c_t", "Tts_c_m", "Tts_c_b"]
Tin_labels = Tin_labels_h + Tin_labels_c

#%% Import data
# os.chdir(os.path.join(os.getenv("HOME"), "Nextcloud/Juanmi_MED_PSA/EURECAT/solarMED_modeling"))
base_path = Path( f'{os.getenv("HOME")}/development_psa/models_psa/data')
data_path = base_path / 'calibration/20230707_20230710_datos_tanques.csv'

datos_date_str = '20230621'
sample_rate_str = '30S'
ts = 0.5*60

# Python <=3.9
# date_parser = lambda x: pd.to_datetime(x, format='%d-%b-%Y %H:%M:%S')
# data = pd.read_csv('datos/datos_tanques.csv', parse_dates=['TimeStamp'], date_parser=date_parser)

# Python >3.9
# data = pd.read_csv(f'datos/datos_tanques_{datos_date_str}.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
# data = pd.read_csv(f'datos/datos_tanques.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
data = pd.read_csv(data_path, parse_dates=['TimeStamp'])

data = data.rename(columns=var_names)

var_names = {v["signal_id"]: k for k, v in vars_info.items() if 'ts' in k or not '_' in k}

data = data.rename(columns=var_names)

data = data.resample(sample_rate_str, on='time').mean()

# Flow meter of Qdis is broken, using the output of the three-way valve 
# model as an alternative
# data_Qdis = get_Q_from_3wv_model(datos_date_str, sample_rate_str=sample_rate_str)

# data["Tts_b_in"] = data["Tts_b_in"].values() if 'Tts_b_in' in data.columns else np.zeros(len(data))

# Merge both dataframes so they are synced in time
# data.rename(columns={'m_ts_dis': 'm_ts_dis_sensor'}, inplace=True) # Rename the invalid signal
# data = pd.merge(data, data_Qdis, left_index=True, right_index=True, how='outer')

# Remove rows with NaN values in place and generate a warning
data = filter_nan(data)

#%% Visualize data
close('all')

idxs_src = (data["mts_src"] > 5).values
idxs_dis = (data["mts_dis"] > 0.1).values if "mts_dis" in data.columns else None
color_src = "#c01c28"
color_dis = "#1c71d8"

# Create a figure and axis
# fig, ax = plt.subplots()
fig = plt.figure(figsize=(8,10),) #constrained_layout=True)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])


ax = fig.add_subplot(gs[0])

# Plot each temperature signal
for var_name in data.columns:
    if var_name not in ["time", "Tts_t_in", "Tts_b_in"] and var_name.startswith('Tts'):
        if var_name.startswith('Tts_h'):
            color = colors[0]
        else:
            color = colors[1]
            
        if var_name.endswith('t'):
            line_type = 'solid'
        elif var_name.endswith('m'):
            line_type = 'dashed'
        else:
            line_type = 'dashdot'
            
        var_label = vars_info[var_name]['label']
        
        ax.plot(data.index, data[var_name], linestyle=line_type, label=var_label, color=color)
    
# Source temperature
ax.plot(data.iloc[idxs_src].index, data.iloc[idxs_src]["Tts_t_in"], '.',color=color_src, alpha=0.3, label=vars_info['Tts_t_in']['label'], zorder=1)
# Discharge temperature
ax.plot(data.index[idxs_dis], data["Tts_b_in"][idxs_dis], '.',color=color_dis, alpha=0.3, label=vars_info['Tts_b_in']['label'], zorder=1)

ax.set_xticks([]); ax.set_xticklabels([])

ax_r = ax.twinx()
ax_r.set_axisbelow(True)
ax_r.plot(data.index, data['Tamb'], label='$T_{amb}$ (right)', alpha=0.3)
ax_r.set_ylim([0, 100])
ax_r.set_yticks([10, 30])
# ax_r.tick_params(axis='both', which='both', zorder=-10)
ax.set_xticks([]); ax.set_xticklabels([])

# Set labels and title
ax.set_ylabel('Temperature ($^{\circ}C$)')
ax.set_title('Thermal storage temperature profile evolution over time', fontweight='bold')

# Add legend
ax.legend(ncols=4)
# ax.tick_params(axis='x', which='both',
#                 bottom=False) # turn off major & minor ticks on the bottom
# x0 = round(193_000/ts)
# duration = round(20*3600/ts)
# xrange=[x0, round(x0+duration)]
# ax.axvspan(data.index[xrange[0]], data.index[xrange[1]], alpha=0.3)

# Flows plot
ax = fig.add_subplot(gs[1], sharex=ax)

ax.plot(data.index, data["mts_src"], color=color_src)
ax.legend()
# ax.set_ylabel(f'{var_labels["mts_src"]} (L/min)')

ax = ax.twinx()
ax.set_axisbelow(True)
ax.plot(data.index, data["mts_dis"], color=color_dis)
ax.legend()
# ax.set_ylabel(f'{var_labels["mts_dis"]} (L/s)')


ax.set_xlabel('Time')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0)  
# Adjust the spacing between subplots
plt.subplots_adjust(top=0.940, bottom=0.075)

# Display the plot
plt.show()

#%% Test model prior to parameter fit

from models_psa import thermal_storage_model_two_tanks, thermal_storage_model_single_tank
from parameters_fit import calculate_iae, calculate_ise, calculate_itae
from visualization.calibrations import plot_model_result_thermal_storage


# Parameters
Tmin = 60 # Minimum useful temperature, °C
N = 3 # Number of volumes
V = 15 # Volume of an individual tank, m³
Vt = 15*2 # Total volume of the storage system (V tank·N tanks)
Tmin = 60 # Minimum useful temperature, °C
V_i = np.ones(N)*V/N  # Volume of each control volume

# Model parameters
UA_h  = np.array([0.0068, 0.004, 0.0287])
# Vi_h = np.array([2.9722, 1.7128, 9.4346])
Vi_h  = V_i.copy()

UA_c  = np.array([0.0068, 0.004, 0.0287])
# Vi_c = np.array([2.9722, 1.7128, 9.4346])
Vi_c = V_i.copy()

# Inputs 

# Since we are using the first output as starting point, start from the second value
Ti_ant_h = np.array( [data[T][0] for T in Tin_labels_h] )
Ti_ant_c = np.array( [data[T][0] for T in Tin_labels_c] )
Tt_in = data.Tts_t_in.values[1:]
Tb_in = data.Tts_b_in.values[1:] if 'Tts_b_in' in data.columns else np.zeros(len(data)-1)

Tamb = data.Tamb.values[1:]
Qsrc = data.mts_src.values[1:] if 'mts_src' in data.columns else np.zeros(len(data)-1)
Qdis = data.mts_dis.values[1:] if 'mts_dis' in data.columns else np.zeros(len(data)-1)


msrc = np.zeros(len(data)-1, dtype=float); mdis = np.zeros(len(data)-1, dtype=float)
for idx in range(1, len(data)-1):
    msrc[idx] = Qsrc[idx]/60*w_props(P=0.1, T=Tt_in[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/min a kg/s
    mdis[idx] = Qdis[idx]*w_props(P=0.1, T=data.Tts_h_t[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/s a kg/s

# Experimental outputs
Ti_ref = np.concatenate((data[Tin_labels_h].values[1:], data[Tin_labels_c].values[1:]), axis=1)

# Initialize result vectors
Ti_h_mod   = np.zeros((len(data)-1, N), dtype=float)
Ti_c_mod   = np.zeros((len(data)-1, N), dtype=float)


# Evaluate model
for idx in range(len(data)-1):
    # Ti_c_mod[idx] = thermal_storage_model_single_tank(
    #                              Ti_ant_c, Tt_in=0, Tb_in=Tb_in[idx], Tamb=Tamb[idx], 
    #                              mt_in=0, mb_in=mdis[idx], mt_out=mdis[idx]-msrc[idx], mb_out=msrc[idx],
    #                              UA=UA_c, V_i=Vi_c, 
    #                              N=3, ts=ts, calculate_energy=False)
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

Ti_mod = Ti_c_mod
Ti_ref = data[Tin_labels_c].values[1:]
iae  = calculate_iae(Ti_mod, Ti_ref)
ise  = calculate_ise(Ti_mod, Ti_ref)
itae = calculate_itae(Ti_mod, Ti_ref)

# Visualize result
plot_model_result_thermal_storage(N*2, Tin_labels, data, np.concatenate((Ti_h_mod,Ti_c_mod), axis=1), 
                                  np.concatenate((UA_h, UA_c)), np.concatenate((Vi_h, Vi_c)), 
                                  itae, iae, ise)

# plot_model_result_thermal_storage(N, Tin_labels_c, data, Ti_c_mod, 
#                                   UA_c, Vi_c, itae, iae, ise)

#%% Parameter fit

from parameters_fit import objective_function
from models_psa import thermal_storage_model_two_tanks
from visualization.calibrations import plot_model_result_thermal_storage
from optimparallel import minimize_parallel
from parameters_fit import calculate_iae, calculate_ise, calculate_itae

save_figure = True
figure_path = '/home/jmserrano/Nextcloud/Juanmi_MED_PSA/EURECAT/Modelos/attachments'
figure_name = 'result_model_ts_calibration'


# Parameters
Tmin = 60 # Minimum useful temperature, °C
N = 3 # Number of volumes
V = 15 # Volume of an individual tank, m³
Vt = 15*2 # Total volume of the storage system (V tank·N tanks)
V_i = np.ones(N)*V/N  # Volume of each control volume

# Inputs 

# Since we are using the first output as starting point, start from the second value
Ti_ant_h = np.array( [data[T][0] for T in Tin_labels_h] )
Ti_ant_c = np.array( [data[T][0] for T in Tin_labels_c] )
Tt_in = data.Tts_t_in.values[1:]
Tb_in = data.Tts_b_in.values[1:] if 'Tts_b_in' in data.columns else np.zeros(len(data)-1)

Tamb = data.Tamb.values[1:]
Qsrc = data.mts_src.values[1:] if 'mts_src' in data.columns else np.zeros(len(data)-1)
Qdis = data.mts_dis.values[1:] if 'mts_dis' in data.columns else np.zeros(len(data)-1)


msrc = np.zeros(len(data)-1, dtype=float); mdis = np.zeros(len(data)-1, dtype=float)
for idx in range(1, len(data)-1):
    msrc[idx] = Qsrc[idx]/60*w_props(P=0.1, T=Tt_in[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/min a kg/s
    mdis[idx] = Qdis[idx]*w_props(P=0.1, T=data.Tts_h_t[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/s a kg/s

# Experimental outputs
Ti_ref = np.concatenate((data[Tin_labels_h].values[1:], data[Tin_labels_c].values[1:]), axis=1)

# Define optimizer inputs
inputs = [Ti_ant_h, Ti_ant_c, Tt_in, Tb_in, Tamb, msrc, mdis]  # Input values
outputs = Ti_ref  # Actual output values
params = (ts, Tmin, Vt)    # Constant model parameters
params_objective_function = {'metric': 'IAE', 'recursive':True, 'n_outputs':2, 
                             'n_parameters': 4} # 'len_outputs':[N, N]

# Set initial parameter values
# initial_parameters = [0.01 for _ in range(N)]
initial_parameters = np.concatenate((np.array([0.02136564, 0.01593324, 0.01918577]), # UA_h
                                     np.array([0.02136564, 0.01593324, 0.01918577]), # UA_c
                                     np.ones(N)*V/N, # Vi_h,
                                     np.ones(N)*V/N, # Vi_c
                                    ))
#         UAmin, UAmax    UAmin, UAmax    Vi_min     Vi_max       Vi_min     Vi_max
bounds = ((1e-4, 1),)*N + ((1e-4, 1),)*N + ((0.1*V/N, 2*V/N),)*N + ((0.1*V/N, 2*V/N),)*N

# Perform parameter calibration
optimized_parameters = minimize_parallel(
    objective_function,
    initial_parameters,
    args=(thermal_storage_model_two_tanks, inputs, outputs, params, params_objective_function),
    bounds = bounds,
    # method='L-BFGS-B'
).x

op = optimized_parameters

L = int(len(op)/4)
UA_h  = op[:L]
UA_c  = op[L:2*L]
Vi_h  = op[2*L:3*L]
Vi_c  = op[3*L:]

# optimized_parameters = array([6.77724155e-03, 3.96580419e-03, 2.87258611e-02, 6.88542188e-03, 2.97217468e+00, 1.71277001e+00, 9.43455760e+00, 3.78073750e+00])
# Run model with optimized parameters

# Reset initial input
Ti_ant_h = np.array( [data[T][0] for T in Tin_labels_h] )
Ti_ant_c = np.array( [data[T][0] for T in Tin_labels_c] )

# Initialize result vectors
Ti_h_mod   = np.zeros((len(data)-1, N), dtype=float)
Ti_c_mod   = np.zeros((len(data)-1, N), dtype=float)


# Evaluate model
for idx in range(len(data)-1):
    # Ti_c_mod[idx] = thermal_storage_model_single_tank(
    #                              Ti_ant_c, Tt_in=0, Tb_in=Tb_in[idx], Tamb=Tamb[idx], 
    #                              mt_in=0, mb_in=mdis[idx], mt_out=mdis[idx]-msrc[idx], mb_out=msrc[idx],
    #                              UA=UA_c, V_i=Vi_c, 
    #                              N=3, ts=ts, calculate_energy=False)
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

# Ti_mod = Ti_c_mod
# Ti_ref = data[Tin_labels_c].values[1:]
# iae  = calculate_iae(Ti_mod, Ti_ref)
# ise  = calculate_ise(Ti_mod, Ti_ref)
# itae = calculate_itae(Ti_mod, Ti_ref)

# Visualize result

# Since we don't have fill it with zeros
if "mts_dis" not in data.columns:
    data["mts_dis"] = np.zeros(len(data))

plot_model_result_thermal_storage(N*2, Tin_labels, data, np.concatenate((Ti_h_mod,Ti_c_mod), axis=1), 
                                  np.concatenate((UA_h, UA_c)), np.concatenate((Vi_h, Vi_c)), 
                                  itae, iae, ise, 
                                  save_figure=save_figure, figure_path=figure_path, 
                                  figure_name=f'{figure_name}_{data.index[0].to_pydatetime().date().isoformat()}')


"""
Calibrated parameters:
    
    - UA_h: [0.00561055, 0.00225925, 0.04767485]
    - UA_c: [0.01019435, 0.00299455, 0.11281388]
    - Vi_h: [2.44754599, 4.86137431, 2.4105236 ]
    - Vi_c: [4.50502171,  1.33711331, 10.      ]
    
    
    V2. 20230714
    
    - UA_h: [0.0069818 , 0.00584034, 0.03041486]
    - UA_c: [0.01396848, 0.0001    , 0.02286885]
    - Vi_h: [5.94771006, 4.87661781, 2.19737023]
    - Vi_c: [5.33410037, 7.56470594, 0.90547187]
"""