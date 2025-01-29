#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:58:58 2023

@author: patomareao
"""

#%% Imports and initial configuration
# from models import thermal_storage
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.pyplot import close
from iapws import IAPWS97 as w_props
import os
# from results_visualization import (plot_model_result_thermal_storage, 
#                                    plot_energy_thermal_storage)
from utils.constants import vars_info, colors
# from curve_fitting import polynomial_interpolation

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

#%% Functions
    

#%% Import data
os.chdir(os.path.join(os.getenv("HOME"), "Nextcloud/Juanmi_MED_PSA/EURECAT/solarMED_modeling"))

date_parser = lambda x: pd.to_datetime(x, format='%d-%b-%Y %H:%M:%S')
# data = pd.read_csv('datos/datos_tanques.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')

data = pd.read_csv('datos/datos_campo.csv', parse_dates=['TimeStamp'], date_parser=date_parser)
#%%

var_names = {v["signal_id"]: k for k, v in vars_info.items()}

data = data.rename(columns=var_names)
data = data.iloc[:-100]
# data.set_index('time', inplace=True)
data = data.resample('1Min', on='time').mean()
ts = 60

groups = ['Loop 2', 'Loop 3', 'Loop 4', 'Loop 5', 'Global', 'Environment']
idxs   = ['l2', 'l3', 'l4', 'l5', 'global']
#%% 
# np.zeros((len(idxs)-1, len(data)))
cps = []; powers = []; out_temp_cols = []; in_temp_cols = []; flow_cols = []; flow_vals = []
for i in range(len(idxs)-1):
    out_temp_cols.append(f'T_sf_{idxs[i]}_out')
    in_temp_cols.append(f'T_sf_{idxs[i]}_in')
    flow_cols.append(f'q_sf_{idxs[i]}')
    flow_vals.append(data[flow_cols[-1]].values)
    
    # Sadly iapws does not support vectorization
    cp = np.zeros(len(data))
    for idx in range(len(data)):
        cp[idx] = w_props(P=0.1, T=data[out_temp_cols[-1]][idx]+data[in_temp_cols[-1]][idx]/2+273.15).cp
    cps.append( cp )
    
    powers.append( cp*data[flow_cols[-1]]*(data[out_temp_cols[-1]]-data[in_temp_cols[-1]]) )

# Calculate the weighted sum of temperatures
weighted_temperatures = (data[out_temp_cols].values * data[flow_cols].values).sum(axis=1)

# Calculate the total flow
qsf = data[flow_cols].sum(axis=1).values
    
# Calculate mixed temperature
Tsf_out =  weighted_temperatures/qsf
cp_out = np.zeros(len(data))
for idx in range(len(data)):
    cp_out[idx] = w_props(P=0.1, T=cp_out[idx]+273.15).cp

# Calculate contribution per loop
loops_contribution = [(data[out_temp_col].values - data['T_sf_in'].values) * data[flow_col] * cp_out for out_temp_col, flow_col in zip(out_temp_cols, flow_cols) ]

#%% Visualize data

close('all')


# Create the figure and outer gridspec
fig = plt.figure(figsize=(8, 12), )
outer_grid = gridspec.GridSpec(len(groups)*2, 1, figure=fig, hspace=0.5)

# fig, axs = plt.subplots(nrows=len(groups)*2, ncols=1, sharex=True, figsize=(8, 10))


# Loops
idx=0
for i in range(0, len(idxs)*2, 2):
    # Create the inner gridspec for each pattern
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[i:i+2], hspace=0)

    if idxs[idx] == 'global':
        ylabel = ''
    else:
        ylabel = f"_{idxs[idx]}"

    # Plot subplots within the pattern
    ax = fig.add_subplot(inner_grid[0])
    ax.plot(data.index, data[f'T_sf{ylabel}_in'], '--', color=plot_colors[idx])
    ax.plot(data.index, data[f'T_sf{ylabel}_out'], color=plot_colors[idx])
    if idxs[idx]== 'global': 
        ax.plot(data.index, Tsf_out, ':',color=plot_colors[idx])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel('T')
    ax.spines['bottom'].set_visible(False)
    
    ax.set_title(groups[idx])
    

    ax = fig.add_subplot(inner_grid[1])
    if idxs[idx]== 'global': 
        ax.stackplot(data.index, *flow_vals, alpha=0.5)
    ax.plot(data.index, data[f'q_sf{ylabel}'], color=plot_colors[idx])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel('$\dot{m}$')
    ax.spines['bottom'].set_visible(False)
        
    if idx < len(powers):
        ax = ax.twinx()
        ax.plot(data.index, powers[idx], ':', color=plot_colors[idx])
        
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('power')
        ax.spines['bottom'].set_visible(False)
        
    idx=idx+1
    
# Global
# ax = fig.add_subplot(outer_grid[-3])
# ax.plot(data.index, data['T_sf_in'])
# ax.plot(data.index, data['T_sf_out'])
# ax.plot(data.index, Tsf_out, '--')
# ax.set_xticklabels([])
# ax.set_xticks([])
# ax.set_xlabel('')
# ax.set_ylabel('Tª (ºC)')
# ax.spines['bottom'].set_visible(False)

# Power
ax = fig.add_subplot(outer_grid[-2])

ax.stackplot(data.index, *loops_contribution, alpha=0.5)
ax.plot(data.index, (data['T_sf_out']-data['T_sf_in'])*cp_out*data['q_sf'])

ax.set_ylabel('$kW_{th}$')
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_xlabel('')
ax.set_title('Thermal power')
ax.spines['bottom'].set_visible(False)
    
# Environment
ax = fig.add_subplot(outer_grid[-1])
ax.plot(data.index, data['Tamb'], label='$T_{amb}$ ($^{\circ}C$)')
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=[0.5, -0])
ax.set_title('Environment')
ax = ax.twinx()
ax.plot(data.index, data['I'], '--', label='$I$ ($W/m^2$)')
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=[0.5, -0.4])


# # Apply tight layout
# plt.constrained_layout()
    # Format xaxis
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

plt.subplots_adjust(top=0.965, bottom=0.07,)

# Show the figure
plt.show()


#%%
# Create a figure and axis
# fig, ax = plt.subplots()
fig = plt.figure(figsize=(6,7),) #constrained_layout=True)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])


ax = fig.add_subplot(gs[0])

# Plot each temperature signal
for var_name in data.columns:
    if var_name not in ["time", "Tts_t_in"] and var_name.startswith('Tts'):
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
            
        var_label = var_labels[var_name]
        
        ax.plot(data.index, data[var_name], linestyle=line_type, label=var_label, color=color)
    
# Source temperature
ax.plot(data.iloc[idxs_src].index, data.iloc[idxs_src]["Tts_t_in"], '.',color=color_src, alpha=0.3, label=var_labels["Tts_t_in"], zorder=1)
# Discharge temperature
# ax.plot(data.index[idxs_dis], data["Tts_b_in"][idxs_dis], color=color_src)

ax_r = ax.twinx()
ax_r.set_axisbelow(True)
ax_r.plot(data.index, data['Tamb'], label='$T_{amb}$ (right)', alpha=0.3)
ax_r.set_ylim([0, 100])
ax_r.set_yticks([10, 30])
# ax_r.tick_params(axis='both', which='both', zorder=-10)

# Set labels and title
ax.set_ylabel('Temperature ($^{\circ}C$)')
ax.set_title('Thermal storage temperature profile evolution over time', fontweight='bold')

# Add legend
ax.legend()
# ax.tick_params(axis='x', which='both',
#                 bottom=False) # turn off major & minor ticks on the bottom
x0 = round(193_000/ts)
duration = round(20*3600/ts)
xrange=[x0, round(x0+duration)]
ax.axvspan(data.index[xrange[0]], data.index[xrange[1]], alpha=0.3)

# Flows plot
ax = fig.add_subplot(gs[1], sharex=ax)

ax.plot(data.index, data["m_ts_src"], color=color_src, label=var_labels["m_ts_src"])
ax.legend()
ax.set_ylabel(f'{var_labels["m_ts_src"]} (L/min)')

ax = ax.twinx()
ax.set_axisbelow(True)
ax.plot(data.index, data["m_ts_dis"], color=color_dis, label = var_labels["m_ts_dis"]) if "m_ts_dis" in data.columns else None
ax.legend()
ax.set_ylabel(f'{var_labels["m_ts_dis"]} (L/s)')


ax.set_xlabel('Time')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0)  

# Display the plot
plt.show()
