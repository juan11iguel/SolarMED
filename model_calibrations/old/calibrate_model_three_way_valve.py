#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:05:08 2023

@author: patomareao
"""
# MATLAB model
import os
os.environ["MR"] = f"{os.environ['HOME']}/PSA/MATLAB_runtime"
MR = os.environ["MR"]
os.environ["LD_LIBRARY_PATH"] = f"{MR}/v911/runtime/glnxa64:{MR}/v911/bin/glnxa64:{MR}/v911/sys/os/glnxa64:{MR}/v911/sys/opengl/lib/glnxa64"

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


#%% Import data
# os.chdir(os.path.join(os.getenv("HOME"), "Nextcloud/Juanmi_MED_PSA/EURECAT/models_psa"))

var_names = {v["signal_id"]: k for k, v in vars_info.items() if '3wv' in k or not '_' in k}

data = pd.read_csv('data/datos_valvula_20230414.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')


data = data.rename(columns=var_names)
data = data.resample('1Min', on='time').mean()

data = data[10:-5]

#%% Visualize data
close('all')

# Check FT-AQU-100 and compare with estimation
fig, axs = plt.subplots(1, 1, figsize=(6, 8), sharex=True)

R_3wv_estimated = ( data.T_3wv_dis_in -  data.T_3wv_src )/( data.T_3wv_dis_out - data.T_3wv_src )
R_3wv_estimated2 = np.copy(R_3wv_estimated)
R_3wv_estimated2[R_3wv_estimated > 1] = 1
R_3wv_estimated2[R_3wv_estimated < 0] = 0

R_3wv_estimated3 = ( data.T_3wv_dis_in -  data.T_3wv_src2 )/( data.T_3wv_dis_out - data.T_3wv_src2 )
R_3wv_estimated3[R_3wv_estimated3 > 1] = 1
R_3wv_estimated3[R_3wv_estimated3 < 0] = 0

m_3wv_src_estimated = data.m_3wv_dis * ( data.T_3wv_dis_in -  data.T_3wv_dis_out )/( data.T_3wv_src - data.T_3wv_dis_out )
m_3wv_src_estimated2 = data.m_3wv_dis * (1-R_3wv_estimated2)
m_3wv_src_estimated3 = data.m_3wv_dis * (1-R_3wv_estimated3)

for var_name, color in zip(['m_3wv_src', 'm_3wv_dis'], plot_colors):
    axs.plot(data.index, data[var_name], label=vars_info[var_name]['label'], color=color)
axs.plot(data.index, m_3wv_src_estimated, '--',label='estimated', color=plot_colors[0])
axs.set_ylabel('Flows ($L/s$)')
axs.legend()

plt.show()

#%% Summary plot
close('all')

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# First subplot: Temperatures
for var_name, color in zip(['T_3wv_src', 'T_3wv_dis_out', 'T_3wv_dis_in'], plot_colors):
    axs[0].plot(data.index, data[var_name], label=vars_info[var_name]['label'], 
                color=vars_info[var_name]['color'])
    
for var_name, color in zip(['T_3wv_src2', 'T_3wv_dis_out2'], plot_colors):
    axs[0].plot(data.index, data[var_name], '--', 
                label=vars_info[var_name]['label'], color=vars_info[var_name]['color'])
axs[0].set_ylabel('Temperature (${^\circ}C$)')
axs[0].legend(ncols=1, frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))

# Second subplot: Flows
for var_name in ['m_3wv_src', 'm_3wv_dis']:
    axs[1].plot(data.index, data[var_name], label=vars_info[var_name]['label'])
axs[1].plot(data.index, m_3wv_src_estimated, '--',label='estimated', color=plot_colors[0])
axs[1].plot(data.index, m_3wv_src_estimated2, ':', linewidth=3,label='estimated with $T_{ts,h,out}$ (sat)', color=plot_colors[0])
axs[1].plot(data.index, m_3wv_src_estimated3, ':', linewidth=3,label='estimated with $T_{ts,h,t}$ (sat)', color=plot_colors[2])


# scaling = 80
# axs[1].plot(data.index, data.m_3wv_src*scaling, '--',label=f'{vars_info["m_3wv_src"]["label"]} x{scaling}', color=plot_colors[1])
# axs[1].set_ylim([np.min(data.m_3wv_dis)*0.9, np.max(data.m_3wv_dis)*1.1])
# axs[1].set_ylim([0, 20])

axs[1].set_ylabel('Flows ($L/s$)')
axs[1].legend(ncols=1, frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))

# Third subplot: Valve position
axs[2].step(data.index, data.R_3wv, color=plot_colors[0], label=vars_info['R_3wv']['label'])
# axs[2].step(data.index, (1-R_3wv_estimated)*100, '--', color=plot_colors[0], label='estimated')
axs[2].step(data.index, (1-R_3wv_estimated2)*100, ':', linewidth=3, color=plot_colors[1], label='estimated with $T_{ts,h,out}$')
axs[2].step(data.index, (1-R_3wv_estimated3)*100, '--', color=plot_colors[2], label='estimated with $T_{ts,h,t}$')

axs[2].set_ylabel('Valve Position ($\%$)')
legend = axs[2].legend(ncols=1, frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))

# axs[2].set_yticks([0, 1])
# axs[2].set_ylim([-0.5, 1.5])

# Global
axs[2].set_xlabel('Time')

# Adjust spacing between subplots
fig.tight_layout()

# Format xaxis
axs[-1].xaxis.set_major_locator(locator)
axs[-1].xaxis.set_major_formatter(formatter)

for ax in axs:
    ## Get the current width of the window
    window_width = fig.get_window_extent().width
    legend_width = legend.get_window_extent().width
    # Shrink current axis to fit legends
        
    # Calculate the available width for the plot
    available_width = (window_width - 1.3*legend_width)/window_width  # Adjust the spacing as needed
    
    # Adjust the width of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, available_width, box.height])

# 

plt.subplots_adjust(left=0.1, right=available_width)

# Show the plot
plt.show()



# #%%  Fit curve for valve scada signal to valve mix ratio

# from curve_fitting import polynomial_interpolation

# R_3wv_estimated2 = polynomial_interpolation(x=R_3wv_estimated, y=data.R_3wv, x_interp=np.arange(0, 100, 0.2))