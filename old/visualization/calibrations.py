#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:51:30 2023

@author: patomareao
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.pyplot import close
from utils.constants import vars_info
from utils.curve_fitting import (polynomial_interpolation, 
                           sigmoid, 
                           sigmoid_interpolation)

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

colors = ["#E77C8D", "#5AA9A2"]

#%%

def saveFigure(figure_path, figure_name):
    plt.savefig(f'{figure_path}/{figure_name}.eps', format='eps')
    plt.savefig(f'{figure_path}/{figure_name}.png', format='png')
    plt.savefig(f'{figure_path}/{figure_name}.svg', format='svg')

def plot_model_result_thermal_storage(N, Tin_labels, data, Ti_mod, UA, V_i, itae, iae, ise,
                                      save_figure=False, figure_path=None, figure_name=None):
    
    colors = ["#E77C8D", "#5AA9A2"]
    color_src = "#c01c28"
    color_dis = "#1c71d8"
    
    # var_names = {v["signal_id"]: k for k, v in vars_info.items()}
    var_labels = {k: v["label"] for k, v in vars_info.items()}
    
    idxs_src = (data["m_ts_src"] > 5).values
    idxs_dis = (data["m_ts_dis"] > 0.1).values if "m_ts_dis" in data.columns else None
    
    # Create a figure and axis
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(ncols=1, nrows=N+1, figsize=(10, 10), sharex=True)
    
    # Plot each temperature signal
    for idx, var_name in enumerate(Tin_labels):
        if var_name.startswith('Tts_h'):
            color = colors[0]
        else:
            color = colors[1]
            
        var_label = var_labels[var_name]
        
        ax[idx].plot(data.index[1:], data[var_name][1:], linestyle='dashed', label='experimental', color=color)
        ax[idx].plot(data.index[1:], Ti_mod[:,idx], linestyle='solid', label='model', color=color)
        
        # Set title
        ax[idx].set_title(f'{var_label} ($^{{\circ}}C$), UA={UA[idx]:.3f}, V={V_i[idx]:.2f} m³')
        
    # Source temperature
    ax[0].plot(data.iloc[idxs_src].index, data.iloc[idxs_src]["Tts_t_in"], '.',color=color_src, alpha=0.3, label=var_labels["Tts_t_in"], zorder=1)
    # Discharge temperature
    ax[-2].plot(data.iloc[idxs_src].index, data.iloc[idxs_src]["Tts_b_in"], '.',color=color_dis, alpha=0.3, label=var_labels["Tts_b_in"], zorder=1)
    
    # Set labels and title
    # ax.set_ylabel('Temperature ($^{\circ}C$)')
    # ax.set_title('Thermal storage temperature profile evolution over time')
    
    # Flows plot
    ax[-1].plot(data.index, data["m_ts_src"]/60, color=color_src, label=var_labels["m_ts_src"])
    # ax[-1].legend()
    # ax[-1].set_ylabel(f'{var_labels["m_ts_src"]} (L/s)')
    
    # Add background indicating the recirculating flow direction
    hot_to_cold = data["m_ts_src"]/60-data["m_ts_dis"] > 0
        
    start_date = hot_to_cold.index[0]
    prev_value = hot_to_cold.values[0]
    first_pos  = True; first_neg = True
    
    # Iterate over the data and add vertical bars
    for i in range(1, len(hot_to_cold)):
        curr_value = hot_to_cold.values[i]
    
        if curr_value != prev_value:
            if prev_value:
                color = color_src
                label = 'Hot to cold' if first_pos else None
                first_pos = False
            else:
                color = color_dis
                label = 'Cold to hot' if first_neg else None
                first_neg = False
                
            plt.axvspan(start_date, hot_to_cold.index[i-1], facecolor=color, alpha=0.2, label=label)
            start_date = data.index[i]
    
        prev_value = curr_value
        
    plt.axvspan(hot_to_cold.index[i], hot_to_cold.index[-1], facecolor=color_src if curr_value else color_dis, alpha=0.2)

    
    # ax[-1] = ax[-1].twinx()
    ax[-1].plot(data.index, data["m_ts_dis"], color=color_dis, label = var_labels["m_ts_dis"]) if "m_ts_dis" in data.columns else None
    ax[-1].set_ylabel('Flows (L/s)')
    ax[-1].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.3))
    # ax[-1].set_ylabel(f'{var_labels["m_ts_dis"]} (L/s)')
    
    # Add legend
    ax[0].legend(bbox_to_anchor=(0.65, -N*1.3), ncol=3)
    
    # Format xaxis
    ax[-1].xaxis.set_major_locator(locator)
    ax[-1].xaxis.set_major_formatter(formatter)
    
    # Set title of figure
    fig.suptitle(f'Model fit to experimental data for each control volume\n ITAE={itae:.2E}, IAE={iae:.2E}, ISE={ise:.2E}', fontweight='bold')
    
    # Display the plot
    plt.show()
    
    # Save figure
    if save_figure: saveFigure(figure_path, figure_name)
    
def plot_energy_thermal_storage(volumes, temperatures, energy, Tmin, area=2,
                                save_figure=False, figure_path=None, figure_name=None):
    interp_volumes = np.linspace(np.min(volumes), np.max(volumes), num=100)  # Generate interpolated volume points
    
    z = np.cumsum(volumes)/area
    interp_z = np.linspace(np.min(z), np.max(z), num=100)
    
    # Temperatures are given from top to bottom, but in order to obtain the 
    # temperature profile at any given height of the tank, we need to reverse
    # the order
    temperatures = temperatures[::-1]
    
    popt = sigmoid_interpolation(z, temperatures) # obtain sigmoid curve parameters
    
    interp_temperatures = sigmoid(interp_z, *popt[0]) # Interpolate temperatures
    
    # interp_temperatures = polynomial_interpolation(volumes, temperatures, interp_volumes)

    volume_above_Tmin = interp_z[interp_temperatures > Tmin]
    temperatures_above_Tmin = interp_temperatures[interp_temperatures > Tmin]

    # Create the subplot with two columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # First subplot - Energy Stored Above Tmin
    axs[0].plot(interp_z, interp_temperatures, label='Interpolated Temperature')
    axs[0].axhline(Tmin, color='red', linestyle='--', label='Tmin')

    # Shade the area under the curve above Tmin
    axs[0].fill_between(volume_above_Tmin, Tmin, temperatures_above_Tmin, alpha=0.3, color='blue', label='Energy Stored')

    axs[0].plot(z, temperatures, 'x', label='Reference temperatures')

    # Set labels and title
    axs[0].set_xlabel('Height (m)')
    axs[0].set_ylabel('Temperature ($^{\circ}C$)')
    axs[0].set_title(f'Energy Stored Above Tmin = {energy:.1f} kWh')

    # Add reference temperatures
    # axs[0].text(np.max(interp_volumes), Tmin, 'Tmin', verticalalignment='bottom', horizontalalignment='right', color='red')

    # Add legend
    axs[0].legend(loc='upper center', ncols=4, bbox_to_anchor=(1, 1.25), frameon=False)

    # Second subplot - Temperature Distribution in Storage
    # Define the width (constant temperature)
    width = 1.0

    # Create meshgrid for volume and width
    volumes, widths = np.meshgrid(interp_volumes, np.array([0, width]))

    # Create temperature meshgrid
    temperatures_mesh = np.tile(interp_temperatures, (2, 1))

    # Plot the temperature color gradient
    im = axs[1].pcolormesh(widths, volumes, temperatures_mesh, cmap='Pastel2')

    # Add a horizontal line for volumes below Tmin
    # Find the index where interp_temperatures intersects with Tmin
    intersection_indices = np.argwhere(interp_temperatures <= Tmin)

    # Extract the index of the first intersection
    if len(intersection_indices) > 0:
        first_intersection_index = intersection_indices[0, 0]
        print("Index of intersection:", first_intersection_index)
    else:
        print("No intersection found.")
        first_intersection_index = None

    if first_intersection_index:
        axs[1].axhline(y=volumes[0][first_intersection_index], color='red', linestyle='--', label='Tmin')

    # Set labels and title
    axs[1].set_ylabel('Volume (m³)')
    axs[1].set_title('Temperature Distribution in Storage')

    # Remove width axis labels and ticks
    axs[1].tick_params(axis='x', labelbottom=False, bottom=False)

    # Invert the direction of the y-axis
    axs[1].invert_yaxis()

    # Add reference temperatures
    # axs[1].text(np.max(widths), Tmin, 'Tmin',) #verticalalignment='bottom', horizontalalignment='right', color='red')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axs[1], label='Temperature ($^{\circ}C$)')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.25)

    # Display the plot
    plt.show()
    
    # Save figure
    if save_figure: saveFigure(figure_path, figure_name)
    
def plot_energy_evolution_thermal_storage(volumes, temperatures, energies, Tmin, ts=600,
                                          save_figure=False, figure_path=None, figure_name=None):
    interp_volumes = np.linspace(np.min(volumes), np.max(volumes), num=100)  # Generate interpolated volume points
    
    # Create the subplot with two columns
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    
    for idx in range(len(temperatures)):
        interp_temperatures = polynomial_interpolation(volumes, temperatures[idx], interp_volumes)
    
        volume_above_Tmin = interp_volumes[interp_temperatures > Tmin]
        temperatures_above_Tmin = interp_temperatures[interp_temperatures > Tmin]
    
        # Second subplot - Temperature Distribution in Storage over time
        # Define the width (constant temperature)
        duration = np.arange(0, len(temperatures), step=ts/3600)
    
        # Create meshgrid for volume and width
        volumes, durations = np.meshgrid(interp_volumes, duration)
    
        # Create temperature meshgrid
        temperatures_mesh = np.tile(interp_temperatures, (2, 1))
    
        # Plot the temperature color gradient
        im = axs[1].pcolormesh(durations, volumes, temperatures_mesh, cmap='Pastel2')
    
        # Add a horizontal line for volumes below Tmin
        # Find the index where interp_temperatures intersects with Tmin
        intersection_indices = np.argwhere(interp_temperatures <= Tmin)
    
        # Extract the index of the first intersection
        if len(intersection_indices) > 0:
            first_intersection_index = intersection_indices[0, 0]
            print("Index of intersection:", first_intersection_index)
        else:
            print("No intersection found.")
            first_intersection_index = None
    
        if first_intersection_index:
            axs[1].axhline(y=volumes[0][first_intersection_index], color='red', linestyle='--', label='Tmin')

    # Set labels and title
    axs[1].set_ylabel('Volume (m³)')
    axs[1].set_title('Temperature Distribution in Storage')

    # Remove width axis labels and ticks
    axs[1].tick_params(axis='x', labelbottom=False, bottom=False)

    # Invert the direction of the y-axis
    axs[1].invert_yaxis()

    # Add reference temperatures
    # axs[1].text(np.max(widths), Tmin, 'Tmin',) #verticalalignment='bottom', horizontalalignment='right', color='red')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axs[1], label='Temperature ($^{\circ}C$)')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.25)

    # Display the plot
    plt.show()
    
    # Save figure
    if save_figure: saveFigure(figure_path, figure_name)
    
    
def plot_model_result_three_way_valve(data, m_3wv_src_estimated, R_3wv_estimated,
                                      itae, iae, ise,
                                      save_figure=False, figure_path=None, figure_name=None):
    
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
    axs[1].plot(data.index, m_3wv_src_estimated, '--',label=f"{vars_info[var_name]['label']} model", color=plot_colors[0])
    
    
    # scaling = 80
    # axs[1].plot(data.index, data.m_3wv_src*scaling, '--',label=f'{vars_info["m_3wv_src"]["label"]} x{scaling}', color=plot_colors[1])
    # axs[1].set_ylim([np.min(data.m_3wv_dis)*0.9, np.max(data.m_3wv_dis)*1.1])
    # axs[1].set_ylim([0, 20])
    
    axs[1].set_ylabel('Flows ($L/s$)')
    axs[1].legend(ncols=1, frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Third subplot: Valve position
    axs[2].step(data.index, data.R_3wv, color=plot_colors[0], label=vars_info['R_3wv']['label'])
    # axs[2].step(data.index, (1-R_3wv_estimated)*100, '--', color=plot_colors[0], label='estimated')
    axs[2].step(data.index, (1-R_3wv_estimated)*100, ':', linewidth=3, color=plot_colors[1], label=f"{vars_info['R_3wv']['label']} model")
    
    axs[2].set_ylabel('Valve Position ($\%$)')
    legend = axs[2].legend(ncols=1, frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # axs[2].set_yticks([0, 1])
    # axs[2].set_ylim([-0.5, 1.5])
    
    # Global
    axs[2].set_xlabel('Time')
    
    # Set title of figure
    fig.suptitle(f'Model fit to experimental data\n ITAE={itae:.2E}, IAE={iae:.2E}, ISE={ise:.2E}', fontweight='bold')
    
    
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
    
    # plt.subplots_adjust(left=0.1, right=available_width)
    # Adjust spacing between subplots
    fig.tight_layout()
    
    # Show the plot
    plt.show()

    
    # Save figure
    if save_figure: 
        saveFigure(figure_path, figure_name)