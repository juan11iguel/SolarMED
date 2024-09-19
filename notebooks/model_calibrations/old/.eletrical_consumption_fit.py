#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:02:48 2023

@author: jmserrano
"""

#%% Import libraries
import pandas as pd
import numpy
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep,splev # Spline fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
import matplotlib.dates as mdates
import seaborn as sns
import json
from scipy.interpolate import UnivariateSpline
import os

sns.set_theme()
myFmt = mdates.DateFormatter('%H:%M')
plot_colors = sns.color_palette()

#%% Functions
# Define curve functions for each model


# Define an objective function to evaluate the goodness of fit
def objective_function(params, curve_function):
    residuals = y_data - curve_function(x_data, *params)
    return np.sum(residuals**2)  # Sum of squared residuals

def ensure_monotony(data:np.array, gap=0.01):
    # non_increasing_indices = np.where(data[1:] < data[:-1])[0]
    # non_increasing_values = data[non_increasing_indices]
    # replacement_values = np.maximum.accumulate(non_increasing_values)+0.01
    
    # data[non_increasing_indices + 1] = replacement_values
    
    for i in range(1, len(data)):
        if data[i] < data[i-1]:
            data[i] = data[i-1] + gap
    
    return data

def fit_curve(x_data: np.array, y_data: np.array, fit_name:str, unit='kW', visualize_result=True, 
              save_result=False, result_path=None, include_spline=False):
    # Define a list of candidate curve functions
    curve_functions = [
        exponential_curve,
        quadratic_curve,
        logarithmic_curve,
        linear_fit,
    ]
    
    curve_functions.append(spline_fit) if include_spline else None
    
    # Perform curve fitting with different curve models
    best_params = None
    best_cost = float('inf')
    params = {}
    fit = {
        'best_fit': None,
        'params': {},
        'cost': None,
        'x_data': None,
        'y_data': None
    }
    
    if unit == 'W':
        y_data = y_data*1e-3
        print('Converted ydata to kW from W (·1e⁻³)')
    
    
    for curve_function in curve_functions:
        # Fit the curve to the data using curve_fit
        try:
            if curve_function.__name__ == 'linear_fit':
                popt, _ = curve_fit(curve_function, x_data, y_data, method='lm')
            elif curve_function.__name__ == 'spline_fit':
                popt = splrep(x_data, y_data, k=3, s=0.1)
            else:
                popt, _ = curve_fit(curve_function, x_data, y_data)
        except RuntimeError:
            print(f'Failed attemp to fit using {curve_function.__name__}, skipping')
            params[curve_function.__name__] = None
        else:
            # Evaluate the goodness of fit using the objective function
            cost = objective_function(popt, curve_function)
        
            # Save params for each alternative
            if isinstance(popt, numpy.ndarray):
                params[curve_function.__name__] = popt.tolist()
            else:
                params[curve_function.__name__] = []
                for p in popt:
                    if isinstance(p, numpy.ndarray):
                        params[curve_function.__name__].append( p.tolist() )
                    else:
                        params[curve_function.__name__].append( p )
                        
            
            # Check if this model provides a better fit
            if cost < best_cost:
                best_params = popt
                best_cost = cost
                fit['best_fit'] = curve_function.__name__
                fit['cost'] = best_cost
    
    fit['params'] = params
    fit['x_data'] = x_data.tolist()
    fit['y_data'] = y_data.tolist()
    
    # Print the best-fit parameters
    print('Best-fit parameters:')
    for i, param in enumerate(best_params):
        print(f'Parameter {i+1}: {param}')
    
    if visualize_result:
        close('all')
        
        # Plot the data points
        plt.scatter(x_data, y_data, label='Data')
        
        # Plot the best-fit curve for each alternative
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        
        for curve_function in curve_functions:
            # if curve_function.__name__ == 'linear_fit':
            if params[curve_function.__name__] is not None:
                y_fit = curve_function(x_range, *params[curve_function.__name__])
                # else:
                #     y_fit = curve_function(x_range, *best_params)
                label = curve_function.__name__
                plt.plot(x_range, y_fit, label=label)
            
        # Add labels and legend to the plot
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.legend()
        
        # Display the plot
        plt.show()
        
    # Save results
    if save_result:
        # Read existing JSON file, if it exists
        existing_data = {}
        try:
            with open(result_path, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            pass
    
        # Append new data to the existing data
        existing_data[fit_name] = fit
    
        # Write the serialized JSON to the file
        with open(result_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    
    return fit

#%%
# 20240204 Legacy, moved to a notebook `electrical_consumption.ipynb`

if __name__=='__main__':
    # Load data and perform fit
    # df = pd.DataFrame({'independent': [1, 2, 3, 4, 5],
    #                    'dependent': [2.5, 3.6, 4.8, 5.7, 6.9]})

    # # Extract the independent and dependent variables as NumPy arrays
    # x_data = df['independent'].values
    # y_data = df['dependent'].values

    save_result = True
    result_path = 'datos/curve_fits.json'
    base_path = f'{os.environ["HOME"]}/Nextcloud'

    #%% Feedwater pump consumption
    data_path = f'{base_path}/Juanmi_MED_PSA/MATLAB/workspaces/datos_consumo_Mcw_Mf.csv'
    fit_name = 'feedwater_electrical_consumption'
    include_spline = False
    df = pd.read_csv(data_path)
    xrange = np.arange(552, 795)
    unit='W'

    x_data = df['FT-DES-003'][xrange]    # (m³/h)
    y_data = df['PK-MED-E02-pa'][xrange] # (W)

    #%% Cooling pump consumption
    data_path = f'{base_path}/Juanmi_MED_PSA/MATLAB/workspaces/datos_consumo_Mcw_Mf.csv'
    fit_name = 'cooling_electrical_consumption'
    include_spline = False
    df = pd.read_csv(data_path)
    xrange = np.arange(151, 389)
    unit='W'

    x_data = df['FT-DES-002_VFD'][xrange] # (m³/h)
    y_data = df['PK-MED-E03-pa'][xrange]  # (W)

    #%% Brine pump consumption
    fit_name = 'brine_electrical_consumption'
    include_spline = True
    unit = 'kW'

    # df = pd.read_csv(data_path)
    # xrange = np.arange(151, 389)

    # Power consumption from VFD (kWh)
    y_data = np.array([0, 0, 0, 0, 0, 0.24, 0.26, 0.3, 0.33, 0.34, 0.35, 0.47, 0.42, 0.49, 0.65, 0.82, 1.2]) # kW
    # Flow rate (m³/h?)
    x_data = np.array([0.086, 0.126, 0.115, 0.12, 0.113, 2.146, 4.577, 6.22, 7.112, 7.465, 7.551, 8.122, 8.422, 9.442, 9.247, 9.572, 9.757]) # m³/h
    x_data = ensure_monotony(x_data)

    #%% Distillate pump consumption
    data_path = f'{os.environ["HOME"]}/Nextcloud/Juanmi_MED_PSA/Python/steady_state/csvs/distillate_pump_consumption.csv'
    fit_name = 'distillate_electrical_consumption'
    include_spline = False
    df = pd.read_csv(data_path)
    xrange = np.arange(8000, len(df)-1500)
    unit='kW'

    x_data = df['Mprod']
    y_data = df['Eprod']

    #%% Heat source pump consumption
    data_path = f'{base_path}/Juanmi_MED_PSA/MATLAB/workspaces/datos_consumo_Ms.csv'
    fit_name = 'hotwater_electrical_consumption'
    include_spline = False
    df = pd.read_csv(data_path)
    xrange = np.arange(241, 550)
    unit='kW'

    x_data = df['FT-AQU-100'][xrange]*3.6 # (m³/h)
    y_data = df['IT-DES-001'][xrange]-(df['IT-DES-001'].iloc[241:243]).mean()  # (kW)

    #%%
    fit = fit_curve(x_data, y_data, fit_name, unit=unit, include_spline = include_spline,
                    visualize_result=True, save_result=save_result, result_path=result_path)
