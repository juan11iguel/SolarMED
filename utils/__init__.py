#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:06:32 2023

@author: patomareao
"""

from utils.constants import vars_info
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
import models 

def get_Q_from_3wv_model(datos_name, sample_rate_str='1Min'):
    var_names = {v["signal_id"]: k for k, v in vars_info.items() if '3wv' in k or not '_' in k}

    # Read data
    data = pd.read_csv(f'datos/datos_valvula_{datos_name}.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
    
    data = data.rename(columns=var_names)
    data = data.resample(sample_rate_str, on='time').mean()
    
    # Remove rows with NaN values in place and generate a warning
    data_temp = data.dropna(how='any')
    if len(data_temp) < len(data):
        print(f"Some rows contain NaN values and were removed ({len(data)-len(data_temp)}).")
    data = data_temp.copy()
    del data_temp
    
    # Inputs
    Mdis = data.m_3wv_dis.values
    Tsrc = data.T_3wv_src2.values
    T_dis_in  = data.T_3wv_dis_in.values
    T_dis_out = data.T_3wv_dis_out.values
    
    # Initialize result vectors
    Msrc_mod = np.zeros(len(data), dtype=float)
    
    # Evaluate model
    for idx in range(len(data)):
        Msrc_mod[idx], _ = models.three_way_valve_model(Mdis[idx], Tsrc[idx], T_dis_in[idx], T_dis_out[idx])
        
    # Create a new dataframe with the output from the model
    data_mod = pd.DataFrame({'m_ts_dis': Msrc_mod}, index=data.index)
        
    return data_mod # L/s

def filter_nan(data):
    data_temp = data.dropna(how='any')
    if len(data_temp) < len(data):
        print(f"Some rows contain NaN values and were removed ({len(data)-len(data_temp)}).")
    
    return data_temp


def generate_env_variables(base_data_path, sample_period=pd.Timedelta(days=9), sample_rate:int=30, visualize_result=True) -> pd.DataFrame:

    # Define the given sample length
    # given_sample_length = pd.Timedelta(days=9)  # Adjust this value as needed
    # base_timeseries = pd.read_csv(base_data_path, parse_dates=['time'], date_parser=date_parser)
    base_timeseries = pd.read_csv(base_data_path, parse_dates=['time'], date_format='ISO8601')

    base_timeseries.set_index('time', inplace=True)
    
    # Get the first and last timestamp in the selected data
    start_time = base_timeseries.index[0]; end_time = base_timeseries.index[-1]
    
    # Calculate the time difference between start and end timestamps
    time_diff = end_time - start_time
    
    # Create a regular time series with the desired frequency using resample and linear interpolation
    base_timeseries = base_timeseries.resample(f'{sample_rate}S').interpolate(method='linear')
    
    # Calculate the number of periods that fit in the given sample length
    num_periods = sample_period // time_diff
    
    # Calculate the remaining duration after the last complete period
    remaining_duration = sample_period - (num_periods * time_diff)
    
    # Create a new time index spanning the desired duration
    new_time_index = pd.date_range(start=start_time, end=start_time + sample_period, freq=f'{sample_rate}S')
    
    # Repeat the selected data as many times as needed to cover the desired duration
    output_timeseries = pd.concat([base_timeseries.set_index(base_timeseries.index + i * time_diff) for i in range(num_periods)])
    
    # Adjust the time component of the end portion to accumulate over the previous value
    end_portion = base_timeseries.iloc[:int(remaining_duration.total_seconds() // sample_rate)].copy()
    end_portion.index = new_time_index[-len(end_portion):]
    
    # Concatenate the repeated selected data with the end portion
    output_timeseries = pd.concat([output_timeseries, end_portion])
        
    
    if visualize_result:
        num_cols = 2
        num_rows = base_timeseries.shape[1] // num_cols
        num_rows = num_rows +1 if base_timeseries.shape[1] % num_cols > 0 else num_rows
        
        # Create the subplots with shared x-axis
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, figsize=(10, 6))
        
        # Loop through each column and plot in the corresponding subplot
        for i, column in enumerate(base_timeseries.columns):
            axs[i // num_cols, i % num_cols].plot(output_timeseries.index, output_timeseries[column])
            axs[i // num_cols, i % num_cols].plot(base_timeseries.index, base_timeseries[column])
            
            axs[i // num_cols, i % num_cols].set_ylabel(column)
                
        # Adjust layout and show the plot
        # plt.tight_layout()
        plt.show()
        
    return output_timeseries



#%% Test environment variable generation


if __name__ == '__main__':

    ts = 30    

    from constants import vars_info
    var_names = {v["signal_id"]: k for k, v in vars_info.items() if 'ts' in k or not '_' in k}
    
    data = pd.read_csv('datos/datos_tanques.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
    data = data.rename(columns=var_names)
        
    data = data.resample(f'{ts}S', on='time').mean()

    mask = (data.index >= '2023-07-08 05:00:00') & (data.index <= '2023-07-10 05:00:00')
    selected_data = data[mask]
    selected_data = selected_data[ ['Tts_t_in', 'm_ts_src', 'Tamb'] ]
    selected_data.to_csv('base_environment_data.csv', index=True)

    environment_vars_timeseries = generate_env_variables('base_environment_data.csv', sample_rate = ts)