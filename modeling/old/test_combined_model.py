#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:35:49 2023

@author: patomareao

If using MATLAB code, see how to install the package in "Ejecución modelo MED v1.md"

and in linux before running this code need to export LD_LIBRARY_PATH:
    export MR=$HOME/PSA/MED_model
    export LD_LIBRARY_PATH=$MR/v911/runtime/glnxa64:$MR/v911/bin/glnxa64:$MR/v911/sys/os/glnxa64:$MR/v911/sys/opengl/lib/glnxa64

"""

""" Script to test the combined model """
# MATLAB model
import os
os.environ["MR"] = f"{os.environ['HOME']}/PSA/MATLAB_runtime"
MR = os.environ["MR"]
os.environ["LD_LIBRARY_PATH"] = f"{MR}/v911/runtime/glnxa64:{MR}/v911/bin/glnxa64:{MR}/v911/sys/os/glnxa64:{MR}/v911/sys/opengl/lib/glnxa64"

# print(os.environ["MR"])
# print(os.environ["LD_LIBRARY_PATH"])

import pandas as pd
import numpy as np
import time
from utils import generate_env_variables

from models_psa import med_storage_model



# Environment variables
total_time = pd.Timedelta(days=9) # Duration of the simulation
ts = 30 # Sample rate (seg)
L = int( total_time.total_seconds()/ts )

# Source: https://www.seatemperature.org/mediterranean-sea
mediterranean_sea_mean_temp_malaga = [16, 15, 15, 16, 17, 20, 22, 23, 22, 20, 18, 16] # ºC

env_vars_timeseries = generate_env_variables('datos/base_environment_data.csv', sample_period=total_time, sample_rate=ts)

Tts_t_in = env_vars_timeseries.Tts_t_in.values # ºC
Tamb = env_vars_timeseries.Tamb.values # ºC
mts_src = env_vars_timeseries.m_ts_src.values # L/min
Tmed_c_in = np.ones(L)*mediterranean_sea_mean_temp_malaga[7] # Hulio, ºC
wmed_f = np.ones(L)*35.5 # g/kg

# Thermal storage initial conditions
Tts_h = [83.95, 83.67, 75.99] 
Tts_c = [83.40, 83.50, 54.42]

# Decision variables should be updated/generated at every step, but to test 
# just create some series
mmed_s_sp     = 12*3.6 * np.ones(L)
mmed_f_sp     = 8  * np.ones(L)
Tmed_s_in_sp  = 65 * np.ones(L)
Tmed_c_out_sp = 28 * np.ones(L)

#%%
# Initialize combined system model
model = med_storage_model(
    ts=ts,
    curve_fits_path='datos/curve_fits.json',
    # Initial states. Thermal storage
    Tts_h=Tts_h, Tts_c=Tts_c,
)


# Run model
results = []
for idx in range(1, L):
    start_time = time.time()
    
    model.step(
        # Decision variables
        Tmed_s_in=float(Tmed_s_in_sp[idx]),
        Tmed_c_out=Tmed_c_out_sp[idx],
        mmed_s=float(mmed_s_sp[idx]),
        mmed_f=mmed_f_sp[idx],
        
        # Environment variables
        Tamb=Tamb[idx],
        Tmed_c_in=Tmed_c_in[idx],
        wmed_f=wmed_f[idx],
        
        mts_src=mts_src[idx],
        Tts_t_in=Tts_t_in[idx]
        )
    
    print(f'Iteration {idx} of {L} completed, took {time.time() - start_time:.3f} seconds')
    
    # Store system states
    results.append( model.get_properties() )

# Terminate 
model.terminate()

#%% Calculate stored energy evolution
from utils.thermal_storage import calculate_stored_energy

# Create time variable and extract results as a dataframe

x = env_vars_timeseries.index[1:len(results)+1].values
results_df = pd.DataFrame(results, index=x)

# Separate the columns of Tts_h into separate variables
for tank in ['h','c']:
    for idx, position in enumerate(['t','m','b']):
        results_df[f'Tts_{tank}_{position}'] = results_df[f'Tts_{tank}'].apply(lambda x: x[idx])


E_avail_h = calculate_stored_energy(results_df.Tts_h.values, np.array(model.Vts_h), model.Tmed_s_in_min)
E_avail_c = calculate_stored_energy(results_df.Tts_c.values, np.array(model.Vts_c), model.Tmed_s_in_min)