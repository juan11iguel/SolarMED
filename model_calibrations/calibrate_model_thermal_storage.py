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
from visualization.calibrations import (plot_model_result_thermal_storage, 
                                   plot_energy_thermal_storage)
from utils.constants import var_labels, var_names, colors
from utils.curve_fitting import polynomial_interpolation

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
os.chdir(os.path.join(os.getenv("HOME"), "Nextcloud/Juanmi_MED_PSA/EURECAT/models_psa"))

date_parser = lambda x: pd.to_datetime(x, format='%d-%b-%Y %H:%M:%S')
# data = pd.read_csv('datos/datos_tanques.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')

data = pd.read_csv('datos/datos_tanques.csv', parse_dates=['TimeStamp'], date_parser=date_parser)
#%%
data = data.rename(columns=var_names)
# data.set_index('time', inplace=True)
data = data.resample('10Min', on='time').mean()
ts = 600

#%% Visualize data

close('all')

idxs_src = (data["m_ts_src"] > 5).values
idxs_dis = (data["m_ts_dis"] > 0.1).values if "m_ts_dis" in data.columns else None
color_src = "#c01c28"
color_dis = "#1c71d8"

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

#%% Thermal storage model parameters fit
# Select a time window when tank is not being loaded nor discharged

data_val = data.iloc[xrange[0]:xrange[1]]
data_val.reset_index(drop=True, inplace=True) # reset index
# data_val = data_val.resample('10Min', on='time').mean()
# ts = 600


#%%% Test model prior to parameter fit
from models import thermal_storage_model    
from parameters_fit import calculate_iae, calculate_ise, calculate_itae

Tin_labels = ["Tts_h_t", "Tts_h_m", "Tts_c_t", "Tts_c_m"] # "Tts_h_b", "Tts_c_b"

# Parameters
N = 4 # Number of volumes
V = 30 # Total volume of the storage, m³
Tmin = 60 # Minimum useful temperature, °C
V_i = np.ones(N)*V/N  # Volume of each control volume

# Inputs
# Since we are using the first output as starting point, start from the second value
Ti_ant = np.array( [data_val[T][0] for T in Tin_labels] )
Tt_in = data_val.Tts_t_in.values[1:]
Tb_in = data_val.Tts_b_in.values[1:] if 'Tts_b_in' in data_val.columns else np.zeros(len(data_val)-1)

Tamb = data_val.Tamb.values[1:]
Qsrc = data_val.m_ts_src.values[1:] if 'm_ts_src' in data_val.columns else np.zeros(len(data_val)-1)
Qdis = data_val.m_ts_dis.values[1:] if 'm_ts_dis' in data_val.columns else np.zeros(len(data_val)-1)

msrc = np.zeros(len(data_val)-1, dtype=float); mdis = np.zeros(len(data_val)-1, dtype=float)
for idx in range(1, len(data_val)-1):
    msrc[idx] = Qsrc[idx]/60*w_props(P=0.1, T=Tt_in[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/min a kg/s
    mdis[idx] = Qdis[idx]*w_props(P=0.1, T=data_val.Tts_h_t[idx]+273.15).rho*1e-3 # rho [kg/m³] # Convertir L/s a kg/s

# Experimental outputs
Ti_ref = data_val[Tin_labels].values[1:]

# Unkown parameter initial guess
# UA0 = [0.6 for _ in range(N)]
UA0 = [1.3e-2, 1.1e-2, 1.3e-2, 1.1e-2, ] # 1e-1, 1e-1

Ti_mod = np.zeros((len(data_val)-1, N), dtype=float)
Ti_ant = Ti_ref[0]

for idx in range(len(data_val)-1):
    Ti_mod[idx] = thermal_storage_model(Ti_ant, Tt_in[idx], Tb_in[idx], Tamb[idx], msrc[idx], mdis[idx], UA0, V_i, N=N, ts=ts)
    Ti_ant = Ti_mod[idx]
    
iae  = calculate_iae(Ti_mod, Ti_ref)
ise  = calculate_ise(Ti_mod, Ti_ref)
itae = calculate_itae(Ti_mod, Ti_ref)

# Visualize result
plot_model_result_thermal_storage(N, Tin_labels, data_val, Ti_mod, UA0, V_i, itae, iae, ise)

#%%% Perform parameter fit

from parameters_fit import objective_function
# from scipy.optimize import Bounds

# Tt_in[idx], Tb_in[idx], Ti_ant, Tamb[idx], msrc[idx], mdis[idx], UA0, N=N

# Define your inputs and outputs
Ti_ant = Ti_ref[0]
inputs = [Ti_ant, Tt_in, Tb_in, Tamb, msrc, mdis]  # Input values
outputs = Ti_ref  # Actual output values
params = (V_i, N, ts)    # Constant model parameters
params_objective_function = {'metric': 'IAE', 'recursive':True, 'n_outputs':N}

L = len(outputs)  # Number of samples

# Set initial parameter values
# initial_parameters = [0.01 for _ in range(N)]
initial_parameters = [1.3e-2, 1.1e-2, 1.3e-2, 1.1e-2] # 1e-1, 1e-1,
bounds = ((1e-4, 1) for _ in range(N))

# Perform parameter calibration
optimized_parameters = scipy.optimize.minimize(
    objective_function,
    initial_parameters,
    args=(thermal_storage_model, inputs, outputs, params, params_objective_function),
    bounds = bounds
).x

#%% Use the optimized parameters to evaluate model

# To not have to run the optimization  every time
# optimized_parameters = np.array([0.02136564, 0.01593324, 0.01918577, 0.01323035])

predicted_outputs = np.zeros((L, N), dtype=float)  
for idx in range(0, L):
    current_inputs = [inputs[i][idx] for i in range(1, len(inputs))]
    current_inputs.insert(0, inputs[0])
    
    predicted_outputs[idx] = thermal_storage_model(*current_inputs, optimized_parameters, *params)
    inputs[0] = predicted_outputs[idx]

iae  = calculate_iae(Ti_mod, Ti_ref)
ise  = calculate_ise(Ti_mod, Ti_ref)
itae = calculate_itae(Ti_mod, Ti_ref)

# Visualize model performance with optimized parameters
plot_model_result_thermal_storage(N, Tin_labels, data_val, predicted_outputs, optimized_parameters, V_i, itae, iae, ise)

#%% Once UA is fitted, fit V_i evaluating model during load and discharge data
from parameters_fit import objective_function

Tin_labels = ["Tts_h_t", "Tts_h_m", "Tts_c_t", "Tts_c_m"] # "Tts_h_b", "Tts_c_b"

# Parameters
N = 4 # Number of volumes
V = 30 # Total volume of the storage, m³
Tmin = 60 # Minimum useful temperature, °C
ts = 600
UA = optimized_parameters

# Outputs
Ti_ref = data[Tin_labels].values[1:]

# Inputs
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

Ti_ant = Ti_ref[0]
inputs = [Ti_ant, Tt_in, Tb_in, Tamb, msrc, mdis, UA]  # Input values
outputs = Ti_ref  # Actual output values
params = (N, ts)    # Constant model parameters
params_objective_function = {'metric': 'IAE', 'recursive':True, 'n_outputs':N}

# Set initial parameter values
# initial_parameters = [0.01 for _ in range(N)]
initial_parameters = np.ones(N)*V/N # 1e-1, 1e-1,
bounds = ((0.1*V/N, 2*V/N) for _ in range(N))

# Perform parameter calibration
optimized_parameters = scipy.optimize.minimize(
    objective_function,
    initial_parameters,
    args=(thermal_storage_model, inputs, outputs, params, params_objective_function),
    bounds = bounds
).x


# Use the optimized parameters to evaluate model
# optimized_parameters = np.array([6.76134767, 4.66134767, 6.76134767, 1.6134767])
Ti_ant = Ti_ref[0]
for idx in range(len(data)-1):
    Ti_mod[idx] = thermal_storage_model(Ti_ant, Tt_in[idx], Tb_in[idx], Tamb[idx], msrc[idx], mdis[idx], UA, N=N, ts=ts)
    Ti_ant = Ti_mod[idx]
    
iae  = calculate_iae(Ti_mod, Ti_ref)
ise  = calculate_ise(Ti_mod, Ti_ref)
itae = calculate_itae(Ti_mod, Ti_ref)

# Visualize result
plot_model_result_thermal_storage(N, Tin_labels, data, Ti_mod, UA, V_i, itae, iae, ise)


#%% Second approach, calibrate UA and V_i at the same time by evaluating model 
#   during load and discharge data

from parameters_fit import objective_function

Tin_labels = ["Tts_h_t", "Tts_h_m", "Tts_c_t", "Tts_c_m"] # "Tts_h_b", "Tts_c_b"

# Parameters
N = 4 # Number of volumes
V = 30 # Total volume of the storage, m³
Tmin = 60 # Minimum useful temperature, °C
ts = 600

# Outputs
Ti_ref = data[Tin_labels].values[1:]

# Inputs
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

Ti_ant = Ti_ref[0]
inputs = [Ti_ant, Tt_in, Tb_in, Tamb, msrc, mdis]  # Input values
outputs = Ti_ref  # Actual output values
params = (N, ts)    # Constant model parameters
params_objective_function = {'metric': 'IAE', 'recursive':True, 'n_outputs':N, 'parameter_groups': 2}

# Set initial parameter values
# initial_parameters = [0.01 for _ in range(N)]
initial_parameters = np.concatenate((np.array([0.02136564, 0.01593324, 0.01918577, 0.01323035]), np.ones(N)*V/N)) # 1e-1, 1e-1,
bounds = ((1e-4, 1),) * N + ((0.1*V/N, 2*V/N),) * N

# Perform parameter calibration
optimized_parameters = scipy.optimize.minimize(
    objective_function,
    initial_parameters,
    args=(thermal_storage_model, inputs, outputs, params, params_objective_function),
    bounds = bounds,
    method='L-BFGS-B'
).x

UA  = optimized_parameters[:int(len(optimized_parameters)/2)]
V_i = optimized_parameters[int(len(optimized_parameters)/2):len(optimized_parameters)]

# optimized_parameters = array([6.77724155e-03, 3.96580419e-03, 2.87258611e-02, 6.88542188e-03, 2.97217468e+00, 1.71277001e+00, 9.43455760e+00, 3.78073750e+00])
Ti_ant = Ti_ref[0]
for idx in range(len(data)-1):
    Ti_mod[idx] = thermal_storage_model(Ti_ant, Tt_in[idx], Tb_in[idx], Tamb[idx], msrc[idx], mdis[idx], UA, V_i, N=N, ts=ts)
    Ti_ant = Ti_mod[idx]
    
iae  = calculate_iae(Ti_mod, Ti_ref)
ise  = calculate_ise(Ti_mod, Ti_ref)
itae = calculate_itae(Ti_mod, Ti_ref)

# Visualize result
plot_model_result_thermal_storage(N, Tin_labels, data, Ti_mod, UA, V_i, itae, iae, ise)



#%% Calculate energy stored
# Define minimum temperature threshold
Tmin = 84

volumes = V_i
temperatures = Ti_mod[100]

# Perform polynomial interpolation for temperatures
interp_volumes = np.linspace(np.min(volumes), np.max(volumes), num=100)  # Generate interpolated volume points
interp_temperatures = polynomial_interpolation(volumes, temperatures, interp_volumes)

volume_above_Tmin = interp_volumes[interp_temperatures > Tmin]
temperatures_above_Tmin = interp_temperatures[interp_temperatures > Tmin]

# Calculate the energy stored above Tmin
# Define specific heat capacity (cp)
cp  = np.array([w_props(T=T+273.15, P=0.1).cp if T>=Tmin else 0 for T in interp_temperatures])
rho = np.array([w_props(T=T+273.15, P=0.1).rho if T>=Tmin else 0 for T in interp_temperatures])

# Calculate the temperature differences above Tmin
temperature_diff = np.maximum(0, interp_temperatures - Tmin)

# Calculate the energy
energy = np.sum(interp_volumes * rho * cp * temperature_diff)/3600 # m³·kg/m³·KJ/KgK·K == kWh

#%% Visualization
plot_energy_thermal_storage(volumes, temperatures, Tmin, energy)