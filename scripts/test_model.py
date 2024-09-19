from pathlib import Path
import os
import hjson
import numpy as np
import pandas as pd
from IPython.display import display
from loguru import logger
import time

from solarmed_modeling.solar_med import SolarMED
from solarmed_modeling.utils import data_preprocessing, data_conditioning
from solarmed_modeling.utils.matlab_environment import set_matlab_environment
# from solarmed_modeling.metrics import calculate_metrics

set_matlab_environment()
logger.enable("solarmed_modeling")

# Paths definition
src_path = Path(f'{os.getenv("HOME")}/Nextcloud/Juanmi_MED_PSA/EURECAT/')
data_path: Path = src_path / 'data'
config_path: Path = Path('data')

# Constants
date_str: str = '20230703'
filename_process_data = f'{date_str}_solarMED.csv'
# filename_process_data = '20230505_solarMED.csv'
filename_process_data2 = f'{date_str}_MED.csv'

# Parameters
sample_rate = '60s'
# sample_rate = '300s'
cost_w: float = 3 # €/m³, cost of water
cost_e: float = 0.05 # €/kWh, cost of electricity

# Load configuration
with open(config_path / "variables_config.hjson") as f:
    vars_config = hjson.load(f)

# Data preprocessing
sample_rate_numeric = int(sample_rate[:-1])
data_paths = [data_path / filename_process_data, data_path / filename_process_data2]
# Load data and preprocess data
df = data_preprocessing(data_paths, vars_config, sample_rate_key=sample_rate)
# Condition data
df = data_conditioning(df, cost_w=cost_w, cost_e=cost_e, sample_rate_numeric=sample_rate_numeric)


# Evaluate model
idx_start = 1
span = 1
idx_end = len(df)
df_mod = pd.DataFrame()

# Initialize model
model = SolarMED(
    resolution_mode='simple',
    use_models=True,
    use_finite_state_machine=True,

    sample_time=sample_rate_numeric,

    # If a slow sample time is used, the solar field internal PID needs to be detuned
    # Ki_sf=-0.0001,
    # Kp_sf=-0.005,

    # Initial states
    ## Thermal storage
    Tts_h=[df['Tts_h_t'].iloc[idx_start], df['Tts_h_m'].iloc[idx_start], df['Tts_h_b'].iloc[idx_start]],
    Tts_c=[df['Tts_c_t'].iloc[idx_start], df['Tts_c_m'].iloc[idx_start], df['Tts_c_b'].iloc[idx_start]],

    ## Solar field
    Tsf_in_ant=df['Tsf_in'].iloc[idx_start - span:idx_start].values,
    msf_ant=df['qsf'].iloc[idx_start - span:idx_start].values,

    # cost_w = 3, # €/m³
    # cost_e = 0.05, # €/kWhe,
)

# Save model initial state and configuration
model_config = model.model_dump_configuration()
df_mod = model.to_dataframe(df_mod)

model_dump = model.model_dump()

# Run model
# %autoreload 2

start_time_total = time.time()

for idx in range(idx_start, idx_end):
    # idx = 1
    ds = df.iloc[idx]

    # logger.info(f"Iteration {idx} / {idx_end}")
    start_time = time.time()

    model.step(
        # Decision variables
        ## MED
        mmed_s=ds['qmed_s'],
        mmed_f=ds['qmed_f'],
        Tmed_s_in=ds['Tmed_s_in'],
        Tmed_c_out=ds['Tmed_c_out'],
        ## Thermal storage
        mts_src=ds['qhx_s'],
        ## Solar field
        Tsf_out=ds['Tsf_out'],

        med_vacuum_state=2,

        # Inputs
        # When the solar field is starting up, a flow can be provided to sync the model with the real system, if a valid Tsf_out is provided, it will be prioritized
        msf=ds['qsf'] if ds['qsf'] > 4 else None,

        # Environment variables
        Tmed_c_in=ds['Tmed_c_in'],
        Tamb=ds['Tamb'],
        I=ds['I'],
    )

    logger.info(
        f"Finished Iteration {idx} / {idx_end} - {df.index[idx]:%H:%M:%S}, elapsed time: {time.time() - start_time:.2f} seconds.")

    df_mod = model.to_dataframe(df_mod)

end_time_total = time.time()
logger.info(f"Total elapsed time: {end_time_total - start_time_total:.2f} seconds.")

#%%
# Sync model index with measured data
# df_mod.index = df.index[idx_start-1:idx if idx<idx_end-1 else idx_end]
#
# # Calculate metrics
# metrics =