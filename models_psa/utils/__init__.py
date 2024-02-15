#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:06:32 2023

@author: patomareao
"""

# from utils.constants import vars_info
import numpy as np

# def filter_nan(data):
#     data_temp = data.dropna(how='any')
#     if len(data_temp) < len(data):
#         print(f"Some rows contain NaN values and were removed ({len(data) - len(data_temp)}).")
#
#     return data_temp


def ensure_monotony(data: np.array, gap=0.01) -> np.array:
    """
    This function ensures that the given data is monotonically increasing. If a value is found to be less than its
    predecessor, it is replaced with the value of its predecessor plus a small gap.

    Parameters:
    data (np.array): The input data array that needs to be checked for monotonicity.
    gap (float, optional): The minimum difference between consecutive values in the output array. Default is 0.01.

    Returns:
    np.array: The input data array adjusted to ensure monotonicity.
    """

    # Iterate over the data array starting from the second element
    for i in range(1, len(data)):
        # If the current element is less than its predecessor
        if data[i] < data[i - 1]:
            # Replace it with the value of its predecessor plus the gap
            data[i] = data[i - 1] + gap

    return data


# %% Test environment variable generation


# if __name__ == '__main__':
#     ts = 30
#
#     from constants import vars_info
#
#     var_names = {v["signal_id"]: k for k, v in vars_info.items() if 'ts' in k or not '_' in k}
#
#     data = pd.read_csv('datos/datos_tanques.csv', parse_dates=['TimeStamp'], date_format='%d-%b-%Y %H:%M:%S')
#     data = data.rename(columns=var_names)
#
#     data = data.resample(f'{ts}S', on='time').mean()
#
#     mask = (data.index >= '2023-07-08 05:00:00') & (data.index <= '2023-07-10 05:00:00')
#     selected_data = data[mask]
#     selected_data = selected_data[['Tts_t_in', 'm_ts_src', 'Tamb', 'I']]
#     selected_data.to_csv('base_environment_data.csv', index=True)
#
#     environment_vars_timeseries = generate_env_variables('base_environment_data.csv', sample_rate=ts)