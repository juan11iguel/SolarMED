#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:06:32 2023

@author: patomareao
"""

# from utils.constants import vars_info
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from solarMED_modeling import calculate_aux_variables, calculate_powers_solar_field, calculate_powers_thermal_storage
from solarMED_modeling.heat_exchanger import estimate_flow_secondary
from solarMED_modeling.power_consumption import actuator_coefficients # Weird structure I know, should be in utils probably
from solarMED_modeling.curve_fitting import evaluate_fit


from phd_visualizations.utils.units import unit_conversion
from phd_visualizations.utils import rename_signal_ids_to_var_ids

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


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df.index = df.index.round('s')
    df = df.tz_localize('UTC')
    if 'time.1' in df.columns:
        df.drop(columns='time.1', inplace=True)

    # Select columns of 'object' data type
    object_columns = df.select_dtypes(include=['object'])

    # Print the column names
    logger.debug(object_columns.columns)  # %%

    if not object_columns.empty:
        logger.warning(f"Columns with object data type: {object_columns.columns}")

    # Drop duplicate index values from df
    duplicates_df = df.index.duplicated(keep='first').sum()
    logger.info(f"Number of duplicate index values in df: {duplicates_df}")
    df = df[~df.index.duplicated(keep='first')]

    return df

def data_preprocessing(paths: list[str, Path] | str | Path, vars_config: dict, sample_rate_key: str) -> pd.DataFrame:

    """
    This function reads the data from the provided paths, concatenates the dataframes and preprocesses the data.
    This processing includes:
    - Resampling the data to the provided sample rate.
    - Renaming the columns from signal_id to var_id using the information contained in `vars_config`.
    - Converting the units of the variables to the model units using the information contained in `vars_config`.
    - Filtering out NaNs until the first value in Tts.

    Args:
        paths: The path or list of paths to the CSV files containing the data.
        vars_config: The configuration dictionary containing the information about the variables.
        sample_rate_key: The pandas key of the sample rate.

    Returns:
        pd.DataFrame: The preprocessed dataframe.

    """

    index_cols = ['time', 'TimeStamp']

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

    # Read reference dataframe
    df = None
    for index_col in index_cols:
        try:
            df = pd.read_csv(paths[0], parse_dates=True, index_col=index_col)
            break
        except Exception as e:
            pass
            # logger.error(f'Error while reading data from {paths[0]} with index_col={index_col}: {e}')

    if df is None:
        raise RuntimeError(f'Failed to read data from CSV file with any of the provided index columns: {index_cols}')

    df = process_dataframe(df)

    # Read additional dataframes and concatenate them
    for idx, path in enumerate(paths[1:]):
        df_aux = None
        for index_col in index_cols:
            try:
                df_aux = pd.read_csv(path, parse_dates=True, index_col=index_col)
                df_aux.index.names = ['time']
                break
            except Exception as e:
                logger.error(f'Error while reading data from {path} with index_col={index_col}: {e}')

        if df_aux is None:
            logger.error(f'Failed to read data from CSV file with any of the provided index columns for the {idx+2} path: {index_cols}')

        df_aux = process_dataframe(df_aux)

        # Find the common columns in both dataframes and drop them from the second
        common_columns = df.columns.intersection(df_aux.columns)
        df_aux = df_aux.drop(columns=common_columns)
        df = pd.concat([df, df_aux], axis=1)

    # Preprocessing
    # Sample every `sample_rate` seconds to reduce the size of the dataframe
    df = df.resample(sample_rate_key).mean()

    # Rename columns from signal_id to var_id
    df = rename_signal_ids_to_var_ids(df, vars_config)

    # Convert units to model units
    df = unit_conversion(df, vars_config, input_unit_key='units_scada', output_unit_key='units_model')

    # Filter out nans until first value in Tts
    logger.warning(f"Removing {df['Tts_h_t'].isna().sum()} NaNs from the dataframe")
    df = df[df['Tts_h_t'].notna()]

    return df


def data_conditioning(df: pd.DataFrame, cost_w:float=None, cost_e:float=None, sample_rate_numeric:int=None) -> pd.DataFrame:

    """
    This function conditions the data by:
        - Estimating the secondary flow rate of the heat exchanger
        - calculating the auxiliary variables (benefit, cost, estimate states)
        - calculating the power consumption of the solar field and thermal storage.
        - Make sure some variables are within the allowed range.
    Args:
        df:
        cost_w:
        cost_e:
        sample_rate_numeric:

    Returns:

    """

    # Check that Thx_p_in, Thx_s_in, Thx_p_out, Thx_s_out, qhx_p are in the dataframe
    required_columns = ['Thx_p_in', 'Thx_s_in', 'Thx_p_out', 'Thx_s_out', 'qhx_p']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f'Skipping estimation of qhx_s, missing required columns: {missing_columns}')
    else:
        # Estimate `qhx_s` since the measurement cannot be trusted
        df["qhx_s_estimated"] = np.nan
        df["qhx_s_estimated"] = df.apply(lambda row: estimate_flow_secondary(Tp_in=row['Thx_p_in'], Ts_in=row['Thx_s_in'], qp=row['qhx_p'], Tp_out=row['Thx_p_out'], Ts_out=row['Thx_s_out']), axis=1)

        # Where the heat exchanger was not operating normally, the estimation cannot be applied and the curve fit is used as an alternative
        nan_idxs = df['qhx_s_estimated'].isna()
        # Add idxs where UK-SF-P001-fq < 20, since the energy balance is not reliable if low or no flow is present
        nan_idxs = nan_idxs | (df['UK-SF-P001-fq'] < 20) # 20%?

        # Va a dar error porque las trazas de SolarMED no contienen la señal UK-SF-P001-fq
        try:
            df.loc[nan_idxs, 'qhx_s_estimated'] = evaluate_fit(df.loc[nan_idxs, 'UK-SF-P001-fq'], fit_type='quadratic_curve', params=actuator_coefficients['FTSF001_calibration'])
        except KeyError:
            logger.warning(f'No data for UK-SF-P001-fq, using original signal to fill gaps in qhx_s_estimated ({np.sum(nan_idxs.values)} NaNs)')
            # Usar señal original, aunque no esté bien
            df.loc[nan_idxs, 'qhx_s_estimated'] = df.loc[nan_idxs, 'qhx_s']

        # Make sure there are non-negative values
        df.loc[df['qhx_s_estimated'] < 0, 'qhx_s_estimated'] = 0

        # Rename qhx_s_estimated to qhx_s and copy it to qts_src, rename original signals to qhx_s_original and qts_src_original
        df.rename(columns={'qhx_s': 'qhx_s_original', 'qts_src': 'qts_src_original'}, inplace=True)
        df['qts_src'] = df['qhx_s_estimated']
        df.rename(columns={'qhx_s_estimated': 'qhx_s'}, inplace=True)

        logger.info('Heat exchanger secondary flow rate estimated successfully.')

    # TODO: Estimate `qts_dis` since measurement from FT-AQU-101 cannot be trusted
    # Estimar a partir de válvula de tres vías () y FT-AQU-100 (mmed,s). Pendiente de hacer en cuaderno
    # model_calibrations / calibrate_three_way_valve.ipynb y luego incluir aquí
    # De momento dar un warning si difieren mucho
    if df['qts_dis'].mean() < 0.5:
        logger.error('Measured flow rate from `qts,dis` is too low, probably flow sensor was not working properly. This will affect the outputs dependent on this variable (Pts,dis)')

    try:
        df = calculate_aux_variables(df, cost_w=cost_w, cost_e=cost_e, sample_rate_numeric=sample_rate_numeric)
    except Exception as e:
        logger.error(f'Error while calculating auxiliary variables: {e}')
    else:
        logger.info('Auxiliary variables calculated successfully.')

    try:
        df = df.apply(calculate_powers_solar_field, args=(250, 0, False), axis=1)
    except Exception as e:
        logger.error(f'Error while calculating solar field power: {e}')
    else:
        logger.info('Solar field power calculated successfully.')

    try:
        df = df.apply(calculate_powers_thermal_storage, axis=1)
    except Exception as e:
        logger.error(f'Error while calculating thermal storage power: {e}')
    else:
        logger.info('Thermal storage power calculated successfully.')

    try:
        # This should be in `calculate_aux_variables`
        df["Jmed"] = df['Jmed_b'] + df['Jmed_c'] + df['Jmed_d'] + df['Jmed_s_f'] # kW
        # TODO: Add electrical consumption of thermal storage and solar field and vacuum system
        df["Jts"] = 0 # kW
        df["Jsf"] = 0 # kW

        df["Jtotal"] = df["Jmed"] + df["Jts"] + df["Jsf"] # kW
    except Exception as e:
        logger.error(f'Error while calculating total electricity power consumption: {e}')
    else:
        logger.info('Total electricity power consumption calculated successfully.')

    try:
        # Tmed,c,in should not be greater than 28ºC
        logger.info(f'Corrected {np.sum(df["Tmed_c_in"] > 28)} values of Tmed_c_in above maximum allowed temperature (28ºC) to 28ºC.')
        df.loc[df['Tmed_c_in'] >= 28, 'Tmed_c_in'] = 27.9

        # qmed_f sometimes is about five but just lower, which fails the validation of the MED model, when the difference qmed_f - 5 > -0.2, it is set to 5
        qmed_f_condition = (np.abs(df['qmed_f'] - 5) < 0.2) & (df['qmed_f'] < 5)
        logger.info(f'Corrected {qmed_f_condition.sum()} values of qmed_f just below minimum allowed flow (5 - 0.2 m³/h), set to 5 m³/h.')
        df.loc[qmed_f_condition, 'qmed_f'] = 5

        # Tmed,c,out should bot be smaller than 18ºC
        logger.info(f'Corrected {np.sum(df["Tmed_c_out"] < 18)} values of Tmed_c_out below minimum allowed temperature (18ºC) to 18ºC.')
        df.loc[df['Tmed_c_out'] < 18, 'Tmed_c_out'] = 18

    except Exception as e:
        logger.error(f'Error while conditioning MED variables: {e}')

    try:
        # Make sure I is not negative
        df.loc[df['I'] < 0, 'I'] = 0
    except Exception as e:
        logger.error(f'Error while conditioning solar field variables: {e}')

    # Check that none of the model inputs is nan
    try:
        model_inputs = ['qmed_s','qmed_f','Tmed_s_in','Tmed_c_out', 'qhx_s', 'Tsf_out','qsf', 'Tmed_c_in','Tamb','I']
        nan_values = df[model_inputs].isna().sum()

        if nan_values.sum() > 0:
            # Remove rows with NaN values
            logger.warning(f"Removing {nan_values.sum()} rows with NaN values in model inputs.")
            df = df.dropna(subset=model_inputs)

            # Interpolate the data to fill the gaps
            df = df.interpolate(method='time')
    except Exception as e:
        logger.error(f'Error while checking model inputs for NaN values: {e}')

    return df

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