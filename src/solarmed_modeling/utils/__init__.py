from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props

from solarmed_modeling.power_consumption import actuator_coefficients # Weird structure I know, should be in utils probably
from solarmed_modeling.curve_fitting import evaluate_fit

from phd_visualizations.utils.units import unit_conversion
from phd_visualizations.utils import rename_signal_ids_to_var_ids

# def filter_nan(data):
#     data_temp = data.dropna(how='any')
#     if len(data_temp) < len(data):
#         print(f"Some rows contain NaN values and were removed ({len(data) - len(data_temp)}).")
#
#     return data_temp
default_pressure_MPa: float = 0.16

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
    """Function to preprocess the dataframe before any further processing.
    In particular, it rounds the index to the nearest second, localizes the index to UTC timezone, and removes any
    duplicate index values.

    Args:
        df (pd.DataFrame): The input dataframe to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """

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

def data_preprocessing(paths: list[str | Path] | str | Path, vars_config: dict, sample_rate_key: str, fill_nans: bool = True) -> pd.DataFrame:

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

    index_cols = ['time', 'TimeStamp', 0]

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

    # Read reference dataframe
    logger.info(f"Reading data from {paths[0].name}")

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
    # Sample every `sample_rate` seconds
    df = df.resample(sample_rate_key).mean()

    # Read additional dataframes and concatenate them
    for idx, path in enumerate(paths[1:]):
        logger.info(f"Reading data from {path.name}")

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
        if common_columns.any():
            logger.debug(f"Common columns in both dataframes: {common_columns}, dropping them from the auxiliary dataframe.")

        df_aux = df_aux.drop(columns=common_columns)

        # Sample every `sample_rate` seconds
        df_aux = df_aux.resample(sample_rate_key).mean()

        # Align the two dataframes on their index, so they start and end at the same time
        df_aux, _ = df_aux.align(df, axis=0, join='right')

        df = pd.concat([df, df_aux], axis=1)


    # Preprocessing
    ## Fill nans, first forward and then backward
    df = df.ffill().bfill()

    ## Rename columns from signal_id to var_id
    df = rename_signal_ids_to_var_ids(df, vars_config)

    ## Convert units to model units
    df = unit_conversion(df, vars_config, input_unit_key='units_scada', output_unit_key='units_model')

    ## Filter out nans until first value in Tts
    logger.warning(f"Removing {df['Tts_h_t'].isna().sum()} NaNs from the dataframe")
    df = df[df['Tts_h_t'].notna()]

    return df

def data_conditioning(df: pd.DataFrame, vars_config: dict, cost_w:float=None, cost_e:float=None, sample_rate_numeric:int=None,
                      estimate_qhx_s: bool = True, fast: bool = True) -> pd.DataFrame:

    """
    This function conditions the data by:
        - Estimating the secondary flow rate of the heat exchanger
        - calculating the auxiliary variables (benefit, estimate states, solar field and ts power, etc)
        - Make sure some variables are within the allowed range.
    Args:
        df (pd.DataFrame): The input dataframe to be conditioned.
        cost_w (float): The cost of water in u.m./m³.
        cost_e (float): The cost of electricity in u.m./kWhe.
        sample_rate_numeric: The numeric value of the sample rate in seconds.

    Returns:
        pd.DataFrame: The conditioned dataframe.
    """

    # Check that Thx_p_in, Thx_s_in, Thx_p_out, Thx_s_out, qhx_p are in the dataframe
    required_columns = ['Thx_p_in', 'Thx_s_in', 'Thx_p_out', 'Thx_s_out', 'qhx_p']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f'Skipping estimation of qhx_s, missing required columns: {missing_columns}')
    else:
        from solarmed_modeling.heat_exchanger.utils import estimate_flow_secondary  # To avoid circular import errors
        
        # Estimate `qhx_s` since the measurement cannot be trusted
        water_props = None
        if fast:
            # Intialize water properties objects just once to increase performance, does not introduce much error
            water_props: tuple[w_props, w_props] = (w_props(P=default_pressure_MPa, T=(df['Thx_p_in'].median() + df['Thx_p_out'].median()) / 2 + 273.15), 
                                                    w_props(P=default_pressure_MPa, T=(df['Thx_s_in'].median() + df['Thx_s_out'].median()) / 2 + 273.15))
        
        # df["qhx_s_estimated"] = np.nan
        # df["qhx_s_estimated"] = df.apply(lambda row: estimate_flow_secondary(Tp_in=row['Thx_p_in'], Ts_in=row['Thx_s_in'], qp=row['qhx_p'], 
        #                                                                      Tp_out=row['Thx_p_out'], Ts_out=row['Thx_s_out'], water_props=water_props), axis=1)
        df["qhx_s_estimated"] = estimate_flow_secondary(Tp_in=df['Thx_p_in'].values, 
                                                        Ts_in=df['Thx_s_in'].values, 
                                                        qp=df['qhx_p'].values, 
                                                        Tp_out=df['Thx_p_out'].values, 
                                                        Ts_out=df['Thx_s_out'].values, 
                                                        water_props=water_props)

        # Where the heat exchanger was not operating normally, the estimation cannot be applied and the curve fit is used as an alternative
        nan_idxs = df['qhx_s_estimated'].isna()
        # Add idxs where UK-SF-P001-fq < 20, since the energy balance is not reliable if low or no flow is present
        if 'UK-SF-P001-fq' in df.columns:
            nan_idxs = nan_idxs | (df['UK-SF-P001-fq'] < 20) # 20%?

        try:
            # Va a dar error porque las trazas de SolarMED no contienen la señal UK-SF-P001-fq
            df.loc[nan_idxs, 'qhx_s_estimated'] = evaluate_fit(df.loc[nan_idxs, 'UK-SF-P001-fq'], fit_type='quadratic_curve', params=actuator_coefficients['FTSF001_calibration'])
        except KeyError:
            logger.warning(f'No data for UK-SF-P001-fq, using original signal to fill gaps in qhx_s_estimated ({np.sum(nan_idxs.values)} NaNs)')
            # Usar señal original, aunque no esté bien
            df.loc[nan_idxs, 'qhx_s_estimated'] = df.loc[nan_idxs, 'qhx_s']

        # Ensure non-negative values for qhx_s_estimated
        df['qhx_s_estimated'] = df['qhx_s_estimated'].clip(lower=0)

        # Rename qhx_s_estimated to qhx_s and copy it to qts_src, rename original signals to qhx_s_original and qts_src_original
        df.rename(columns={'qhx_s': 'qhx_s_original', 'qts_src': 'qts_src_original'}, inplace=True)
        df['qts_src'] = df['qhx_s_estimated']
        df.rename(columns={'qhx_s_estimated': 'qhx_s'}, inplace=True)

        logger.info('Heat exchanger secondary flow rate estimated successfully.')

    # Estimate `qts_dis`/`q3wv_src` since measurement from FT-AQU-101 cannot be trusted
    required_columns = ['q3wv_dis', 'T3wv_dis_in', 'T3wv_dis_out', 'T3wv_src']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f'Skipping estimation of qts_dis, missing required columns: {missing_columns}')
    else:
        from solarmed_modeling.three_way_valve.utils import estimate_flow_ts_discharge # To avoid circular import errors
        
        # if df['qts_dis'].mean() < 0.5:
        # De momento dar un warning si difieren mucho
        # logger.error('Measured flow rate from `qts,dis` is too low, probably flow sensor was not working properly. This will affect the outputs dependent on this variable (Pts,dis)')

        # Find flow upper limit from `qmed_s`
        # try:
        #     qts_dis_upper_limit = vars_config['qmed_s']['range'][1]
        # except Exception as e:
        #     qts_dis_upper_limit = 54 # m³/h
        #     logger.warning(f'Error while finding upper limit for qmed_s: {e}, defaulting to {qts_dis_upper_limit} m³/h')

        if fast:
            df["qts_dis_estimated"] = df["q3wv_dis"] * (df["T3wv_dis_in"] - df["T3wv_dis_out"]) / (df["T3wv_src"] - df["T3wv_dis_out"])
            df["qts_dis_estimated"] = df["qts_dis_estimated"].clip(lower=0)
            # df["qts_dis_estimated"] = df["qts_dis_estimated"].clip(upper=df["qmed_s"])

        else:
            df["qts_dis_estimated"] = df.apply(
                lambda row: estimate_flow_ts_discharge(qdis=row["q3wv_dis"], Tdis_in=row["T3wv_dis_in"], Tdis_out=row["T3wv_dis_out"], Tsrc=row["T3wv_src"], upper_limit_m3h=row["qmed_s"]), axis=1)

        # Make nan to zeros
        df['qts_dis_estimated'] = df['qts_dis_estimated'].fillna(0)

        # Rename qts_dis_estimated to qts_dis and also copy it to q3wv_src, rename original signals
        df.rename(columns={'qts_dis': 'qts_dis_original', 'q3wv_src': 'q3wv_src_orginal'}, inplace=True)
        df['q3wv_src'] = df['qts_dis_estimated']
        df.rename(columns={'qts_dis_estimated': 'qts_dis'}, inplace=True)

        logger.info('Thermal storage discharge flow rate estimated successfully (qts_dis/q3wv_src).')

    # Calculate auxiliary variables
    df = calculate_aux_variables(df, vars_config=vars_config, cost_w=cost_w, cost_e=cost_e, sample_rate_numeric=sample_rate_numeric, fast=fast)

    # Correct some (special) variables that might out of range
    try:
        # Tmed,c,in should not be greater than 28ºC
        logger.info(f'Corrected {np.sum(df["Tmed_c_in"] > 28)} values of Tmed_c_in above maximum allowed temperature (28ºC) to 28ºC.')
        df.loc[df['Tmed_c_in'] >= 28, 'Tmed_c_in'] = 27.9

        # qmed_f sometimes is about five but just lower, which fails the validation of the MED model, when the difference qmed_f - 5 > -0.2, it is set to 5
        qmed_f_condition = (np.abs(df['qmed_f'] - 5) < 0.2) & (df['qmed_f'] < 5)
        logger.info(f'Corrected {qmed_f_condition.sum()} values of qmed_f just below minimum allowed flow (5 - 0.2 m³/h), set to 5 m³/h.')
        df.loc[qmed_f_condition, 'qmed_f'] = 5

        # Tmed,c,out should not be smaller than 18ºC
        logger.info(f'Corrected {np.sum(df["Tmed_c_out"] < 18)} values of Tmed_c_out below minimum allowed temperature (18ºC) to 18ºC.')
        df.loc[df['Tmed_c_out'] < 18, 'Tmed_c_out'] = 18
    except Exception as e:
        logger.error(f'Error while conditioning MED variables: {e}')

    # try:
    #     # Make sure I is not negative
    #     df.loc[df['I'] < 0, 'I'] = 0
    # except Exception as e:
    #     logger.error(f'Error while conditioning solar field variables: {e}')

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
        
    # Correct any other variable with specified range in configuration
    df = enforce_operating_ranges(df, vars_config)

    return df

def resample_results(data: np.ndarray[float], current_index: pd.DatetimeIndex, new_index: pd.DatetimeIndex, reshape: bool = False) -> np.ndarray[float]:
    """Resample results to a new index"""
    values = pd.DataFrame(data, index=current_index).reindex(new_index, method='ffill').values
    
    return np.array(values).reshape(-1) if reshape else values


def enforce_operating_ranges(df: pd.DataFrame, vars_config: dict, var_ids_to_enforce: list[str] = None):
    """_summary_
    
    IMPORTANT! Dataframe should have model units!

    Args:
        df (pd.DataFrame): _description_
        vars_config (dict): _description_
        var_ids (tuple[str], optional): _description_. Defaults to None.
    """
    
    if var_ids_to_enforce is None:
        var_ids_to_enforce: list[str] = [c["var_id"] for c in vars_config.values() if "range_model" in c and c["var_id"] in df]
    
    # Gather lists for common signal identification
    var_ids: list[str] = []
    signal_ids: list[str] = []
    for c in vars_config.values():
        var_ids.append(c["var_id"])
        signal_ids.append(c["signal_id"])
        
    for var_id in var_ids_to_enforce:
        var_config = vars_config[var_id]
        
        if "range_model" not in var_config or var_id not in df:
            continue
        
        assert len(var_config["range_model"]) == 2, "Range should be a list of two elements, an upper and a lower limit, use `null` when either of them is not required"
        
        # Identify if var_id has mirroring variables in configuration, 
        # at least there will be one element (the variable itself)
        var_ids_with_common_signal_ids: list[str] = [v_id for v_id, s_id in zip(var_ids, signal_ids) if s_id==var_config["signal_id"]]
        
        # Enforce range
        lower_limit = var_config["range_model"][0]
        upper_limit = var_config["range_model"][1]
        for v_id in var_ids_with_common_signal_ids:
            if lower_limit is not None:
                df.loc[df[v_id] < lower_limit, v_id] = 0
            if upper_limit is not None:
                df[v_id] = df[v_id].clip(upper=upper_limit)
                
            logger.debug(f"Enforced ranges: [0, {lower_limit} < {v_id} < {upper_limit}] {var_config['units_model']}")
                
    return df

def calculate_costs(df: pd.DataFrame, cost_w: float, cost_e: float, 
                    sample_rate_numeric: int) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        cost_w (float): _description_ [u.m./m³]
        cost_e (float): _description_ [u.m./kWh]
        sample_rate_numeric (int): _description_

    Returns:
        _type_: _description_
    """
    # df = calculate_costs(df)

    df["benefit"] = (cost_w * df["qmed_d"] - cost_e * df["Jtotal"]) * sample_rate_numeric / 3600  # u.m.

    return df

def estimate_states(df: pd.DataFrame, vars_config: dict) -> pd.DataFrame:
    """
        NOTE: Units of dataframe should be in model units
    """
    df["sf_active"] = df["qsf"] > vars_config["qsf"]["range_model"][0]
    df["med_active"] = df["qmed_f"] > vars_config["qmed_f"]["range_model"][0]
    df["ts_active"] = df["qts_src"] > vars_config["qts_src"]["range_model"][0]

    return df

def calculate_powers_solar_field(row: pd.Series | pd.DataFrame, max_power: float = 250, min_power=0, 
                                 calculate_per_loop: bool = True, water_props: w_props = None) -> pd.Series | pd.DataFrame:
    """_summary_
    
    power = q [m³/h] / 3600 * rho [kg/m³] * cp [kJ/kgK] * ΔT [K\ºC]

    Args:
        row (pd.Series | pd.DataFrame): _description_
        max_power (float, optional): _description_. Defaults to 250.
        min_power (int, optional): _description_. Defaults to 0.
        calculate_per_loop (bool, optional): _description_. Defaults to True.
        water_props (w_props, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    # Solar field
    try:
        w_p = w_props(P=0.16, T=(row["Tsf_in"] + row["Tsf_out"]) / 2 + 273.15) if water_props is None else water_props
            
        row["Pth_sf"] = row["qsf"] / 3600 * w_p.rho * w_p.cp * (row["Tsf_out"] - row["Tsf_in"])  # kW
        row["Pth_sf"] = row["Pth_sf"].clip(lower=min_power, upper=max_power)
        row["Pth_hx_p"] = row["Pth_sf"]
    except Exception as e:
        logger.error(f'Error estimated solar field main loop power: missing {e} in data')

    # Solar field loops
    if calculate_per_loop:
        try:
            for loop_str in ['l2', 'l3', 'l4', 'l5']:
                loop_id = f"Psf_{loop_str}"
                row[loop_id] = row[f"qsf_{loop_str}"] / 3600 * w_p.rho * w_p.cp * (
                            row[f"Tsf_out_{loop_str}"] - row[f"Tsf_in_{loop_str}"])  # kW
                row[loop_id] = row[loop_id].clip(lower=min_power, upper=max_power)

        # row["Pth_sf"] = np.nan
        except Exception as e:
            logger.error(f'Error estimated solar field individual loops power: missing {e} in data')

    return row

def calculate_powers_thermal_storage(row: pd.Series | pd.DataFrame, max_power: float = 250, 
                                     min_power=0, water_props: tuple[w_props, w_props] = None) -> pd.Series | pd.DataFrame:
    """_summary_
    
    power = q [m³/h] / 3600 * rho [kg/m³] * cp [kJ/kgK] * ΔT [K\ºC]

    Args:
        row (pd.Series | pd.DataFrame): _description_
        max_power (float, optional): _description_. Defaults to 250.
        min_power (int, optional): _description_. Defaults to 0.
        water_props (tuple[w_props, w_props], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    try:
        # Thermal storage input powerdf.loc[df["qsf"] < qsf_min, "qsf"] = 0
        w_p = w_props(P=0.16, T=(row["Thx_s_out"] + row["Thx_s_in"]) / 2 + 273.15) if water_props is None else water_props[0]
        row["Pth_ts_src"] = row["qts_src"] / 3600 * w_p.rho * w_p.cp * (row["Thx_s_out"] - row["Thx_s_in"])  # kW
        row["Pth_ts_src"] = row["Pth_ts_src"].clip(lower=min_power, upper=max_power)
        row["Pth_hx_s"] = row["Pth_ts_src"]

        # Thermal storage output power
        w_p = w_props(P=0.16, T=(row["Tts_h_out"] + row["Tts_c_in"]) / 2 + 273.15) if water_props is None else water_props[1]
        row["Pth_ts_dis"] = row["qts_dis"] / 3600 * w_p.rho * w_p.cp * (row["Tts_h_out"] - row["Tts_c_in"])  # kW
        row["Pth_ts_dis"] = row["Pth_ts_dis"].clip(lower=min_power, upper=max_power)


    except Exception as e:
        logger.error(f'Error calculate thermal storage thermal power (source/discharge): missing {e} in data')
        # row["Pth_ts_src"] = np.nan
        # row["Pth_ts_dis"] = np.nan
        # row["Pth_sf"] = np.nan

    return row

def calculate_aux_variables(df: pd.DataFrame, sample_rate_numeric: int, vars_config: dict, cost_w: float = None, 
                            cost_e: float = None, fast: bool = True) -> pd.DataFrame:
    """Utility function that calculates auxiliary variables for the model:
        - MED and total electricity power consumption
        - Economic benefit
        - Estimate system operating states

    Args:
        df (pd.DataFrame): _description_
        cost_w (float): _description_
        cost_e (float): _description_
        sample_rate_numeric (int): _description_

    Returns:
        _type_: _description_
    """
    try:
        df["Jmed"] = df['Jmed_b'] + df['Jmed_c'] + df['Jmed_d'] + df['Jmed_s_f'] # kW
    except Exception as e:
        logger.error(f'Error while calculating MED electricity power consumption: {e}')
    else:
        logger.info('MED electricity power consumption calculated successfully.')
    try:
        Jtotal = 0
        if 'Jsf_ts' in df:
            Jtotal += df["Jsf_ts"]
        else:
            logger.warning('Solar field and thermal storage power consumption not found, total electricity power consumption will be calculated without it.')
        if 'Jmed' in df:
            Jtotal += df["Jmed"]
        else:
            logger.warning('MED power consumption not found, total electricity power consumption will be calculated without it.')
        df["Jtotal"] = Jtotal
    except Exception as e:
        logger.error(f'Error while calculating total electricity power consumption: {e}')
    else:
        logger.info('Total electricity power consumption calculated successfully.')

    if cost_w is not None and cost_e is not None:
        try:
            df = calculate_costs(df, cost_w, cost_e, sample_rate_numeric)
        except Exception as e:
            logger.error(f"Unable to calculate costs: {e}")
        else:
            logger.info("Costs calculated successfuly")
    else:
        logger.info("Costs not calculated, cost values not provided")
        
    try:
        if fast:
            water_props = w_props(P=default_pressure_MPa, T=(df['Thx_s_in'].median() + df['Thx_s_out'].median()) / 2 + 273.15)
            df = calculate_powers_solar_field(df, water_props = water_props)
        else:
            water_props = None
            df = df.apply(calculate_powers_solar_field, args=(250, 0, False, water_props), axis=1)
    except Exception as e:
        logger.error(f'Error while calculating solar field thermal power: {e}')
    else:
        logger.info('Solar field thermal power calculated successfully.')

    try:
        if fast:
            water_props = (w_props(P=default_pressure_MPa, T=(df['Thx_s_out'].median() + df['Thx_s_in'].median()) / 2 + 273.15),
                           w_props(P=default_pressure_MPa, T=(df['Tts_h_out'].median() + df['Tts_c_in'].median()) / 2 + 273.15))
            df = calculate_powers_thermal_storage(df, water_props = water_props)
        else:
            water_props = None
            df = df.apply(calculate_powers_thermal_storage, args=(250, 0, water_props), axis=1)
    except Exception as e:
        logger.error(f'Error while calculating thermal storage thermal power: {e}')
    else:
        logger.info('Thermal storage thermal power calculated successfully.')
        
    try:
        df = estimate_states(df, vars_config)
    except Exception as e:
        logger.error(f"Unable to estimate system states: {e}")
    else:
        logger.info("States estimated successfuly")

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