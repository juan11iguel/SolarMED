import numpy as np
import pandas as pd
import warnings
from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar


def three_way_valve_model(Mdis, Tsrc, Tdis_in, Tdis_out):
    """Three-way valve steady state model.

    Args:
        Mdis (float): Discharge flow rate [m^3/h or any]
        Tdis_in (float): Discharge / load inlet temperature [ºC]
        Tdis_out (float): Discharge / load outlet temperature [ºC]
        Tsrc (float): Source temperature [ºC]

    Returns:
        Msrc: Source flow rate [Same units as Mdis]
        R: Ratio of mixing, se puede usar, tras ajustar una curva de apertura de válvula,
           para implementar un controlador con feedforward
    """

    if Mdis == 0:
        return 0, 0.5


    Tsrc = Tsrc + 273.15
    Tdis_in = Tdis_in + 273.15
    Tdis_out = Tdis_out + 273.15

    cp_src = w_props(P=0.1, T=Tsrc).cp  # P=0.1 bar->0.1 MPa, T [K], cp [kJ/kg·K]
    cp_dis_in = w_props(P=0.1, T=Tdis_in).cp  # P=0.1 bar->0.1 MPa, T [K], cp [kJ/kg·K]
    cp_dis_out = w_props(P=0.1, T=Tdis_out).cp  # P=0.1 bar->0.1 MPa, T [K], cp [kJ/kg·K]

    # Msrc = Mdis * ( Tdis_in*cp_dis_in  - Tdis_out*cp_dis_out ) / ( Tsrc*cp_src - Tdis_out*cp_dis_out )
    R = (Tdis_in * cp_dis_in - Tsrc * cp_src) / (Tdis_out * cp_dis_out - Tsrc * cp_src)

    # Saturation
    if R > 1:
        R = 1
    elif R < 0:
        R = 0

    Msrc = Mdis * (1 - R)

    return Msrc, R

def get_Q_from_3wv_model(datos_name, vars_info: dict, sample_rate_str='1Min'):
    
    warnings.warn("The function get_Q_from_3wv_model is deprecated and will be removed in future versions.", DeprecationWarning)
    
    var_names = {v["signal_id"]: k for k, v in vars_info.items() if '3wv' in k or k not in '_'}

    # Read data
    data = pd.read_csv(f'datos/datos_valvula_{datos_name}.csv', parse_dates=['TimeStamp'],
                       date_format='%d-%b-%Y %H:%M:%S')

    data = data.rename(columns=var_names)
    data = data.resample(sample_rate_str, on='time').mean()

    # Remove rows with NaN values in place and generate a warning
    data_temp = data.dropna(how='any')
    if len(data_temp) < len(data):
        print(f"Some rows contain NaN values and were removed ({len(data) - len(data_temp)}).")
    data = data_temp.copy()
    del data_temp

    # Inputs
    Mdis = data.m_3wv_dis.values
    Tsrc = data.T_3wv_src2.values
    T_dis_in = data.T_3wv_dis_in.values
    T_dis_out = data.T_3wv_dis_out.values

    # Initialize result vectors
    Msrc_mod = np.zeros(len(data), dtype=float)

    # Evaluate model
    for idx in range(len(data)):
        Msrc_mod[idx], _ = three_way_valve_model(Mdis[idx], Tsrc[idx], T_dis_in[idx], T_dis_out[idx])

    # Create a new dataframe with the output from the model
    data_mod = pd.DataFrame({'m_ts_dis': Msrc_mod}, index=data.index)

    return data_mod  # L/s
