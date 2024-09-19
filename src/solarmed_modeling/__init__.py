from enum import Enum
from loguru import logger
from iapws.iapws97 import IAPWS97 as w_props

# States definition
class SolarFieldState(Enum):
    IDLE = 0
    ACTIVE = 1


class ThermalStorageState(Enum):
    IDLE = 0
    ACTIVE = 1

class SF_TS_State(Enum):
    """First digit: Solar Field, Second digit: Thermal Storage"""
    IDLE = '00'
    RECIRCULATING_TS = '01'
    HEATING_UP_SF = '10'
    SF_HEATING_TS = '11'

SfTsState_with_value = Enum('SfTsState_with_value', {
    f'{state.name}': i
    for i, state in enumerate(SF_TS_State)
})

class MedVacuumState(Enum):
    OFF = 0
    LOW = 1
    HIGH = 2


class MedState(Enum):
    OFF = 0
    GENERATING_VACUUM = 1
    IDLE = 2
    STARTING_UP = 3
    SHUTTING_DOWN = 4
    ACTIVE = 5


# More descriptive manually typed names


SolarMED_State = Enum('SolarMED_State', {
    f'sf_{sf_state.name}_ts_{ts_state.name}_med_{med_state.name}': f'{sf_state.value}{ts_state.value}{med_state.value}'
    for sf_state in SolarFieldState
    for ts_state in ThermalStorageState
    for med_state in MedState
})

SolarMedState_with_value = Enum('SolarMedState_with_value', {
    f'{state.name}': i
    for i, state in enumerate(SolarMED_State)
})

SupportedStatesType = MedState | SolarFieldState | ThermalStorageState | SolarMED_State | SF_TS_State


# TODO: Everything below this line should be moved to utils

def calculate_benefits(df, cost_w, cost_e, sample_rate_numeric):

    df = calculate_costs(df)

    df["B"] = (cost_w * df["qmed_d"] - cost_e * df["Ce"]) * sample_rate_numeric / 3600  # u.m.

    return df

def calculate_costs(df):
    # TODO: Add additional consumptions (solar field and thermal storage)

    if "Jmed" in df.columns:
        # Already provided the MED total consumption
        df["Ce"] = df['Jmed'] * 1e-3
    else:
        # Calculate the MED total consumption from the individual components
        df["Ce"] = df['Jmed_b'] + df['Jmed_c'] + df['Jmed_d'] + df['Jmed_s_f'] * 1e-3  # kW

    return df

def estimate_states(df):

    df["sf_active"] = df["qsf"] > 0.1
    df["med_active"] = df["qmed_f"] > 0.5

    return df

def calculate_aux_variables(df, cost_w, cost_e, sample_rate_numeric):

    # df = calculate_costs(df, cost_w, cost_e, sample_rate_numeric)
    df = calculate_benefits(df, cost_w, cost_e, sample_rate_numeric)
    df = estimate_states(df)

    return df

def calculate_powers_solar_field(row, max_power: float = 250, min_power=0, calculate_per_loop: bool = True):
    try:
        # Solar field
        w_p = w_props(P=0.16, T=(row["Tsf_in"] + row["Tsf_out"]) / 2 + 273.15)  # MPa, K
        row["Psf"] = row["qsf"] / 3600 * w_p.rho * w_p.cp * (row["Tsf_out"] - row["Tsf_in"])  # kW
        row["Psf"] = min(row["Psf"], max_power)
        row["Psf"] = max(row["Psf"], min_power)
        row["Phx_p"] = row["Psf"]

        # Solar field loops
        if calculate_per_loop:
            for loop_str in ['l2', 'l3', 'l4', 'l5']:
                row[f"Psf_{loop_str}"] = row[f"qsf_{loop_str}"] / 3600 * w_p.rho * w_p.cp * (
                            row[f"Tsf_out_{loop_str}"] - row[f"Tsf_in_{loop_str}"])  # kW
                row[f"Psf_{loop_str}"] = min(row[f"Psf_{loop_str}"], max_power)
                row[f"Psf_{loop_str}"] = max(row[f"Psf_{loop_str}"], min_power)

    except Exception as e:
        logger.error(f'Error: {e}')
        # row["Psf"] = np.nan

    return row


def calculate_powers_thermal_storage(row, max_power: float = 250, min_power=0):
    try:
        # Thermal storage input power
        w_p = w_props(P=0.16, T=(row["Thx_s_out"] + row["Thx_s_in"]) / 2 + 273.15)  # MPa, K
        row["Pts_src"] = row["qts_src"] / 3600 * w_p.rho * w_p.cp * (row["Thx_s_out"] - row["Thx_s_in"])  # kW
        row["Pts_src"] = min(row["Pts_src"], max_power)
        row["Pts_src"] = max(row["Pts_src"], min_power)
        row["Phx_s"] = row["Pts_src"]

        # Thermal storage output power
        w_p = w_props(P=0.16, T=(row["Tts_h_out"] + row["Tts_c_in"]) / 2 + 273.15)  # MPa, K
        row["Pts_dis"] = row["qts_dis"] / 3600 * w_p.rho * w_p.cp * (row["Tts_h_out"] - row["Tts_c_in"])  # kW
        row["Pts_dis"] = min(row["Pts_dis"], max_power)
        row["Pts_dis"] = max(row["Pts_dis"], min_power)


    except Exception as e:
        logger.error(f'Error: {e}')
        # row["Pts_src"] = np.nan
        # row["Pts_dis"] = np.nan
        # row["Psf"] = np.nan

    return row