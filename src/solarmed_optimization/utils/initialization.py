from pathlib import Path
import math
from dataclasses import fields
from loguru import logger
import hjson
import datetime

from solarmed_modeling.utils import data_preprocessing, data_conditioning
from solarmed_modeling.solar_med import SolarMED

from solarmed_optimization import (DecisionVariablesUpdates,
                                   ProblemData,
                                   ProblemParameters,
                                   ProblemSamples)
from solarmed_optimization.utils import validate_dec_var_updates, times_to_samples

# @dataclass
# class ProblemInitializationVars:
#     problem_params: ProblemParameters
#     samples_data: SampleParams
#     model: SolarMED

from dataclasses import dataclass, field
from solarmed_modeling.solar_med import FsmInternalState
from solarmed_modeling.fsms import MedState, SfTsState
import numpy as np

# TODO: Do not use this dataclass but the one from solarmed_modeling.solar_med
@dataclass
class InitialStates:
    ## Thermal storage
    Tts_h: np.ndarray[float]
    Tts_c: np.ndarray[float]
    ## Finite state machines
    fsms_internal_states: FsmInternalState = field(default_factory=lambda: FsmInternalState())
    med_state: MedState = MedState.OFF
    sf_ts_state: SfTsState = SfTsState.IDLE
    ## Solar field
    Tsf_in_ant: np.ndarray[float] = field(default_factory=lambda: np.array([0.], dtype=float))
    qsf_ant: np.ndarray[float] = field(default_factory=lambda: np.array([0.], dtype=float))
    
    def __post_init__(self):
        self.Tts_h = np.array(self.Tts_h, dtype=float)
        
renames_dict: dict[str, str] = {
    # First rename the original columns
    "Tmed_c_in": "Tmed_c_in_orig", "wmed_f": "wmed_f_orig",

    # Then rename the new columns
    "DNI_Wm2": "DNI", "DHI_Wm2": "DHI",
    "so": "wmed_f", "thetao": "Tmed_c_in"
}

def problem_initialization(problem_params: ProblemParameters, date_str: str, data_path: Path = Path("./data"),
                           initial_states: InitialStates = None) -> ProblemData:
    
    pp: ProblemParameters = problem_params

    # date_str: str = "20230703" # "20230707_20230710" # '20230630' '20230703'
    filenames: list[str] = [f'{date_str}_solarMED.csv', f'{date_str}_MED.csv', f'env_{date_str}.csv']    
    sample_time_mod_str: str = f'{pp.sample_time_mod}s'

    # Pre-processing
    data_paths = [data_path / f"datasets/{fn}" for fn in filenames]
    # Filter out non-existing files
    data_paths = [data_path for data_path in data_paths if data_path.exists()]
    # logger.info(f"File {data_path} does not exist")

    # 20230707_20230710 data does not include solar irradiation
    # An alternative source from http://heliot.psa.es/meteo_psa_2022 is used for the solar irradiance
    if (data_path / f"datasets/{date_str}_env.csv").exists():
        data_paths.append(data_path / f"datasets/{date_str}_env.csv")
        

    # 20230707_20230710 does not include continuous seawater temperature and salinity
    # An alternative source from https://doi.org/10.25423/CMCC/MEDSEA_ANALYSISFORECAST_PHY_006_013_EAS8 is used for seawater temperature and salinity
    data_paths.append(data_path / "datasets/external_data/env_20220524_20240524.csv")

    with open( data_path / "variables_config.hjson") as f:
        vars_config = hjson.load(f)
        
    # Load data and preprocess data
    df = data_preprocessing(data_paths, vars_config, sample_rate_key=sample_time_mod_str)

    # Condition data
    df = data_conditioning(df, sample_rate_numeric=pp.sample_time_mod, vars_config=vars_config, )
    df = df.rename(columns=renames_dict)

    if 'GHI_Wm2' in df.columns:
        df = df.rename(columns={"I": "I_orig", 'GHI_Wm2': 'I'})
    if 'Tamb_degC' in df.columns:
        df = df.rename(columns={"Tamb": "Tamb_orig", "Tamb_degC": "Tamb"})
        
    # Accomodate external environment data
    if df['Tmed_c_in'].isna().all():
        # Assuming that the external environment data is in the last file (chapucilla)
        df_env = data_preprocessing(paths=[data_paths[-1]], vars_config=vars_config, sample_rate_key=sample_time_mod_str)
        df_env = df_env.rename(columns=renames_dict)

        if df.index.min() > df_env.index.max():
            # External environemnt data only starts after data ends
            # Take the most recent data from external that matches the month and days of data
            ext_year: int = df_env.index.max().year
        else:
            # External environemnt data ends before data starts
            ext_year: int = df_env.index.min().year

        logger.warning(f"External environment data ({df_env.index.min().strftime('%Y%m%d')} - {df_env.index.max().strftime('%Y%m%d')}) (seawater temperature and salinity) is not available for the data period ({df.index.min().strftime('%Y%m%d')} - {df.index.max().strftime('%Y%m%d')}). Using data for same month and days but from closest year {ext_year}.")
        date_span: tuple[datetime.datetime, datetime.datetime] = (df.index.min().replace(year=ext_year), df.index.max().replace(year=ext_year))
        df["Tmed_c_in"] = df_env["Tmed_c_in"].loc[date_span[0]:date_span[1]].values
        df["wmed_f"] = df_env["wmed_f"].loc[date_span[0]:date_span[1]].values
        df["latitude"] = df_env["latitude"].loc[date_span[0]:date_span[1]].values
        df["longitude"] = df_env["longitude"].loc[date_span[0]:date_span[1]].values

    problem_params.episode_duration = problem_params.episode_duration if problem_params.episode_duration is not None else len(df) * pp.sample_time_mod

    # Computed parameters
    problem_samples = times_to_samples(problem_params)
    ps: ProblemSamples = problem_samples
    pp.idx_start = ps.span if pp.idx_start is None else pp.idx_start

    # Problem definition
    # Initialize decision variables updates
    if pp.dec_var_updates is None:
        default_fields = {field.name: ps.default_n_dec_var_updates for field in fields(DecisionVariablesUpdates)}
        pp.dec_var_updates = DecisionVariablesUpdates(**default_fields)
        pp.dec_var_updates.qsf = math.floor(pp.optim_window_time / ((pp.sample_time_mod+pp.sample_time_opt)/2)) # Midpoint between sample_time_mod and sample_time_opt updates

    validate_dec_var_updates(dec_var_updates=pp.dec_var_updates, optim_window_time=pp.optim_window_time, sample_time_mod=pp.sample_time_mod)

    Tts_h = initial_states.Tts_h if initial_states is not None else [df['Tts_h_t'].iloc[pp.idx_start], df['Tts_h_m'].iloc[pp.idx_start], df['Tts_h_b'].iloc[pp.idx_start]]
    Tts_c = initial_states.Tts_c if initial_states is not None else [df['Tts_c_t'].iloc[pp.idx_start], df['Tts_c_m'].iloc[pp.idx_start], df['Tts_c_b'].iloc[pp.idx_start]]
    fsm_internal_states = initial_states.fsms_internal_states if initial_states is not None else pp.fsm_internal_states
    Tsf_in_ant = initial_states.Tsf_in_ant if initial_states is not None else df['Tsf_in'].iloc[pp.idx_start-ps.span:pp.idx_start].values
    qsf_ant = initial_states.qsf_ant if initial_states is not None else df['qsf'].iloc[pp.idx_start-ps.span:pp.idx_start].values
    
    # Initialize model instance
    model = SolarMED(
        use_models=True,
        use_finite_state_machine=True,
        resolution_mode='constant-water-props',
        sample_time=pp.sample_time_mod,
        env_params=pp.env_params,
        fixed_model_params=pp.fixed_model_params,
        model_params=pp.model_params,
        fsms_params=pp.fsm_params,
        
        # Initial states
        ## FSMs
        fsms_internal_states=fsm_internal_states,
        ## Thermal storage
        Tts_h=Tts_h, 
        Tts_c=Tts_c,
        ## Solar field
        Tsf_in_ant=Tsf_in_ant,
        qsf_ant=qsf_ant,
    )
    
    return ProblemData(df=df, 
                       problem_params=problem_params, 
                       problem_samples=problem_samples, 
                       model=model)
    