from pathlib import Path
import math
from enum import Enum
from dataclasses import fields, asdict
import numpy as np
from loguru import logger
import hjson
import datetime
import pygmo as pg
import pandas as pd

from solarmed_modeling.utils import data_preprocessing, data_conditioning
from solarmed_modeling.solar_med import SolarMED, InitialStates
from solarmed_modeling.fsms.med import MedFsm, FsmInputs as MedFsmInputs
from solarmed_modeling.fsms.sfts import SolarFieldWithThermalStorageFsm, FsmInputs as SfTsFsmInputs
from solarmed_modeling.fsms import MedState, SfTsState

from solarmed_optimization import (DecisionVariables, 
                                   DecisionVariablesUpdates,
                                   ProblemData,
                                   ProblemParameters,
                                   ProblemSamples,)
# MINLP
from solarmed_optimization.problems.nlp import Problem as NlpProblem
from solarmed_optimization.utils import (validate_dec_var_updates, 
                                         times_to_samples,
                                         decision_variables_to_decision_vector,
                                         decision_vector_to_decision_variables,
                                         validate_real_dec_vars)
# nNLP
from solarmed_optimization.problems.minlp import BaseProblem, EnvironmentVariables, Problem as MinlpProblem
from solarmed_optimization.utils.operation_plan import OperationPlanner

logger.disable("phd_visualizations")

# @dataclass
# class ProblemInitializationVars:
#     problem_params: ProblemParameters
#     samples_data: SampleParams
#     model: SolarMED

        
renames_dict: dict[str, str] = {
    # First rename the original columns
    "Tmed_c_in": "Tmed_c_in_orig", "wmed_f": "wmed_f_orig",

    # Then rename the new columns
    "DNI_Wm2": "DNI", "DHI_Wm2": "DHI",
    "so": "wmed_f", "thetao": "Tmed_c_in"
}

def problem_initialization(problem_params: ProblemParameters, date_str: str, data_path: Path = Path("./data"),
                           set_dec_var_updates: bool = True) -> ProblemData:
    
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
    if set_dec_var_updates:
        if pp.dec_var_updates is None:
            default_fields = {field.name: ps.default_n_dec_var_updates for field in fields(DecisionVariablesUpdates)}
            pp.dec_var_updates = DecisionVariablesUpdates(**default_fields)
            pp.dec_var_updates.qsf = math.floor(pp.optim_window_time / ((pp.sample_time_mod+pp.sample_time_opt)/2)) # Midpoint between sample_time_mod and sample_time_opt updates
            # New, simplify MINLP by limiting the updates of operation mode variables independent of the horizon
            # TODO: Ideally we would limit the number of updates per horizon duration (e.g. 3 updates every 18 hours)
            # pp.dec_var_updates.med_active = 6
            # pp.dec_var_updates.med_vac_state = 6
            # pp.dec_var_updates.sf_active = 6
            # pp.dec_var_updates.ts_active = 6

        validate_dec_var_updates(dec_var_updates=pp.dec_var_updates, optim_window_time=pp.optim_window_time, sample_time_mod=pp.sample_time_mod)

    Tts_h = pp.initial_states.Tts_h if pp.initial_states is not None else [df['Tts_h_t'].iloc[pp.idx_start], df['Tts_h_m'].iloc[pp.idx_start], df['Tts_h_b'].iloc[pp.idx_start]]
    Tts_c = pp.initial_states.Tts_c if pp.initial_states is not None else [df['Tts_c_t'].iloc[pp.idx_start], df['Tts_c_m'].iloc[pp.idx_start], df['Tts_c_b'].iloc[pp.idx_start]]
    fsm_internal_states = pp.initial_states.fsms_internal_states if pp.initial_states is not None else pp.fsm_internal_states
    Tsf_in_ant = pp.initial_states.Tsf_in_ant if pp.initial_states is not None else df['Tsf_in'].iloc[pp.idx_start-ps.span:pp.idx_start].values
    qsf_ant = pp.initial_states.qsf_ant if pp.initial_states is not None else df['qsf'].iloc[pp.idx_start-ps.span:pp.idx_start].values
    
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
        on_limits_violation_policy=pp.on_limits_violation_policy,
        
        # Initial states
        ## FSMs
        fsms_internal_states=fsm_internal_states,
        ## Thermal storage
        Tts_h=Tts_h, 
        Tts_c=Tts_c,
        ## Solar field
        Tsf_in_ant=np.array(Tsf_in_ant),
        qsf_ant=np.array(qsf_ant),
    )
    
    return ProblemData(df=df, 
                       problem_params=problem_params, 
                       problem_samples=problem_samples, 
                       model=model)
    

def initialize_problem_instance_nNLP(problem_data: ProblemData,
                                     # Integrated in ProblemData.ProblemParameters
                                     #operation_actions: dict[str, list[tuple[str, int]]] = None,
                                     #real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod = RealDecisionVariablesUpdatePeriod(),
                                     #initial_dec_vars_values: InitialDecVarsValues = InitialDecVarsValues(),
                                     idx_mod: int = 0,
                                     store_x=False,
                                     store_fitness=False,
                                    ) -> list[NlpProblem]:
    
    # if operation_actions is None:
    #     operation_actions: dict = {
    #         # Day 1 -----------------------  # Day 2 -----------------------
    #         "sfts": [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)],
    #         "med": [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)],
    #     }
    
    ps: ProblemSamples = problem_data.problem_samples
    pp: ProblemParameters = problem_data.problem_params
    model = problem_data.model
    df = problem_data.df
    
    assert getattr(pp, 'operation_actions', None) is not None, "Operation actions must be defined in the ProblemParameters"
    assert getattr(pp, 'real_dec_vars_update_period', None) is not None, "Real decision variables update period must be defined in the ProblemParameters"
    assert getattr(pp, 'initial_dec_vars_values', None) is not None, "Initial decision variables values must be defined in the ProblemParameters"
    
    hor_span = (idx_mod + 1, idx_mod + 1 + ps.n_evals_mod_in_hor_window)
    ds = df.iloc[hor_span[0] : hor_span[1]]

    env_vars: EnvironmentVariables = EnvironmentVariables(
        I=ds["I"],
        Tamb=ds["Tamb"],
        Tmed_c_in=ds["Tmed_c_in"],
        cost_w=pd.Series(
            data=np.ones((ps.n_evals_mod_in_hor_window,)) * pp.env_params.cost_w,
            index=ds.index,
        ),
        cost_e=pd.Series(
            data=np.ones((ps.n_evals_mod_in_hor_window,)) * pp.env_params.cost_e,
            index=ds.index,
        ),
    )
    # For operation plan, environment variables are only available with a one hour resolution
    env_vars_opt = env_vars.resample(f"{pp.sample_time_opt}s", origin="start")

    print(f"{env_vars.I.index[0]=}, {env_vars.I.index[-1]=}, {env_vars.I.index.freq=}")

    # 3. Build operation plan
    operation_planner = OperationPlanner.initialize(pp.operation_actions)
    print(operation_planner)

    I = [
        env_vars_opt.I.loc[
            df.index[0] + pd.Timedelta(days=n_day) : df.index[0]
            + pd.Timedelta(days=n_day + 1)
        ]
        .resample(f"{10}min", origin="start")
        .interpolate()
        for n_day in range(pp.optim_window_days)
    ]
    int_dec_vars_list = operation_planner.generate_decision_series(I)

    # 4. Initialize problem instances
    return [
        NlpProblem(
            int_dec_vars=int_dec_vars,
            # Planner layer should get time imprecise weather forecasts
            env_vars=env_vars_opt,
            real_dec_vars_update_period=pp.real_dec_vars_update_period,
            model=model,
            initial_dec_vars_values=pp.initial_dec_vars_values,
            sample_time_ts=pp.sample_time_ts,
            
            store_x=store_x,
            store_fitness=store_fitness,
        ) for int_dec_vars in int_dec_vars_list
    ]


def initialize_problem_instance_minlp(problem_data: ProblemData, idx_mod: int,
                       fsm_data_path: Path, return_env_vars: bool = False) -> MinlpProblem | tuple[MinlpProblem, EnvironmentVariables]:
    
    # alias definition    
    pp = problem_data.problem_params
    ps = problem_data.problem_samples
    
    hor_span = (idx_mod+1, idx_mod+1+ps.n_evals_mod_in_hor_window)
    ds = problem_data.df.iloc[hor_span[0]:hor_span[1]]
    
    env_vars: EnvironmentVariables = EnvironmentVariables(
        I=ds['I'].values,
        Tamb=ds['Tamb'].values,
        Tmed_c_in=ds['Tmed_c_in'].values,
        cost_w=np.ones((ps.n_evals_mod_in_hor_window, )) * pp.env_params.cost_w,
        cost_e=np.ones((ps.n_evals_mod_in_hor_window, )) * pp.env_params.cost_e,
    )

    ## Initialize problem
    problem = MinlpProblem(
        model=problem_data.model, 
        sample_time_opt=pp.sample_time_opt,
        optim_window_time=pp.optim_window_time,
        env_vars=env_vars,
        dec_var_updates=pp.dec_var_updates,
        fsm_valid_sequences=pp.fsm_valid_sequences,
        fsm_data_path=fsm_data_path,
        use_inequality_contraints=False
    )
    
    if not return_env_vars:
        return problem
    else:
        return problem, env_vars
    
def generate_integer_pop(model: SolarMED, pp: ProblemParameters, pop_size: int,
                         paths_from_state_df: pd.DataFrame) -> list[DecisionVariables]:
    """Function that generates a population of decision variables for the optimization problem.
    In particular, it sets the logical sequence of inputs for the MED and Solar Field with Thermal Storage FSMs in
    order to go from the initial state ('OFF' / 'IDLE') to the desired state ('ACTIVE').

    Args:
        model (SolarMED): _description_
        pp (ProblemParameters): _description_
        pop_size (int): _description_

    Returns:
        _type_: _description_
        
    Example:
        pop_size = 24
        shifted_active, shifted_vacuum = generate_shifted_sequences(model, pp, pop_size)
        # print(f"{shifted_active=}")
        print(f"{shifted_vacuum=}")
        Current result (first sequence does not repeat)
        shifted_vacuum=array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 2, 2, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 2, 2, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 2, 2, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 2, 2, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 2, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 2, 2, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 2, 2, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 2, 2, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 2, 2, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 2, 2, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 2, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 2, 2, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 2, 2, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 2, 2, 1, 1, 1, 1, 1, 1]
        ])
    """
    
    fsms = [
        # MedFsm(
        #     sample_time=pp.sample_time_opt,
        #     initial_state=model.med_state,
        #     params=model.fsms_params.med,
        #     internal_state=model.fsms_internal_states.med
        # ),
        SolarFieldWithThermalStorageFsm(
            sample_time=pp.sample_time_opt,
            initial_state=model.sf_ts_state,
            params=model.fsms_params.sf_ts,
            internal_state=model.fsms_internal_states.sf_ts
        )
    ]
    fsm_inputs_clss = [#MedFsmInputs, 
                       SfTsFsmInputs]
    desired_states = [#MedState.ACTIVE, 
                      SfTsState.SF_HEATING_TS]
    
    # Initialize the FSM and base arrays
    input_pops = {}
    for fsm, fsm_inputs_cls, desired_state in zip(fsms, fsm_inputs_clss, desired_states):
        step_idx = 0
        n_updates = getattr(pp.dec_var_updates, fields(fsm_inputs_cls)[0].name)
        inputs = {fld.name: np.empty((n_updates, ), dtype=fld.type if not issubclass(fld.type, Enum) else int) 
                 for fld in fields(fsm_inputs_cls)}

        # Generate base input arrays
        
        # Use max() to find the member with the highest value
        input_values = {}
        for fld in fields(fsm_inputs_cls):
            # Either are Enums or booleans
            if issubclass(fld.type, Enum):
                input_values[fld.name] = max(fld.type, key=lambda e: e.value)
            else:
                input_values[fld.name] = True
                
        while fsm.state != desired_state and step_idx < n_updates:
            # fsm.step(fsm.states_inputs_set[ desired_state.name ])
            fsm.step(fsm_inputs_cls(**input_values))
            expected_inputs = fsm.get_inputs_for_current_state()

            for input_name, input_val in asdict(expected_inputs).items():
                input_val = input_val if not isinstance(input_val, Enum) else input_val.value
                inputs[input_name][step_idx] = input_val
            
            step_idx += 1

        if step_idx < n_updates:
            # Fill the rest of the arrays with the last value
            for input_name, input_val in asdict(expected_inputs).items():
                input_val = input_val if not isinstance(input_val, Enum) else input_val.value
                inputs[input_name][step_idx:] = input_val

        # Create shifted arrays with repetition
        inputs_pop = {input_name: np.zeros((pop_size, n_updates), dtype=type(input_value[0]) if not isinstance(type(input_value[0]), Enum) else int) 
                     for input_name, input_value in inputs.items()}

        for i in range(pop_size-1):
            shift = i % n_updates  # Cyclic index for repetition
            if shift == 0:
                # First row is the base
                for input_name, input_val in inputs.items():
                    inputs_pop[input_name][i+1] = input_val
            else:
                # Shift the base array to the right and set the first `shift` elements to 0
                for input_name, input_val in inputs.items():
                    inputs_pop[input_name][i+1] = np.roll(input_val, shift)
                    inputs_pop[input_name][i+1, :shift] = 0
        
        input_pops.update(**inputs_pop)
        
    # New approach for MED. Just use data from path explorer
    n_updates = pp.dec_var_updates.med_mode
    paths_from_state_df = paths_from_state_df.copy()
    # Remove first column and append a new one by duplicating the last one
    paths_from_state_df = paths_from_state_df.iloc[:, 1:]
    paths_from_state_df['N'] = paths_from_state_df.iloc[:, -1]
    
    difference = pop_size - len(paths_from_state_df)
    if difference > 0:
        # Repeat the first difference rows
        repeated_rows = paths_from_state_df.iloc[:difference]
        paths_from_state_df = pd.concat([paths_from_state_df, repeated_rows], ignore_index=True)
    
    # Map 0 and 4s to MedMode.OFF and 1,2,3,5 to MedMode.ACTIVE
    # and take the first n_updates rows
    input_pops["med_mode"] = paths_from_state_df.map(lambda x: 0 if x in [0, 4] else 1).iloc[:pop_size].to_numpy()

    # print(f"{input_pops=}")
    
    return [
        DecisionVariables(
            **{name: value[pop_idx] for name, value in input_pops.items()},
        
            qsf=np.nan * np.ones((pp.dec_var_updates.qsf, )),
            qts_src=np.nan * np.ones((pp.dec_var_updates.qts_src, )),
            qmed_s=np.nan * np.ones((pp.dec_var_updates.qmed_s, )),
            qmed_f=np.nan * np.ones((pp.dec_var_updates.qmed_f, )),
            Tmed_c_out=np.nan * np.ones((pp.dec_var_updates.Tmed_c_out, )),
            Tmed_s_in=np.nan * np.ones((pp.dec_var_updates.Tmed_s_in, )),
        ) 
        for pop_idx in range(pop_size)]
    
    
def generate_population(model: SolarMED, pp: ProblemParameters, 
                        problem: BaseProblem, pop_size: int,
                        paths_from_state_df: pd.DataFrame,
                        return_decision_vector: bool = False,
                        dec_vec: np.ndarray = None, 
                        prob: pg.problem = None) -> list[DecisionVariables] | np.ndarray:
    
    assert dec_vec is not None or prob is not None, "Either a starting population is provided (dec_vec), or a pg.problem instance is provided (prob)"
    
    # Generate integer population based on FSMs logic starting from current state
    pop_dec_vars = generate_integer_pop(model, pp, pop_size, paths_from_state_df)
    
    for i, dec_vars in enumerate(pop_dec_vars):
        
        # Either initialize random decision vectors
        # Or use the decision vector provided
        if dec_vec is None:
            dec_vec = pg.random_decision_vector(prob)

        dec_vars_ = decision_vector_to_decision_variables(dec_vec, dec_var_updates=problem.dec_var_updates, span='none', )
        
        # Set the real decision variables values in the manually generated decision variables
        [setattr(dec_vars, var_id, getattr(dec_vars_, var_id)) for var_id in problem.dec_var_real_ids]
        # Validate the real decision variables values using the manually established logical variables values
        dec_vars = validate_real_dec_vars(dec_vars, problem.real_dec_vars_box_bounds)
        
    if not return_decision_vector:
        return pop_dec_vars
    
    return np.array([decision_variables_to_decision_vector(dec_vars) for dec_vars in pop_dec_vars])
    