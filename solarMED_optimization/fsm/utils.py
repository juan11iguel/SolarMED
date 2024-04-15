from solarMED_optimization.fsm import MedFSM, SolarFieldWithThermalStorage_FSM, SolarMED
from solarMED_optimization import MedState, SF_TS_State, MedVacuumState
import time
from loguru import logger
from pathlib import Path
import pandas as pd

valid_input: float = 1.0
invalid_input: float = 0.0

def test_state(test_state: str, base_cls, model: SolarFieldWithThermalStorage_FSM | MedFSM) -> None:
    expected_st = getattr(base_cls, test_state)

    assert model.state == expected_st, f"Expected state {expected_st}, got {model.state}"


def generate_results(model: SolarMED, df: pd.DataFrame, iteration_idx: int, output_path: Path) -> pd.DataFrame:
    # Generate FSM graph(s)
    model.med_fsm.generate_graph(output_path=Path(f"{output_path}_{model.med_fsm.name}_iteration_{iteration_idx}.svg"))
    model.sf_ts_fsm.generate_graph(output_path=Path(f"{output_path}_{model.sf_ts_fsm.name}_iteration_{iteration_idx}.svg"))

    # Add iteration to results dataframe
    df = model.to_dataframe(df)

    return df

def test_profile(model: SolarMED, attachments_path: Path, n_of_steps: int = 3, episode_id: str = "test"):
    """
    Test the SolarMED class by going through all the possible states of the individual FSMs.

    Notes:
        - For every step in the MED FSM, three steps are given in the SF-TS FSM

    :param model_med:
    :param model_sf_ts:
    :return:
    """

    def advance_st_ts(step_idx: int):
        # Define each step as a function
        steps = [
            # Step 1.
            lambda: model_sf_ts.step(Tsf_out=0, qts_src=0) and test_state('IDLE', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 2.
            lambda: model_sf_ts.step(Tsf_out=1, qts_src=0) and test_state('HEATING_UP_SF', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 3.
            lambda: model_sf_ts.step(Tsf_out=1, qts_src=1) and test_state('SF_HEATING_TS', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 4.
            lambda: model_sf_ts.step(Tsf_out=1, qts_src=1) and test_state('SF_HEATING_TS', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 5.
            lambda: model_sf_ts.step(Tsf_out=1, qts_src=1) and test_state('SF_HEATING_TS', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 6.
            lambda: model_sf_ts.step(Tsf_out=1, qts_src=1) and test_state('SF_HEATING_TS', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 7.
            lambda: model_sf_ts.step(Tsf_out=0, qts_src=0) and test_state('IDLE', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 8.
            lambda: model_sf_ts.step(Tsf_out=1, qts_src=1) and test_state('HEATING_UP_SF', base_cls=SF_TS_State,
                                                                          model=model_sf_ts),

            # Step 9.
            lambda: model_sf_ts.step(Tsf_out=0, qts_src=0) and test_state('IDLE', base_cls=SF_TS_State,
                                                                          model=model_sf_ts)
        ]

        # Every time step_idx is greater than len(steps), the steps are repeated
        step_idx = step_idx % len(steps)

        logger.debug(f"Step {step_idx} in SF-TS FSM")

        # Execute the steps starting from step_idx for n_of_steps
        for step in steps[step_idx:step_idx + n_of_steps]:
            step()

        return step_idx + n_of_steps

    model_med = model.med_fsm
    model_sf_ts = model.sf_ts_fsm

    step_idx = 0
    iteration_idx = 0
    df = pd.DataFrame()
    output_path = attachments_path / episode_id

    # Define each step as a function
    steps = [
        # Step 1.
        lambda: (
            model_med.step(mmed_s=valid_input, mmed_f=valid_input, Tmed_s_in=valid_input, Tmed_c_out=valid_input,
                           med_vacuum_state=MedVacuumState.HIGH),
            test_state(test_state='GENERATING_VACUUM', base_cls=MedState, model=model_med),
        ),

        # Step 2.
        lambda: (
            model_med.step(mmed_s=valid_input, mmed_f=valid_input, Tmed_s_in=valid_input, Tmed_c_out=valid_input,
                           med_vacuum_state=MedVacuumState.OFF),
            test_state(test_state='OFF', base_cls=MedState, model=model_med),
        ),

        # Step 3.
        lambda: (
            model_med.step(mmed_s=valid_input, mmed_f=valid_input, Tmed_s_in=valid_input, Tmed_c_out=valid_input,
                           med_vacuum_state=MedVacuumState.LOW),
            test_state(test_state='IDLE', base_cls=MedState, model=model_med),
        ),

        # Step 4.
        lambda: (
            model_med.step(mmed_s=valid_input, mmed_f=valid_input, Tmed_s_in=valid_input, Tmed_c_out=1,
                           med_vacuum_state=MedVacuumState.HIGH),
            test_state(test_state='ACTIVE', base_cls=MedState, model=model_med),
        ),

        # Step 5.
        lambda: (
            model_med.step(mmed_s=invalid_input, mmed_f=valid_input, Tmed_s_in=valid_input, Tmed_c_out=valid_input,
                           med_vacuum_state=MedVacuumState.LOW),
            test_state(test_state='IDLE', base_cls=MedState, model=model_med),
        ),

        # Step 6.
        lambda: (
            model_med.step(mmed_s=valid_input, mmed_f=valid_input, Tmed_s_in=valid_input, Tmed_c_out=valid_input,
                           med_vacuum_state=MedVacuumState.LOW),
            test_state(test_state='ACTIVE', base_cls=MedState, model=model_med),
        ),

        # Step 7.
        lambda: (
            model_med.step(mmed_s=valid_input, mmed_f=valid_input, Tmed_s_in=valid_input, Tmed_c_out=valid_input,
                           med_vacuum_state=MedVacuumState.OFF),
            test_state(test_state='OFF', base_cls=MedState, model=model_med),
        )
    ]

    # Execute the steps
    exit_flag = False
    while not exit_flag:
        step = steps[iteration_idx]
        try:
            step()
        except AssertionError:
            # Some of the steps take various iterations to complete
            pass
        else:
            step_idx = advance_st_ts(step_idx=step_idx)
            df = generate_results(model=model, df=df, iteration_idx=iteration_idx, output_path=output_path)
            iteration_idx += 1

        if iteration_idx >= len(steps):
            exit_flag = True
