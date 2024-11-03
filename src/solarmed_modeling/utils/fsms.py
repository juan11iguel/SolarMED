from loguru import logger
from pathlib import Path
from typing import Literal
from enum import Enum
import pandas as pd
# import time

from solarmed_modeling.fsms import (MedState, SfTsState, MedVacuumState, 
                                    SupportedSystemsStatesType)
from solarmed_modeling.fsms.med import MedFsm
from solarmed_modeling.fsms.sfts import SolarFieldWithThermalStorageFsm
# from phd_visualizations import save_figure
# from solarmed_modeling.visualization.fsm.state_evolution import state_evolution_plot

from solarmed_modeling.solar_med import SolarMED
# from solarmed_modeling.fsms import SolarMED

valid_input: float = 1.0
invalid_input: float = 0.0

def convert_to_state(
    state: str | int, 
    state_cls: SupportedSystemsStatesType, 
    return_format: Literal["enum", "name", "value"] = "enum"
) -> Enum | str | int:
    
    if isinstance(state, str):
        output = getattr(state_cls, state) 
    elif isinstance(state, int):
        output = state_cls(state)
    else:
        raise ValueError(f"`state` should be either a str or an int, not {type(state)}")
    
    if return_format == "enum":
        return output
    elif return_format == "name":
        return output.name
    elif return_format == "value":
        return output.value
    else:
        raise ValueError(f"`return_format` should be one of enum, str or int. Not {return_format}")


SupportedStateTypes = MedState | SfTsState
SupportedFMSs = MedFsm | SolarFieldWithThermalStorageFsm | SolarMED
def test_state(expected_state: str | SupportedStateTypes, base_cls: SupportedFMSs = None, model: SupportedFMSs = None, current_state:SupportedStateTypes | str = None) -> None:

    """
    Function to test the current state of the FSMs. Need to provide the expected state and either the current state,
    or a model to extract the current state from, to compare.

    Args:
        expected_state:
        base_cls:
        model:
        current_state:

    Returns:

    """

    if isinstance(expected_state, str) or isinstance(current_state, str):
        if not base_cls:
            raise ValueError("base_cls must be provided when using strings as arguments for the states")

    expected_st = getattr(base_cls, expected_state) if isinstance(expected_state, str) else expected_state

    if model:
        current_state = model.state

    current_state = getattr(base_cls, current_state) if isinstance(current_state, str) else current_state


    assert current_state == expected_st, f"Expected state {expected_st}, got {current_state}"


# def generate_results(model: SolarMED, df: pd.DataFrame, iteration_idx: int, output_path: Path = None) -> pd.DataFrame:
#     # Add iteration to results dataframe
#     df = model.to_dataframe(df)

#     # Generate FSM graph(s)
#     if output_path is not None:
#         model._med_fsm.generate_graph(output_path=Path(f"{output_path}_{model._med_fsm.name}_iteration_{iteration_idx}.svg"))
#         model._sf_ts_fsm.generate_graph(output_path=Path(f"{output_path}_{model._sf_ts_fsm.name}_iteration_{iteration_idx}.svg"))

#         logger.info(f"FSM graphs saved in {output_path}_....svg")

#         fig = state_evolution_plot(df, iteration=iteration_idx)
#         # Remove last parent from output_path
#         output_path = output_path.parent
#         save_figure(fig, f'state_evolution_iteration_{iteration_idx}', output_path, formats=['svg'], height=400, width=400)

#     return df


def test_profile(model: SolarMED, attachments_path: Path = None, n_of_steps: int = 3, episode_id: str = "test", loops: int = 1) -> pd.DataFrame:
    """
    Test the SolarMED class by going through all the possible states of the individual FSMs.

    Notes:
        - For every step in the MED FSM, some steps are given in the SF-TS FSM

    :param model: SolarMED instance
    :param attachments_path: Path to save the FSM graphs, if None, the graphs are not saved
    :param n_of_steps: Number of steps to take in the SF-TS FSM
    :param episode_id: Identifier for the test
    :return DataFrame: with the results of the test
    """

    sf_ts_inputs = [
        # The order is important
        # (Tsf_out: float, mts_src: float)
        (invalid_input, invalid_input),
        (valid_input, invalid_input),
        (valid_input, valid_input),
        (valid_input, valid_input),
        (valid_input, valid_input),
        (valid_input, valid_input),
        (invalid_input, invalid_input),
        (valid_input, valid_input),
        (invalid_input, invalid_input)
    ]

    expected_sf_ts_states: list[SfTsState] = [
        # NOTE: Expected states need to be reachable with inputs from the same index, even if they take multiple iterations

        SfTsState.IDLE,
        SfTsState.HEATING_UP_SF,
        SfTsState.SF_HEATING_TS,
        SfTsState.SF_HEATING_TS,
        SfTsState.SF_HEATING_TS,
        SfTsState.SF_HEATING_TS,
        SfTsState.IDLE,
        SfTsState.HEATING_UP_SF,
        SfTsState.IDLE
    ]

    med_inputs = [
        # The order is important
        # (mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float, med_vacuum_state: MedVacuumState | int)
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.HIGH),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.OFF),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.HIGH),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.LOW),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.LOW),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.LOW),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.LOW),
        (invalid_input, valid_input, valid_input, valid_input, MedVacuumState.LOW),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.LOW),
        (valid_input, valid_input, valid_input, valid_input, MedVacuumState.OFF),
    ]

    expected_med_states: list[MedState] = [
        # NOTE: Expected states need to be reachable with inputs from the same index, even if they take multiple iterations
        MedState.GENERATING_VACUUM,
        MedState.OFF,
        MedState.GENERATING_VACUUM,
        MedState.IDLE,
        MedState.ACTIVE,
        MedState.ACTIVE,
        MedState.ACTIVE,
        MedState.IDLE,
        MedState.ACTIVE,
        MedState.OFF
    ]

    if len(sf_ts_inputs) != len(expected_sf_ts_states):
        raise ValueError("The number of inputs and expected states for the SF-TS FSM must be the same")
    if len(med_inputs) != len(expected_med_states):
        raise ValueError("The number of inputs and expected states for the MED FSM must be the same")

    # Initialize some variables
    df = pd.DataFrame()
    output_path = attachments_path / episode_id if attachments_path else None
    iteration_idx = 0

    # Repeat the test for the number of loops
    for _ in range(loops):
        # Execute the steps
        exit_flag = False
        step_cnt = 0
        while not exit_flag:
            try:
                sf_ts_idx = iteration_idx % len(sf_ts_inputs)

                logger.info(f"Evaluating step {step_cnt+1}/{len(expected_med_states)} in the MED cycle, {sf_ts_idx} in the SF-TS cycle, iteration {iteration_idx}")

                # The order is important, keep it synchronized with the inputs
                model.step(
                    Tsf_out=sf_ts_inputs[sf_ts_idx][0],
                    mts_src=sf_ts_inputs[sf_ts_idx][1],
                    mmed_s=med_inputs[step_cnt][0],
                    mmed_f=med_inputs[step_cnt][1],
                    Tmed_s_in=med_inputs[step_cnt][2],
                    Tmed_c_out=med_inputs[step_cnt][3],
                    med_vacuum_state=med_inputs[step_cnt][4],

                    Tmed_c_in=valid_input,
                    I=valid_input,
                    Tamb=valid_input,
                )

                # Save iteration results
                df = generate_results(model=model, df=df, iteration_idx=iteration_idx, output_path=output_path)

                # Validate states
                test_state(expected_state=expected_sf_ts_states[sf_ts_idx], current_state=model.sf_ts_state)
                test_state(expected_state=expected_med_states[step_cnt], current_state=model.med_state)

            except AssertionError:
                # Some of the steps take various iterations to complete
                logger.debug(f"Step {iteration_idx} needs to be repeated with the same inputs for the MED FSM, expected state: {expected_med_states[step_cnt]}, got {model.med_state}")
                iteration_idx += 1
            else:

                logger.debug(f"Step {step_cnt+1} completed successfully")

                iteration_idx += 1 # Index for SF-TS cycle increases every iteration
                step_cnt += 1 # Index for MED cycle increases only when the state change is completed

            if iteration_idx > 100:
                raise RuntimeError("Ow Jeez, those are too many iterations, check the inputs and expected states, they are provoking an infinite loop.")


            # Exit condition
            if step_cnt >= len(expected_med_states):
                exit_flag = True

    return df
