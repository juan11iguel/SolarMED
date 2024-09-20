from loguru import logger
import inspect
from enum import Enum
import copy
from typing import Literal
from multiprocessing import Pool
from pathlib import Path
import json

import numpy as np
import pandas as pd

from solarMED_modeling import MedState, SF_TS_State
from solarMED_modeling.fsms import SupportedFSMTypes, MedFSM, SolarFieldWithThermalStorage_FSM

from solarMED_optimization.utils import timer_decorator

FSMPathType = list[Enum]

# def generate_default_inputs(system):
# Define expected inputs to the step method, can't think of a cleaner way to only do it once and for every option
expected_inputs_default = {}
for system, default_value in zip(['MED', 'SFTS'], [None, 0.0]):
    if system == 'MED':
        machine_cls = MedFSM
    elif system == 'SFTS':
        machine_cls = SolarFieldWithThermalStorage_FSM
    else:
        raise NotImplementedError(f'Unsupported system {system}')

    args = list(inspect.signature(machine_cls.step).parameters.keys())
    expected_inputs_default[system] = {arg: default_value for arg in args}
    expected_inputs_default[system].pop('self')

    # return expected_inputs_default

# expected_inputs_default = generate_default_inputs()


def get_transitions_with_inputs(model: SupportedFSMTypes, prior_inputs: list | np.ndarray) -> list[tuple[str, dict]]:
    """
    Get all possible transitions from the current state together with the inputs that need to be satisfied

    WARNING: TLDR: Depending on the order of the condition definitions, the expected inputs might be different.

    Explanation: When multiple conditions are defined for a transition, the expected inputs are obtained sequentially
    and overwritten if the same key is defined in multiple conditions. So the definition order of the conditions is
    important!

    :param model: Finite State Machine model instance
    :param prior_inputs: Inputs from the prior step, used to add as a transition staying in the same state
    :return:
    """

    outputs = []
    for trigger in model.machine.get_triggers(model.get_state()):

        if isinstance(model, MedFSM):
            system = 'MED'
        elif isinstance(model, SolarFieldWithThermalStorage_FSM):
            system = 'SFTS'
        else:
            raise NotImplementedError(f'Unsupported system {type(model).__name__}')

        expected_inputs = copy.deepcopy(expected_inputs_default[system])

        # First the inputs need to be retrieved
        condition_objs = model.machine.get_transitions(trigger=trigger)[0].conditions
        for condition_obj in condition_objs:
            condition = getattr(model, condition_obj.func)

            # print(f"{trigger} | {condition_obj.func}: {'invalid' if condition_obj.target == False else 'valid'}")

            try:
                if condition_obj.target == False:
                    inputs = condition(return_invalid_inputs=True)
                else:
                    inputs = condition(return_valid_inputs=True)
            except TypeError:
                # logger.debug(f"Condition {condition_obj.func} does not implement return_invalid_inputs or return_valid_inputs logic, this is interpreted as condition not dependant on inputs and is skipped")
                pass
            else:
                # Update expected_inputs with new obtained values, how to ensure the insertion order is correct?
                # For example, when going from shutting down to OFF, inputs need to be invalid, and vacuum OFF but not LOW
                # When going from shutting down to IDLE, inputs need to be invalid, and vacuum LOW not OFF
                # Depending on the order of the condition definitions, the resulting expected inputs might be different
                for key, new_value in inputs.items():
                    expected_inputs[key] = new_value

        outputs.append((trigger, expected_inputs))

    # Add the transition to stay in the same state
    outputs.append(('none', prior_inputs))

    return outputs


def generate_all_paths(machines: list[SupportedFSMTypes], current_state: Enum, current_path: FSMPathType,
                       all_paths: list[FSMPathType], max_step_idx: int, recursitron_cnt: int = 0) -> int:
    """
    Recursive function that generates all possible paths for a given instance of the machine starting from its current state.
    It will explore every possible path from the current state up to a final state defined by the prediction horizon (`max_step_idx`). Then it will
    backtrack one step and explore the next possible path from the prior state up to the final state. This keeps going until all possible
    paths are explored.

    :param machines: instances of the machine with different states
    :param current_state: current state of the machine
    :param current_path: current path from the initial state to the current state
    :param all_paths: list of all possible paths
    :param max_step_idx: prediction horizon
    :param current_step_idx: current step index
    :return:
    """

    if len(current_path) == 0 and recursitron_cnt > max_step_idx:
        # Exit

        logger.info(f"Path exploration completed for initial state {current_state}")
        return True

    current_path.append(current_state)
    machines.append(copy.deepcopy(machines[-1]))  # Create a copy from the machine before starting a new path

    logger.info(
        f"Hello, this is recursitron {recursitron_cnt:03d}, current state is: {current_state}, current step: {len(current_path) - 1} and we are {len(machines)} machines deep with samples {[machine.current_sample for machine in machines]}")
    recursitron_cnt += 1

    if len(current_path) - 1 != machines[-1].current_sample:
        raise RuntimeError(
            f"Current step_idx ({len(current_path) - 1}) does not match the sample for the current machine instance ({machines[-1].current_sample})")

    # If the current state is a final state, add the current path to all_paths
    if len(current_path) >= max_step_idx:
        all_paths.append(list(current_path))  # Make a copy of current_path

        current_path.pop()
        machine_sample = machines[-1].current_sample
        machines.pop()  # Remove the current machine from the list of machines, to effectively go back one step
        machines[-1] = copy.deepcopy(machines[-2])  # Make sure we start the next path from the machine

        logger.info(
            f"Path completed ({len(all_paths)}): {all_paths[-1]}. Backtracking machine, sample {machine_sample} -> {machines[-1].current_sample} ({machines[-1].get_state()})")

        return recursitron_cnt

    for transition, expected_inputs in get_transitions_with_inputs(machines[-1],
                                                                   prior_inputs=machines[-1].get_inputs(format='dict')):
        logger.info(f"Transition: {transition}")  # , expected inputs: {expected_inputs}")
        try:
            machines[-1].step(**expected_inputs)
        except Exception as e:
            logger.error("ay mi madre")
            raise e

        next_state = machines[-1].get_state()
        recursitron_cnt = generate_all_paths(machines, next_state, current_path, all_paths, max_step_idx,
                                             recursitron_cnt)

    if recursitron_cnt == True:
        return True

    # Remove the current state from the current path after exploring all paths from it
    machine_sample = machines[-1].current_sample
    machines.pop()  # Remove the current machine from the list of machines, to effectively go back one step
    machines[-1] = copy.deepcopy(machines[-2]) if len(machines) > 1 else copy.deepcopy(
        machines[-1])  # Make sure we start the next path from the machine
    logger.info(
        f'All paths explored starting from {current_path}. Backtracking machine instance from {machine_sample} to {machines[-1].current_sample} ({machines[-1].get_state()})')

    if len(current_path) - 1 > machines[-1].current_sample:
        # Remove the current state from the current path after exploring all paths from it, unless it's succeeding a final state
        current_path.pop()

    return recursitron_cnt


def worker(args):
    machine_cls, machine_init_args, max_step_idx, initial_state = args
    machines = [machine_cls(**machine_init_args, initial_state=initial_state)]
    all_paths = []
    generate_all_paths(machines, initial_state, [], all_paths, max_step_idx, 0)
    logger.info(f"Completed paths search (Deep First Search - DFS) for initial state {initial_state}, in total {len(all_paths)} paths were found")

    return all_paths

@timer_decorator
def get_all_paths(system: Literal['MED', 'SFTS'], machine_init_args: dict, max_step_idx: int,
                  initial_states: list[Enum] = None, use_parallel: bool = False,
                  filter_duplicates: bool = True) -> list[list[str]]:

    if system == 'MED':
        machine_cls = MedFSM
        state_cls = MedState

    elif system == 'SFTS':
        machine_cls = SolarFieldWithThermalStorage_FSM
        state_cls = SF_TS_State

    else:
        raise NotImplementedError(f'Unsupported system {system}')

    # Define the possible initial states
    if not initial_states:
        initial_states = [state for state in state_cls]
        # initial_states = [MedState.IDLE]

    if not use_parallel:
        all_paths = []
        for initial_state in initial_states:

            # The machines are re-initialized for every initial state
            machines = [machine_cls(**machine_init_args, initial_state=initial_state)]

            generate_all_paths(machines, initial_state, [], all_paths, max_step_idx, 0)
            logger.info(f"Completed paths search (Deep first search?) for initial state {initial_state}, in total {len(all_paths)} paths were found")

    else:
        with Pool() as p:
            all_paths = p.map(
                worker,
                [(machine_cls, machine_init_args, max_step_idx, initial_state) for initial_state in initial_states]
            )

            # Flatten the list of lists
            all_paths = [path for paths in all_paths for path in paths]

    if filter_duplicates:
        # Filter out duplicates using sets
        original_size = len(all_paths)
        set_of_tuples = set(tuple(x) for x in all_paths)
        all_paths = [list(x) for x in set_of_tuples]

        logger.info(f"Removed {original_size - len(all_paths)} duplicate paths from the results")

    logger.info(f"Total number of paths found using a Deep First Search algorithm: {len(all_paths)}")

    return all_paths


# def get_paths_from_state_to_dst_state(machine: SupportedFSMTypes, dst_state: Enum | str, dst_step_idx: int) -> list[
#     str]:
#     """
#     WIP
#
#     Function that returns all possible paths for a given instance of the machine starting from its current state,
#     and ends in the destination state at the destination step index.
#
#     :param machine: Instance of the machine
#     :param dst_state: Destination state
#     :param dst_step_idx: Destination step index
#     :return: list of paths
#     """
#     machines = [copy.deepcopy(machine)]
#
#     all_paths = []
#     generate_all_paths(machines, machine.get_state(), [], all_paths, dst_step_idx)
#
#     # Filter out the paths that do not end in the destination state
#     paths = [path for path in all_paths if path[-1] == dst_state]
#
#     return paths

def dump_as(paths: list[list[str]], file_path: Path, file_format: Literal['csv', 'json'] = 'csv'):
    """
    Save the paths to a CSV file or a JSON

    :param paths: list of paths
    :param file_path: path to the file
    :return:
    """

    # Convert the state type elements to string using the name attribute
    paths = [[state.name for state in path] for path in paths] # Terrible

    if file_format == 'json':
        with open(file_path.with_suffix('.json'), 'w') as f:
            json.dump(paths, f, indent=4)

    else:
        df = pd.DataFrame(paths)
        df.to_csv(file_path.with_suffix('csv'), index=False)

    logger.info(f"Paths saved to {file_path}")


if __name__ == '__main__':

    max_step_idx = 12
    system: Literal['MED', 'SFTS'] = 'SFTS'
    machine_init_args = dict(sample_time=1)

    if system == 'MED':
        machine_init_args.update(
            vacuum_duration_time=3,
            brine_emptying_time=1,
            startup_duration_time=1
        )

    all_paths = get_all_paths(
        system=system,
        machine_init_args=machine_init_args,
        max_step_idx=max_step_idx,
        # initial_states=[MedState.IDLE, MedState.OFF],
        use_parallel=True
    )

    dump_as(all_paths, Path(f'results/all_paths_{system}'), file_format='json')