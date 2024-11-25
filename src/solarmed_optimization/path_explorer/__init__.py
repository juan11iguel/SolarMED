from dataclasses import asdict, fields
from loguru import logger
import inspect
from enum import Enum
import copy
from typing import Literal
from multiprocessing import Pool
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from solarmed_modeling.fsms import MedState, SfTsState, FsmParameters
from solarmed_modeling.fsms.med import MedFsm, FsmInputs as MedFsmInputs
from solarmed_modeling.fsms.sfts import (SolarFieldWithThermalStorageFsm,
                                         FsmInputs as SfTsFsmInputs)
from solarmed_optimization.utils import timer_decorator
from .utils import export_results, filter_paths

SupportedFSMTypes = MedFsm | SolarFieldWithThermalStorageFsm
SupportedStates = MedState | SfTsState

class FsmInputsMapping(Enum):
    MED = MedFsmInputs
    SFTS = SfTsFsmInputs
    
class FsmMapping(Enum):
    MED = MedFsm
    SFTS = SolarFieldWithThermalStorageFsm
    
class StateMapping(Enum):
    MED = MedState
    SFTS = SfTsState

# def generate_default_inputs(system):
# Define expected inputs to the step method, can't think of a cleaner way to only do it once and for every option
expected_inputs_default = {}
for system, default_value in zip(['MED', 'SFTS'], [0.0, 0.0]):
    # machine_cls = FsmMapping[system].value

    # args = list(inspect.signature(machine_cls.step).parameters.keys())
    # expected_inputs_default[system] = {arg: default_value for arg in args}
    # expected_inputs_default[system].pop('self')

    # return expected_inputs_default
    
    inputs_cls = FsmInputsMapping[system].value
    expected_inputs_default[system] = {field.name: default_value for field in fields(inputs_cls)}

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

        if isinstance(model, MedFsm):
            system = 'MED'
        elif isinstance(model, SolarFieldWithThermalStorageFsm):
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
                if not condition_obj.target:
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


def generate_all_paths(machines: list[SupportedFSMTypes], current_state: Enum, current_path: list[SupportedStates],
                       all_paths: list[list[SupportedStates]], max_step_idx: int, fsm_inputs_cls: object,
                       recursitron_cnt: int = 0, valid_inputs: list[list[list[float]]] = None) -> int | bool:
    """
    Recursive function that generates all possible paths for a given instance of the machine starting from its current state.
    It will explore every possible path from the current state up to a final state defined by the prediction horizon (`max_step_idx`). Then it will
    backtrack one step and explore the next possible path from the prior state up to the final state. This keeps going until all possible
    paths are explored.

    :param machines: instances of the machine with different states
    :param current_state: current state of the machine
    :param current_path: current path from the initial state to the current state
    :param all_paths: list of all possible paths, this is the actual "output" of the function
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
    # if current_valid_inputs is not None: current_valid_inputs.append(machines[-1].get_inputs(format='array'))

    logger.info(
        f"Hello, this is recursitron {recursitron_cnt:03d}, current state is: {current_state}, current step: {len(current_path) - 1} and we are {len(machines)} machines deep with samples {[machine.current_sample for machine in machines]}")
    recursitron_cnt += 1

    if len(current_path) - 1 != machines[-1].current_sample:
        raise RuntimeError(
            f"Current step_idx ({len(current_path) - 1}) does not match the sample for the current machine instance ({machines[-1].current_sample})")

    # If the current state is a final state, add the current path to all_paths
    if len(current_path) >= max_step_idx:
        all_paths.append(list(current_path))  # Make a copy of current_path
        if valid_inputs is not None:
            current_path_inputs = [np.nan_to_num(machines[machine_idx].get_inputs(format="array"), nan=0.0) for machine_idx in range(len(machines)-1)]
            
            valid_inputs.append( current_path_inputs )
        # if valid_inputs is not None: valid_inputs.append(list(current_valid_inputs))

        current_path.pop()
        # if current_valid_inputs is not None: current_valid_inputs.pop()
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
            machines[-1].step(fsm_inputs_cls(**expected_inputs))
        except Exception as e:
            logger.error("ay mi madre")
            raise e

        next_state = machines[-1].get_state()
        recursitron_cnt = generate_all_paths(machines=machines, current_state=next_state, 
                                             current_path=current_path, all_paths=all_paths, 
                                             max_step_idx=max_step_idx,
                                             fsm_inputs_cls=fsm_inputs_cls, 
                                             recursitron_cnt=recursitron_cnt,
                                             valid_inputs=valid_inputs)

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
    machine_cls, machine_init_args, max_step_idx, fsm_inputs_cls, fsm_params, initial_state, include_valid_inputs = args
    machines = [machine_cls(**machine_init_args, initial_state=initial_state, params=fsm_params)]
    
    all_paths = []
    valid_inputs = None
    if include_valid_inputs:
        valid_inputs = []
    generate_all_paths(machines=machines, current_state=initial_state, 
                       current_path=[], all_paths=all_paths, 
                       valid_inputs=valid_inputs,
                       max_step_idx=max_step_idx, 
                       fsm_inputs_cls=fsm_inputs_cls,
                       recursitron_cnt=0)
    logger.info(f"Completed paths search (Deep First Search - DFS) for initial state {initial_state}, in total {len(all_paths)} paths were found")

    if include_valid_inputs:
        return all_paths, valid_inputs
    return all_paths

@timer_decorator
def get_all_paths(system: Literal['MED', 'SFTS'], machine_init_args: dict, max_step_idx: int,
                  fsm_params: FsmParameters, initial_states: list[SupportedStates] = None, 
                  use_parallel: bool = False, filter_duplicates: bool = True, 
                  save_results: bool = False, output_path: Path = None,
                  id: str = None, include_valid_inputs: bool = False,
                  valid_sequences: list[list[int]] = None,  
                  ) -> list[list[str]] | tuple[list[list[str]], list[list[list[float]]]]:
    
    """Function that generates all possible paths for a given system and its parameters. 
    The function will explore every possible path from the `initial_states` list up to a 
    final state defined by the prediction horizon (`max_step_idx`).
    - If `use_parallel` is enabled, the function will use the multiprocessing module to
    parallelize the search for paths. 
    - The results can be saved to a file if `save_results` is enabled.
    - If `include_valid_inputs` is enabled, the function will also return the valid inputs that allow the machine to 
    transition to the next state.
    - If an `id` is provided, the results will be saved to a file with including an `alternative_id` field.

    Args:
        system (Literal["MED", "SFTS"]): System to evaluate
        machine_init_args (dict): FSM class initialization arguments, usually just the `sample_time`
        max_step_idx (int): Prediction horizon
        fsm_params (FsmParameters): A dataclass containing the parameters for the FSM
        initial_states (list[SupportedStates], optional): List of initial states to start the path exploration from. 
        Defaults to automatically extracting all possible states for the machine.
        use_parallel (bool, optional): Whether to use parallel processing for path exploration. Defaults to False.
        filter_duplicates (bool, optional): Whether to filter out duplicate paths. Defaults to True.
        save_results (bool, optional): Whether to save the results to a file. Defaults to False.
        output_path (Path, optional): Path to save the results if `save_results` is enabled. Defaults to None.
        id (str, optional): Identifier to include in the saved results file. Defaults to None.
        include_valid_inputs (bool, optional): Whether to include valid inputs in the results. Defaults to False.

    Raises:
        NotImplementedError: If the system is not supported.

    Returns:
        list[list[str]] | tuple[list[list[str]], list[list[list[float]]]]: List of paths or tuple of paths and valid inputs.
    """

    # Validation
    if save_results:
        assert output_path is not None, "If `save_results` is enabled, a valid output_path needs to be provided"
    valid_inputs: list[list[list[float]]] = None

    start_time = time.time()
    
    assert system in ['MED', 'SFTS'], f"Unsupported system {system}"
    
    fsm_inputs_cls = FsmInputsMapping[system].value
    machine_cls = FsmMapping[system].value
    state_cls = StateMapping[system].value

    # Define the possible initial states
    if not initial_states:
        initial_states = [state for state in state_cls]

    if not use_parallel:
        all_paths = []
        valid_inputs = None
        if include_valid_inputs:
            valid_inputs = []
            
        for initial_state in initial_states:

            # The machines are re-initialized for every initial state
            machines = [machine_cls(**machine_init_args, params=fsm_params, initial_state=initial_state)]

            generate_all_paths(machines=machines, current_state=initial_state, 
                               current_path=[], all_paths=all_paths, 
                               valid_inputs=valid_inputs, 
                               max_step_idx=max_step_idx, 
                               fsm_inputs_cls=fsm_inputs_cls,
                               recursitron_cnt=0)
            logger.info(f"Completed paths search (Deep first search?) for initial state {initial_state}, in total {len(all_paths)} paths were found")

    else:
        with Pool() as p:
            output = p.map(
                worker,
                [(machine_cls, machine_init_args, max_step_idx, fsm_inputs_cls, fsm_params, initial_state, include_valid_inputs) for initial_state in initial_states]
            )
            
            if include_valid_inputs:
                all_paths, valid_inputs = zip(*output)
                # Flatten the list of lists
                valid_inputs = [inputs for inputs_list in valid_inputs for inputs in inputs_list]
            else:
                all_paths = output

            # Flatten the list of lists
            all_paths = [path for paths in all_paths for path in paths]

    # if filter_duplicates:
    #     # Filter out duplicates using sets
    #     original_size = len(all_paths)
    #     set_of_tuples = set(tuple(x) for x in all_paths)
    #     all_paths = [list(x) for x in set_of_tuples]
        
    if filter_duplicates:
        
        original_size = len(all_paths)
        if not include_valid_inputs:
            # Filter out duplicates using sets
            set_of_tuples = set(tuple(x) for x in all_paths)
            all_paths = [list(x) for x in set_of_tuples]
        
        else:
            # Step 1: Use a set for unique paths and track indices to keep
            unique_paths = set()
            indices_to_keep = []

            for idx, path in enumerate(all_paths):
                path_tuple = tuple(path)
                if path_tuple not in unique_paths:
                    unique_paths.add(path_tuple)
                    indices_to_keep.append(idx)  # Track index of unique item

            # Step 2: Filter all_paths and valid_inputs based on indices to keep
            all_paths = [all_paths[i] for i in indices_to_keep]
            valid_inputs = [valid_inputs[i] for i in indices_to_keep]
            
        logger.info(f"Removed {original_size - len(all_paths)} duplicate paths from the results")
        
    # Filter out invalid sequences
    if valid_sequences is not None:
        original_size = len(all_paths)
        for valid_sequence in valid_sequences:
            original_size2 = len(all_paths)
            all_paths, valid_inputs = filter_paths(paths=all_paths, valid_sequence=valid_sequence, aux_list=valid_inputs)
            
            logger.info(f"Removed {original_size2 - len(all_paths)} paths not matching the valid sequence {valid_sequence}")
        logger.info(f"Removed a total of {original_size - len(all_paths)} paths not matching all valid sequences")

    logger.info(f"Total number of paths found using a Deep First Search algorithm: {len(all_paths)}")

    computation_time: float = time.time() - start_time

    if save_results:
        export_results(
            paths=all_paths, output_path=output_path, system=system, 
            params={"valid_sequences": valid_sequences, **machine_init_args, **asdict(fsm_params)}, 
            computation_time=computation_time, id=id,
            valid_inputs=valid_inputs
        )

    if include_valid_inputs:
        return all_paths, valid_inputs
    
    return all_paths


if __name__ == '__main__':

    max_step_idx = 12
    machine_init_args = dict(sample_time=1)
    systems_to_evaluate: list[str] = ['SFTS', 'MED']
    
    for system in systems_to_evaluate:

        logger.info(f"Evaluating possible paths for system: {system}")
        
        if system == 'MED':
            machine_init_args.update(
                vacuum_duration_time=3,
                brine_emptying_time=1,
                startup_duration_time=1
            )

        start_time = time.time()
        all_paths = get_all_paths(
            system=system,
            machine_init_args=machine_init_args,
            max_step_idx=max_step_idx,
            # initial_states=[MedState.IDLE, MedState.OFF],
            use_parallel=True
        )

        dump_as(all_paths, Path(f'results/all_paths_{system}'), file_format='json')
        
        logger.info(f"Finished evaluation of system {system}, took {time.time()-start_time:.2f} seconds")