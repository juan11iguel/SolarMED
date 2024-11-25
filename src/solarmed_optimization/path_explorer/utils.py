from enum import Enum
import inspect
import json
from pathlib import Path
from typing import Literal
import uuid
from dataclasses import asdict, fields, is_dataclass
import pickle
from datetime import datetime, timezone

from loguru import logger
import numpy as np
import pandas as pd
from solarmed_modeling.fsms.utils import convert_to

from solarmed_modeling.fsms import MedState, SfTsState
from solarmed_modeling.fsms.med import (MedFsm, FsmInputs as MedFsmInputs,
                                        FsmParameters as MedFsmParams)
from solarmed_modeling.fsms.sfts import (SolarFieldWithThermalStorageFsm as SfTsFsm,
                                         FsmInputs as SfTsFsmInputs,
                                         FsmParameters as SfTsFsmParams)

class SupportedSubsystemsStatesMapping(Enum):
    MED = MedState
    SFTS = SfTsState
    
class SupportedFsmsMapping(Enum):
    MED = MedFsm
    SFTS = SfTsFsm
    
class FsmInputsMapping(Enum):
    MED = MedFsmInputs
    SFTS = SfTsFsmInputs
    
class FsmParamsMapping(Enum):
    MED = MedFsmParams
    SFTS = SfTsFsmParams

def export_results(paths: list[list[Enum]], output_path: Path, system: Literal['SFTS', 'MED'], 
                   params: dict[str, int], computation_time: float, id: str = None,
                   valid_inputs: list[list[list[float]]] = None) -> None:

    """
    Results export structure:
    - metadata.json
    - {system}_n_{n_horizon}_{uuid}.csv
    - ...
    - {system}_n_{n_horizon}_{uuid}.csv

    metadata.json
    {
        system_A: [
            {
                "n_horizon": XX,
                "computation_time": XXXX.XX # seconds
                "n_paths": XXX,
                "parameters": {...}
                "file_id": "system_A_n_{n_horizon}_{uuid}.csv",
                "n_paths_per_initial_state": {
                    STATE_ID: n_paths,
                }
            },
            ...
            {...}
        ],
        ...
        system_Z: [
            ...
        ]
    }

    For every export with unique parameters (n_horizon, fsm parameters), a new csv 
    is generated and identified with an unique uuid. In metatada.json the information
    of which information each csv contains is stored
    """
    
    n_horizon: int = len(paths[0])
    n_paths: int = len(paths)
    uuid_str: str = str(uuid.uuid4())
    paths_file_id: str = f"{system.lower()}_n_{n_horizon}_{uuid_str}.csv"
    valid_inputs_file_id: str = f"{system.lower()}_n_{n_horizon}_{uuid_str}.pkl"

    output_path_metadata = output_path / "metadata.json"

    # Convert path Enums to integers
    paths = [[state.value for state in path] for path in paths]
    paths_df = pd.DataFrame(paths, columns=[str(i) for i in range(len(paths[0]))])
    states_cls = SupportedSubsystemsStatesMapping[system].value
    # fsm_cls = SupportedFsmsMapping[system].value
    
    # Convert valid inputs to DataFrame
    # This is absurdly complex, we should just have inputs be another dataclass
    # method_signature = inspect.signature(fsm_cls.step)
    # input_ids: list[str] = [param.name for param in method_signature.parameters.values() if param.default == inspect.Parameter.empty and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and param.name != "self"]
    input_ids: list[str] = [field.name for field in fields(FsmInputsMapping[system].value)]
    
    # if valid_inputs is not None:
        # Flatten data and prepare for DataFrame
        # flattened_data = [[path_index, state_index, input_index, input_value] 
        #                   for path_index, path in enumerate(valid_inputs) 
        #                   for state_index, input_values in enumerate(path) for input_index, input_value in enumerate(input_values)]
        # # Convert to DataFrame
        # df = pd.DataFrame(flattened_data, columns=["path", "state", "input", "value"])

        # # Map input names to the appropriate index in each array
        # df["input"] = df["input"].map(lambda x: input_ids[x])

        # # Set MultiIndex
        # df.set_index(["state", "element", "input"], inplace=True)
        # df = df.unstack(level=-1)
        
        # valid_inputs_df = df

    # Count number of paths per initial state
    paths_per_initial_state = {
        states_cls(state_value).name: len(paths_from_state)
        for state_value, paths_from_state in paths_df.groupby("0").groups.items()
    }
    
    output_dict_new: dict = {
        "alternative_id": id,
        "n_horizon": n_horizon,
        "computation_time": computation_time,
        "n_paths": n_paths,
        "n_paths_per_initial_state": paths_per_initial_state,
        "parameters": params,
        "input_ids": input_ids,
        "paths_file_id": paths_file_id,
        "valid_inputs_file_id": valid_inputs_file_id,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

    if output_path_metadata.exists():
        with open(output_path_metadata, 'r') as f:
            output_dict = json.load(f)
            # logger.info(f"Found existing results in {output_path_metadata} for system {system}")
    else:
        output_dict = {}
        
    if system not in output_dict:
        output_dict[system] = []
        
    out = output_dict[system]

    existing_data_paths: list[str] | None = None
    # Check if there is any existing parameter that matches the provided params
    for item_idx, item in enumerate(out):
        if item["parameters"] == params and item["n_horizon"] == n_horizon:
            existing_data_paths = [str(item["paths_file_id"]), str(item["valid_inputs_file_id"])]
            out[item_idx] = output_dict_new
            
            logger.info(f"Found existing results for parameters: {params}. Replacing data with current one")
            break
    else:
        # If no match is found, append a new entry
        out.append(output_dict_new)
        
        logger.info(f"Added new entry for paths with parameters: {params}")

    # Save the updated output_dict back to the file
    with open(output_path_metadata, 'w') as f:
        json.dump(output_dict, f, indent=4)

    # Delete old data if exists
    if existing_data_paths is not None:
        # Check if the file exists, then delete it
        [(output_path / existing_data_path).unlink() for existing_data_path in existing_data_paths if (output_path / existing_data_path).exists()]
        logger.info(f"Existing data {existing_data_paths} have been deleted.")
        
    # Save data to csv
    paths_df.to_csv(output_path / paths_file_id)
    logger.info(f"Saved paths to {output_path / paths_file_id}")
    if valid_inputs is not None:
        with open(output_path / valid_inputs_file_id, 'wb') as f:
            pickle.dump(valid_inputs, f)
        logger.info(f"Saved paths to {output_path/valid_inputs_file_id}")


def import_results(paths_path: Path, 
                   system: Literal['SFTS', 'MED'], 
                   n_horizon: int, 
                   params: dict[str, int],
                   return_format: Literal["value", "name", "enum"] = "enum", 
                   return_metadata: bool = False,
                   generate_if_not_found: bool = False,
                  ) -> tuple[pd.DataFrame, list[list[list[float]]]] | tuple[pd.DataFrame, list[list[list[float]]], dict]:

    with open(paths_path / "metadata.json", 'r') as f:
        metadata_dict = json.load(f)
        
    if system not in metadata_dict.keys():
        raise ValueError(f"No results found for system {system} with horizon")
    
    if is_dataclass(params):
        params = asdict(params)

    # extra_keys: set = set(metadata_dict[system][0]["parameters"].keys()) - set(params.keys())
    # Check if there is any existing parameter that matches the provided params
    found_results: bool = False
    for item in metadata_dict[system]:
        # Remove any additional keys in item["parameters"] with respect to params
        # item_params = {key: item["parameters"][key] for key in set(params.keys())}
        if item["parameters"] == params and item["n_horizon"] == n_horizon:
            found_results = True
            break
    else:
        # If no match is found, generate results if generate_if_not_found
        if not generate_if_not_found:
            raise ValueError(f"No data found for {system}, horizon={n_horizon} and parameters: {params}")

        # Generate results
        logger.warning(f"Data for system {system} with n_horizon = {n_horizon} and parameters: {params} not found. Generating it...")
        from solarmed_optimization.path_explorer import get_all_paths # Avoid circular import
        
        logger.disable("solarmed_optimization.path_explorer")
        fsm_params = FsmParamsMapping[system].value(**{k: v for k, v in params.items() if k not in ["sample_time", "valid_sequences"]}) # Regulero
        paths, valid_inputs = get_all_paths(
            system=system,
            machine_init_args={"sample_time": params["sample_time"]}, # regulero, si cambia esto se rompe
            fsm_params = fsm_params,
            valid_sequences=params["valid_sequences"],
            max_step_idx=n_horizon,
            use_parallel=True, 
            save_results=True,
            include_valid_inputs=True,
            output_path=paths_path.absolute(),
            id="automatically_generated_from_import_results",
        )
        paths_df = pd.DataFrame(paths, columns=[str(i) for i in range(len(paths[0]))])
        logger.enable("solarmed_optimization.path_explorer")

    if found_results:
        paths_df = pd.read_csv(paths_path / item["paths_file_id"], index_col=0)
        valid_inputs_path: Path = item.get("valid_inputs_file_id", None)
        valid_inputs = None
        if valid_inputs_path is not None:
            valid_inputs_path = paths_path / valid_inputs_path
            if valid_inputs_path.exists():
                with open(valid_inputs_path, 'rb') as f:
                    valid_inputs = pickle.load(f)
    
    # if return_format != "value":
    state_cls = getattr(SupportedSubsystemsStatesMapping, system).value 
    # Already stored as values
    paths_df = paths_df.map(lambda state: convert_to(state, state_cls, return_format=return_format))
    
    if return_metadata:
        return paths_df, valid_inputs, item
    else:
        return paths_df, valid_inputs
    
    
def is_valid_path(path: list[int], valid_seq: list[int]) -> bool:
    """
    Check if a path contains the valid_sequence in order or a prefix of it in order.
    """
    if valid_seq[0] not in path: # or len(valid_seq) > len(path):
        # valid path since the valid sequence is not in the path
        # ~~or valid sequence is longer than the path~~ -> it should match the first elements
        return True
    
    for element_idx, element in enumerate(path):
        if element == valid_seq[0]:
            remaining_path = np.array( path[element_idx:] )
            # Filter out repeated elements
            remaining_path = remaining_path[ np.insert(np.abs(np.diff(remaining_path)) > 0, 0, True) ]
            
            # print(remaining_path[:len(valid_seq)])
            if (list(remaining_path[:len(valid_seq)]) == valid_seq or # Path contains the valid sequence
                list(remaining_path) == valid_seq[:len(remaining_path)]): # The remainining section of the path follows the sequence
                # len(valid_seq) > len(remaining_path)):
                continue
            return False
        
    return True

def filter_paths(paths: list[list], valid_sequence: list, aux_list: list = None) -> list[list] | tuple[list[list], list[list]]:
    """ Filter `paths` that, if present, follow a given `valid_sequence`. Also filters an auxiliary list if provided.
        - Supports both integer and Enum types for the paths and valid_sequence.
        - If enums are provided, the values need to be integers.
        
        TODO: As used in get_all_paths, valid_sequences are chained with AND logic. If starting from a given state, two 
        valid paths should be possible, then we should be able to have OR logic. This is currently not supported.
        
        Proposed interface:
        valid_sequences: list[ list[int] | list[list[int]] ] = [
            [0, 1, 2], # vs1
            [[0, 1, 2], [0, 2, 1]] # vs2, vs3
        ]
        vs1 AND ( vs2 OR vs3 )
        
        TODO: Currently we filter by valid sequences, but sometimes it might be easier to filter by invalid sequences.
        Add support for this.

    Args:
        paths (list[list]): List of paths to be filtered
        valid_sequence (list): Valid sequence of states that the paths should respect

    Returns:
        list[list]: List of paths that, if contain the given sequence, respect it

    Example usage:
        path_sequence: list[int] = [0, 1, 2]
        
        paths: list[list[int]] = [
            [0, 0, 1, 2, 0],       # Should be valid
            [0, 0, 1, 1, 1, 2, 0], # Should be valid
            [0, 1, 1, 1, 1, 1],    # Should be valid
            [0, 0, 0, 0, 0],       # Should be valid
            [0, 1, 0, 1, 0],       # Should be invalid
            [0, 0, 0, 0, 2]        # Should be invalid
        ]

        filtered_paths = filter_paths(paths, path_sequence)
        
        print(f"Original paths ({len(paths)}): {paths}")
        print(f"Filtered paths ({len(filtered_paths)}): {filtered_paths}")
        assert filtered_paths == [*paths[:-2]], "Filtering function not working correctly"
    """
    
    assert isinstance(valid_sequence[0], (Enum, int, float)), "Valid sequence should be a list of integers, Enums, or floats"
    # assert type(paths[0][0]) is type(valid_sequence[0]), "Paths and valid_sequence should have the same type"
    if isinstance(valid_sequence[0], float) or isinstance(paths[0][0], float):
        logger.info("Float values detected in valid_sequence or paths. Converting to integers")
    
    # Convert paths to values
    paths_numeric = paths
    if isinstance(paths[0][0], Enum):
        paths_numeric: list[list[int]] = [[state.value for state in path] for path in paths]
    elif isinstance(paths[0][0], float):
        paths_numeric: list[list[int]] = [[int(state) for state in path] for path in paths]
    valid_seq_numeric = valid_sequence
    if isinstance(valid_sequence[0], Enum):
        valid_seq_numeric: list[int] = [state.value for state in valid_sequence]
    elif isinstance(valid_sequence[0], float):
        valid_seq_numeric: list[int] = [int(state) for state in valid_sequence]
    
    if aux_list is None:
        # return [path for path in paths if is_valid_path(path, valid_sequence)]
        aux_list = [None] * len(paths)

    filtered_paths, filtered_aux_list = zip(*[(path, aux) for path_num, path, aux in zip(paths_numeric, paths, aux_list) if is_valid_path(path_num, valid_seq_numeric)])
    
    if aux_list[0] is not None:
        return list(filtered_paths), list(filtered_aux_list)
    else:
        return list(filtered_paths)
