from enum import Enum
import inspect
import json
from pathlib import Path
from typing import Literal
import uuid
from dataclasses import asdict, is_dataclass
import pickle

from loguru import logger
import pandas as pd
from solarmed_modeling.fsms.utils import convert_to

from solarmed_modeling.fsms import MedState, SfTsState
from solarmed_modeling.fsms.med import MedFsm
from solarmed_modeling.fsms.sfts import SolarFieldWithThermalStorageFsm as SfTsFsm

class SupportedSubsystemsStatesMapping(Enum):
    MED = MedState
    SFTS = SfTsState
    
class SupportedFsmsMapping(Enum):
    MED = MedFsm
    SFTS = SfTsFsm

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
    fsm_cls = SupportedFsmsMapping[system].value
    
    # Convert valid inputs to DataFrame
    # TODO: Check this a hundred times
    # This is absurdly complex, we should just have inputs be another dataclass
    method_signature = inspect.signature(fsm_cls.step)
    input_ids: list[str] = [param.name for param in method_signature.parameters.values() if param.default == inspect.Parameter.empty and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and param.name != "self"]
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
        "valid_inputs_file_id": valid_inputs_file_id
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
                  ) -> tuple[pd.DataFrame, list[list[list[float]]]] | tuple[pd.DataFrame, list[list[list[float]]], dict]:

    with open(paths_path / "metadata.json", 'r') as f:
        metadata_dict = json.load(f)
        
    if system not in metadata_dict.keys():
        raise ValueError(f"No results found for system {system} with horizon")
    
    if is_dataclass(params):
        params = asdict(params)

    # extra_keys: set = set(metadata_dict[system][0]["parameters"].keys()) - set(params.keys())
    # Check if there is any existing parameter that matches the provided params
    for item in metadata_dict[system]:
        # Remove any additional keys in item["parameters"] with respect to params
        # item_params = {key: item["parameters"][key] for key in set(params.keys())}
        if item["parameters"] == params and item["n_horizon"] == n_horizon:
            break
    else:
        # If no match is found, append a new entry
        raise ValueError(f"No data found for {system}, horizon={n_horizon} and parameters: {params}")


    paths_df = pd.read_csv(paths_path / item["paths_file_id"], index_col=0)
    valid_inputs_path: Path = item.get("valid_inputs_file_id", None)
    valid_inputs = None
    if valid_inputs_path is not None:
        valid_inputs_path = paths_path / valid_inputs_path
        if valid_inputs_path.exists():
            with open(valid_inputs_path, 'rb') as f:
                valid_inputs = pickle.load(f)
    
    if return_format != "value":
        state_cls = getattr(SupportedSubsystemsStatesMapping, system).value 
        # Already stored as values
        paths_df = paths_df.map(lambda state: convert_to(state, state_cls, return_format=return_format))
    
    if return_metadata:
        return paths_df, valid_inputs, item
    else:
        return paths_df, valid_inputs