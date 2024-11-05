from enum import Enum
import json
from pathlib import Path
from typing import Literal
import uuid

from loguru import logger
import pandas as pd
from solarmed_modeling.fsms.utils import convert_to

from solarmed_modeling.fsms import MedState, SfTsState

class SupportedSubsystemsStatesMapping(Enum):
    MED = MedState
    SFTS = SfTsState

def export_results(paths: list[Enum], output_path: Path, system: Literal['SFTS', 'MED'], 
                   params: dict[str, int], computation_time: float) -> None:

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
    file_id: str = f"{system.lower()}_n_{n_horizon}_{uuid.uuid4()}.csv"

    output_path_metadata = output_path / "metadata.json"
    output_path_data = output_path / file_id

    # Convert path Enums to integers
    paths = [[state.value for state in path] for path in paths]
    paths = pd.DataFrame(paths, columns=[str(i) for i in range(len(paths[0]))])
    states_cls = SupportedSubsystemsStatesMapping[system].value
    
    # Count number of paths per initial state
    paths_per_initial_state = {
        states_cls(state_value).name: len(paths_from_state)
        for state_value, paths_from_state in paths.groupby("0").groups.items()
    }
    
    output_dict_new: dict = {
        "n_horizon": n_horizon,
        "computation_time": computation_time,
        "n_paths": n_paths,
        "n_paths_per_initial_state": paths_per_initial_state,
        "parameters": params,
        "file_id": file_id
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

    existing_path_data: str | None = None
    # Check if there is any existing parameter that matches the provided params
    for item_idx, item in enumerate(out):
        if item["parameters"] == params and item["n_horizon"] == n_horizon:
            existing_path_data = str(item["file_id"])
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
    if existing_path_data:
        # Check if the file exists, then delete it
        (output_path / existing_path_data).unlink()
        logger.info(f"Existing data {existing_path_data} has been deleted.")
        
    # Save data to csv
    paths.to_csv(output_path_data)
    logger.info(f"Saved paths to {output_path_data}")


def import_results(paths_path: Path, 
                   system: Literal['SFTS', 'MED'], 
                   n_horizon: int, 
                   params: dict[str, int],
                   return_format: Literal["value", "name", "enum"] = "enum", 
                   return_metadata: bool = False
                  ) -> pd.DataFrame | tuple[pd.DataFrame, dict]:

    with open(paths_path / "metadata.json", 'r') as f:
        metadata_dict = json.load(f)
        
    if system not in metadata_dict.keys():
        logger.error(f"No results found for system {system} with horizon")
        return

    # Check if there is any existing parameter that matches the provided params
    for item in metadata_dict[system]:
        if item["parameters"] == params and item["n_horizon"] == n_horizon:
            existing_path_data = item["file_id"]
            break
    else:
        # If no match is found, append a new entry
        logger.error(f"No data found for {system} with parameters: {params}")
        return

    paths_df = pd.read_csv(paths_path / existing_path_data, index_col=0)
    state_cls = getattr(SupportedSubsystemsStatesMapping, system).value 


    if return_format != "value":
        # Already stored as values
        paths_df = paths_df.map(lambda state: convert_to(state, state_cls, return_format=return_format))
    
    if return_metadata:
        return paths_df, item
    else:
        return paths_df