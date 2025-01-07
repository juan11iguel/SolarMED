from pathlib import Path
from dataclasses import asdict, dataclass
import json
from typing import Any
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
import h5py
from enum import Enum

from solarmed_optimization import ProblemParameters, PopulationResults


class AlgoLogColumns(Enum):
    """Enum for the algorithm logs columns."""
    GACO = ["Gen", "Fevals", "Best", "Kernel", "Oracle", "dx", "dp"]
    SGA = ["Gen", "Fevals", "Best", "Improvement", "Mutations"]
    
    @property
    def columns(self) -> list[str]:
        return self.value

class FilenamesMapping(Enum):
    """Enum for the file ids to filenames mapping."""
    METADATA = "metadata.json"
    PROBLEM_PARAMS = "problem_params.json"
    ALGO_LOGS = "algo_logs.h5"
    OPTIM_DATA = "optim_data.h5"
    DF_HORS = "df_hors.h5"
    DF_SIM = "df_sim.h5"
    
    @property
    def fn(self) -> str:
        return self.value
    
def step_idx_to_step_id(step_idx: int) -> str:
    return f"step_{step_idx:03d}"

def export_optimization_results(output_path: Path, 
                                step_idx: int,
                                metadata: dict[str, str],
                                algo_params: dict, 
                                problem_params: ProblemParameters,
                                algo_log: pd.DataFrame | list[tuple[int|float]], 
                                df_hor: pd.DataFrame,
                                df_sim: pd.DataFrame,
                                pop_results: PopulationResults,
                                figs: list[go.Figure] = None,) -> None:
    
    assert "algo_id" in metadata and "date_str" in metadata, "`algo_id` and `date_str` need to be included in `metadata` dict"    
        
    algo_id: str = metadata["algo_id"]
    step_id: str = step_idx_to_step_id(step_idx)
    
    # Export settings and parameters. Only once
    _path = output_path / FilenamesMapping.METADATA.fn
    if not _path.exists():
        assert algo_params is not None, "Algorithm parameters need to be provided to include them in the metadata"
        metadata["algo_params"] = algo_params
        with open(_path, "w") as f:
            json.dump(metadata, f, indent=4)
    
    _path = output_path / FilenamesMapping.PROBLEM_PARAMS.fn
    if not _path.exists():
        with open(_path, "w") as f:
            json.dump(asdict(problem_params), f, indent=4)
    
    # Export dynamic data. At each step
    
    # Extract algorithm logs
    if not isinstance(algo_log, pd.DataFrame):
        algo_log = pd.DataFrame(algo_log, columns=AlgoLogColumns[algo_id.upper()].columns)
    with pd.HDFStore(output_path / FilenamesMapping.ALGO_LOGS.fn, mode='a') as store:
        store.put(step_id, algo_log)  # Adds a new dataset
    
    ## New entry to append
    # new_entry = {step_id: asdict(step_results)}
    # with open(output_path / 'optim_data.jsonl', 'a') as f:
    #     f.write(f"{json.dumps(new_entry, indent=4)}\n")
    # Save to HDF5
    with h5py.File(output_path / FilenamesMapping.OPTIM_DATA.fn, 'w') as hdf:
        # Create a group for this instance
        group = hdf.create_group(step_id)
        for name, value in asdict(pop_results).items():
            # Store datasets
            if isinstance(value, (list, tuple)):
                group.create_dataset(name, data=value, compression="gzip")
            else:
                # Store scalar attribute
                group.attrs[name] = value
        
    # Dump horizon evaluation dataframes
    with pd.HDFStore(output_path / FilenamesMapping.DF_HORS.fn, mode='a') as store:
        # for i, df in enumerate(df_hor):
        # store.put(f'{step_id}_{i:03d}', df)
        store.put(step_id, df_hor)
            
    # Dump simulation data
    # Replace, probably would be better to append new rows and for df_sim to only carry the needed data not all of it
    # If doing it properly:
    # Initial save
    # df_sim.to_hdf(output_path / "df_sim.h5", key='df_sim', mode='w', format='table')
    # Subsequent exports
    # df_sim.to_hdf(output_path / "df_sim.h5", key='df_sim', mode='a', append=True)
    df_sim.to_hdf(output_path / FilenamesMapping.DF_SIM.fn, key='df_sim', mode='w')
    
    
    # - plot_dec_vars_evolution and save html and png fig to results folder
    # - plot_obj_space_1d (no anumation) # and save html and png fig to results folder
    (output_path / "figures").mkdir(exist_ok=True)
    for i, fig in enumerate(figs):
        _path: Path = output_path / f"figures/{i:02d}_{step_id}"
        # fig.write_html(_path.with_suffix(".html"))
        fig.write_image(_path.with_suffix(".png"), scale=2)
        fig.write_image(_path.with_suffix('.svg'), )
        
    logger.info(f"Exported results to {output_path}")
        
        
@dataclass
class OptimizationResults:
    """ Stores optimization results data.
        Depending on how it is initialized it contains:
        - a single optimization step: OptimizationResults.load(output_path=..., step_idx=X) 
        - or multiple: OptimizationResults.load(output_path=..., step_idx=None)
        """
    metadata: dict[str, Any]
    problem_params: ProblemParameters
    algo_log: pd.DataFrame
    df_hor: pd.DataFrame | list[pd.DataFrame]
    df_sim: pd.DataFrame
    pop_results: PopulationResults | list[PopulationResults]
    algo_params: dict = None
    figs: list[go.Figure] = None
    
    def dump(self, output_path: Path, step_idx: int) -> None:
        """Export the optimization results."""
        export_optimization_results(output_path, step_idx, **self.__dict__)
        
    @classmethod
    def load(cls, output_path: Path, step_idx: int = None) -> 'OptimizationResults':
    
        def build_pop_results(store) -> PopulationResults:
            pop_results = {k: v for k, v in store.attrs.items()} # Extract attributes
            [pop_results.update({k: v[()]}) for k, v in store.items()] # Extract datasets
            
            return PopulationResults(**pop_results)
            
        
        step_id = step_idx_to_step_id(step_idx) if step_idx is not None else None
        
        # Metadata
        metadata: dict[str, Any] = json.load(open(output_path / FilenamesMapping.METADATA.fn, "r"))
        
        # Problem parameters
        problem_params: ProblemParameters = ProblemParameters(**json.load(open(output_path / FilenamesMapping.PROBLEM_PARAMS.fn, "r")))
        
        # Algorithm logs
        with pd.HDFStore(output_path / FilenamesMapping.ALGO_LOGS.fn, mode='r') as store:
            if step_id is None:
                algo_log: list[pd.DataFrame] = [store[key] for key in store]
            else:
                algo_log: pd.DataFrame = store[step_id]
        
        # Horizon evaluation dataframes
        with pd.HDFStore(output_path / FilenamesMapping.DF_HORS.fn, mode='r') as store:
            if step_id is None:
                df_hor: list[pd.DataFrame] = [store[key] for key in store]
            else:
                df_hor: pd.DataFrame = store[step_id]
                
        # Simulation data
        df_sim: pd.DataFrame = pd.read_hdf(output_path / FilenamesMapping.DF_SIM.fn, key='df_sim')
        
        # Population results
        with h5py.File(output_path / FilenamesMapping.OPTIM_DATA.fn, 'r') as hdf:
            if step_id is None:
                pop_results: list[PopulationResults] = [build_pop_results(hdf[step_id]) for step_id in hdf] 
            else:
                pop_results: PopulationResults = build_pop_results(hdf, step_id)
        
        return cls(metadata, problem_params, algo_log, df_hor, df_sim, pop_results)