from dataclasses import asdict
from pathlib import Path
import time
from loguru import logger
from solarmed_modeling.fsms.med import FsmParameters as MedFsmParameters
from solarmed_modeling.fsms.sfts import FsmParameters as SfTsFsmParameters
from solarmed_optimization.path_explorer import get_all_paths
from solarmed_optimization.path_explorer.utils import import_results

"""WARNING
Depending on the FSM parameters and the max_step_idx, the number of paths can grow exponentially
and produce more paths than stars in the universe :)
"""

logger.disable("solarmed_optimization.path_explorer")

use_parallel: bool = True
output_path: Path = Path("results")
# n_horizons: list[int] = [3, 3, 5, 8, 10] # Repeat the number of horizons to evaluate rewriting the results
n_horizons: list[int] = [5]
return_formats: list[str] = ["value", "name", "enum"]
sample_time: int = 1
include_valid_inputs: bool = True

params_to_test: dict[str, list[MedFsmParameters] | list[SfTsFsmParameters]] = {
    'MED': [
        # MedFsmParameters(
        #     vacuum_duration_time = 2,
        #     brine_emptying_time = 1,
        #     startup_duration_time = 1,
        #     off_cooldown_time=0,
        #     active_cooldown_time=0,
        # ),
        MedFsmParameters(
            vacuum_duration_time = 2,
            brine_emptying_time = 1,
            startup_duration_time = 1,
            off_cooldown_time=9999,
            active_cooldown_time=5,
        )
    ],
    'SFTS': [
        # SfTsFsmParameters(
        #     recirculating_ts_enabled=True,
        #     idle_cooldown_time = 0,
        # ),
        SfTsFsmParameters(
            recirculating_ts_enabled = False,
            idle_cooldown_time = 3,
        ),
    ]
}
alternative_ids: dict[str, list[str]] = {
    'MED': ["lightly-contrained", "restricted"],
    'SFTS': ["lightly-contrained", "restricted"]
}
assert all( [len(alts) == len(params)] for alts, params in zip(alternative_ids.values(), params_to_test.values()) ), \
    "There should same number of alternative ids as number of params to test, per system"

if not output_path.exists():
    output_path.mkdir()

machine_init_args: dict[str, int] = dict(sample_time=sample_time)

for idx_n, n_horizon in enumerate(n_horizons):
    for idx_s, (system, params_list) in enumerate(params_to_test.items()):
        for idx_p, (params, alt_id) in enumerate(zip(params_list, alternative_ids[system])):

            logger.info(f"Evaluating possible paths for system: {system} ({idx_s+1}/{len(params_to_test.keys())}), N={n_horizon} ({idx_n+1}/{len(n_horizons)}), params {idx_p+1}/{len(params_list)}")

            start_time = time.time()
            get_all_paths(
                system=system,
                machine_init_args=machine_init_args,
                fsm_params=params,
                max_step_idx=n_horizon,
                # initial_states=[MedState.IDLE, MedState.OFF],
                use_parallel=use_parallel, 
                save_results=True,
                output_path=output_path.absolute(),
                id=alt_id,
                include_valid_inputs = include_valid_inputs
            )
            
            logger.info(f"Finished evaluation of system {system}, took {time.time()-start_time:.2f} seconds")
            
            # Test importing
            [import_results(output_path, system, n_horizon, 
                            params={**asdict(params), **machine_init_args}, return_format=r) for r in return_formats]
        
    