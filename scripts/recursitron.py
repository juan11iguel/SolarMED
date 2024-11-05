from pathlib import Path
import time
from loguru import logger
from solarmed_optimization.path_explorer import get_all_paths
from solarmed_optimization.path_explorer.utils import import_results

"""WARNING
Depending on the FSM parameters and the max_step_idx, the number of paths can grow exponentially
and produce more paths than stars in the universe :)
"""

logger.disable("solarmed_optimization.path_explorer")

output_path: Path = Path("results")
n_horizons: list[int] = [3, 3, 5, 8, 10]
systems_to_evaluate: list[str] = ['SFTS', 'MED']
return_formats: list[str] = ["value", "name", "enum"]

for n_horizon in n_horizons:
    for system in systems_to_evaluate:
        machine_init_args: dict[str, int] = dict(sample_time=1)

        logger.info(f"Evaluating possible paths for system: {system}, N={n_horizon}")
        
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
            max_step_idx=n_horizon,
            # initial_states=[MedState.IDLE, MedState.OFF],
            use_parallel=True, 
            save_results=True,
            output_path = output_path.absolute(),
        )
        
        logger.info(f"Finished evaluation of system {system}, took {time.time()-start_time:.2f} seconds")
        
        # Test importing
        [import_results(output_path, system, n_horizon, params=machine_init_args, return_format=r) for r in return_formats]
        
    