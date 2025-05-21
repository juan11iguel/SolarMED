from typing import Optional
from itertools import product
from enum import Enum, auto
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from loguru import logger
import copy
import pygmo as pg

from solarmed_optimization import (
    IntegerDecisionVariables, 
    SubsystemId, 
    SubsystemDecVarId, 
    IrradianceThresholds,
    OperationActionType,
    OperationUpdateDatetimesType,
    InitialDecVarsValues,
    AlgoParams,
    PygmoArchipelagoTopologies
)
from solarmed_optimization.problems import BaseNlpProblem
from solarmed_optimization.utils import (infer_attribute_name, 
                                         flatten_list)
    
class ActionType(Enum):
    STARTUP = auto()
    SHUTDOWN = auto()


def generate_pause_update_datetimes(I: pd.Series, t_min: int) -> list[datetime]:
	""" Function that given a timeseries of irradiance returns two candidates for operation pause/restart within the given data span (should be from current datetime+margin up to programmed shutdown-margin)
	The first candidate is placed in the irradiance minimum minus some margin and the second one is located in the irradiance maximum in the range:
	 (1st candidate + t_min, 1st candidate + 2·t_min)
	"""
	...

def check_irradiance_data(I: pd.Series) -> AssertionError | None:
    assert len(I) > 0, "Irradiance series is empty, probably a consequence of more actions in operation_plan than days in irradiance data"

def generate_startup_update_datetimes(I: pd.Series, n: int, irradiance_thresholds: IrradianceThresholds = IrradianceThresholds()) -> list[datetime]:
    """ Function that given a timeseries of irradiance returns n candidates 
        for operation startup within the given data span

        Args:
        I (pd.Series): timeseries of irradiance
        n (int): number of candidates to generate

        Returns:
        list[datetime]: list of n datetimes candidates for operation startup
    """
    # If the max threshold is too high, cap the irradiance threshold to the 
    # available data to make sure we have at least one value
    check_irradiance_data(I)
    
    irradiance_thresholds = copy.deepcopy(irradiance_thresholds) # Avoid modifying the original object
    irradiance_thresholds.upper = min(irradiance_thresholds.upper, I.max())
    
    if n==1:
        irradiance_levels = np.array( [(irradiance_thresholds.lower + irradiance_thresholds.upper)/2] ) 
    else:
        irradiance_levels = np.linspace(irradiance_thresholds.lower, irradiance_thresholds.upper, n)
    # print(irradiance_levels, I.max())
    candidates = [
        I[I >= level].index[0] 
        # if len(I[I > level]) > 0 else I.idxmax() # If no entry is found, use the maximum
        for level in irradiance_levels
    ]
    
    # Avoid having the operation change right at the start, messes up
    # with the problem initialization since some initial value is set
    # from the initial sample
    # if candidates[0] - I.index[0] < I.index.freq:
    #     candidates[0] += I.index.freq

    return candidates

def generate_shutdown_update_datetimes(I: pd.Series, n: int, irradiance_thresholds: IrradianceThresholds = IrradianceThresholds()) -> list[datetime]:
    """ Function that given a timeseries of irradiance returns n candidates 
    for operation shutdown within the given data span

    Args:
        I (pd.Series): timeseries of irradiance
        n (int): number of candidates to generate

    Returns:
        list[datetime]: list of n datetimes candidates for operation shutdown
    """
    check_irradiance_data(I)
    
    # If the max threshold is too high, cap the irradiance threshold to the 
    # available data to make sure we have at least one value
    irradiance_thresholds.upper = min(irradiance_thresholds.upper, I.max())
    
    if n==1:
        irradiance_levels = np.array( [(irradiance_thresholds.lower + irradiance_thresholds.upper)/2] ) 
    else:
        irradiance_levels = np.linspace(irradiance_thresholds.lower, irradiance_thresholds.upper, n)
        
    I_reversed = I[::-1]
    candidates = sorted([
        I_reversed[I_reversed >= level].index[0] 
        # if len(I_reversed[I_reversed > level]) else I_reversed.idxmax() # If no entry is found, use the maximum
        for level in irradiance_levels
    ])
    
    # Avoid having the operation change right at the start, messes up
    # with the problem initialization since some initial value is set
    # from the initial sample
    # if candidates[0] - I.index[0] < I.index.freq:
    #     candidates[0] += I.index.freq
    
    # print(candidates)
    return candidates
	
# Base generators
def generate_startup_plans(subsystem_inputs_cls: Enum, n:int) -> list[tuple[int]]:
    # Not too generic, maybe we could infer the maximum value in the Enum members
    startup_options_per_update: list[list[int]] = [[value.value for value in subsystem_inputs_cls.__members__.values()]]*(n-1) + [[subsystem_inputs_cls.ACTIVE.value]]

    # Generate all combinations
    all_combinations = product(*startup_options_per_update)

    # Filter combinations based on the restriction
    valid_combinations = [
        combination for combination in all_combinations
        if all(x <= y for x, y in zip(combination, combination[1:]))
    ]
    # # Output the valid combinations
    # for comb in valid_combinations:
    #     print(comb)   
    return valid_combinations
    
def generate_shutdown_plans(subsystem_inputs_cls: Enum, n:int) -> list[tuple[int]]:
    shutdown_options_per_update: list[list[int]] = [[value.value for value in subsystem_inputs_cls.__members__.values()]]*(n-1) + [[subsystem_inputs_cls.OFF.value]]

    # Generate all combinations
    all_combinations = product(*shutdown_options_per_update)

    # Filter combinations based on the restriction
    valid_combinations = [
        combination for combination in all_combinations
        if all(x >= y for x, y in zip(combination, combination[1:]))
    ]
    return valid_combinations

def generate_plans(subsystem_inputs_cls: Enum, n:int, action_type: ActionType | str) -> list[tuple[int]]:
    """ Wrapper function to call specific plan generators """
    
    if action_type == ActionType.STARTUP or action_type.lower() == ActionType.STARTUP.name.lower():
        return generate_startup_plans(subsystem_inputs_cls, n)
    if action_type == ActionType.SHUTDOWN or action_type.lower() == ActionType.SHUTDOWN.name.lower():
        return generate_shutdown_plans(subsystem_inputs_cls, n)
    
    raise ValueError(f"Unknown option {action_type}, should be one of {[name for name in action_type.__members__.names()]}")
    
def generate_update_datetimes(I: pd.Series, n: int, action_type: ActionType | str, irradiance_thresholds: IrradianceThresholds) -> list[datetime]:
    """ Wrapper function to call specific update time location generators """
    
    if action_type == ActionType.STARTUP or action_type.lower() == ActionType.STARTUP.name.lower():
        return generate_startup_update_datetimes(I, n, irradiance_thresholds)
    if action_type == ActionType.SHUTDOWN or action_type.lower() == ActionType.SHUTDOWN.name.lower():
        return generate_shutdown_update_datetimes(I, n, irradiance_thresholds)
    
    raise ValueError(f"Unknown option {action_type}, should be one of {[name for name in action_type.__members__.names()]}")

def generate_operation_datetimes(I: pd.Series, operation_actions: OperationActionType, irradiance_thresholds: IrradianceThresholds) -> OperationUpdateDatetimesType:
    """ Function that returns an object equivalent to operation actions but 
    replaces the number of updates with the datetimes of the possible updates"""
    
    irradiance_thresholds = copy.deepcopy(irradiance_thresholds)
    
    operation_datetimes = {}
    for subsystem_id, subsystem_actions in operation_actions.items():
        operation_datetimes[subsystem_id] = []

        current_day = I.index[0].day
        for action_idx, action_tuple in enumerate(subsystem_actions):
            action_type = action_tuple[0]
            n_updates = action_tuple[1]
            
            # Check if we should move to the next date
            if action_idx > 0 and action_tuple[0] == "startup": # Repeated startup action, move to the next day
                current_day += 1

            operation_datetimes[subsystem_id].append((
                action_type,
                generate_update_datetimes(
                    I=I.loc[I.index.day==current_day], 
                    n=n_updates, 
                    action_type=action_type, 
                    irradiance_thresholds=irradiance_thresholds
                )
            ))
            
    operation_datetimes = adjust_timestamps(operation_datetimes)
            
    return operation_datetimes

def generate_operation_combos(operation_actions: dict[str, list[tuple[str, int]]]) -> list[tuple[tuple[int], ...]]:
    """Generate all possible operation plans for the subsystems based on the given actions

    Args:
        operation_actions (dict[str, list[tuple[str, int]]]): {subsystem_id: [(action: startup/shutdown/etc, n: number of updates/DoF), ...]}

    Returns:
        list[tuple[tuple[int], ...]]: list of all possible operation plans
        
    Examples:
        operation_actions: dict = {
                    # Day 1 -----------------------  # Day 2 -----------------------
            "sfts": [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)],
            "med":  [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)]
        }
            
        # Functional approach
        operation_plans = generate_operation_combos(operation_actions)
        print(f"Generated {len(operation_plans)} operation plans")
        for plan in operation_plans:
            output_parts = []
            for subsystem_idx, (subsystem_id, actions) in enumerate(operation_actions.items()):
                # transitions = plan[0] if subsystem_id == "sfts" else plan[1]
                transition_str = " -> ".join(map(str, plan[subsystem_idx]))
                output_parts.append(f"{subsystem_id}: {transition_str}")
            print(", ".join(output_parts))
            
        # Expected output
        # > Generated 100 operation plans
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 1, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (0, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (1, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (1, 1, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (1, 1, 1) -> (0, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (1, 1, 1) -> (1, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (1, 1, 1) -> (1, 1, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 0) -> (0, 0, 0) -> (0,) -> (0,)
        # > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 1, 0) -> (1,) -> (0,)
        # > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (0, 0, 0) -> (1,) -> (0,)
        # ...
    """
                   
    operation_plans_subsystems: list[tuple[tuple[int], tuple[int]]] = []
    for subsystem_id_str, actions in operation_actions.items():
        
        subsystem_id = SubsystemId(subsystem_id_str).name
        subsystem_inputs_cls = SubsystemDecVarId[subsystem_id].value
        combs = []
        # I am not sure I like this, in fact, I am quite sure I don't like it
        # Now it's generic, but at what price?
        for action in actions:
            n = action[1]
            if n>3:
                logger.warning(f"Number of updates greater than 3 ({n=} for {subsystem_id_str}-{action[0]}) are likely to produce an exponential explosion of options")
            combs.append(
                generate_plans(subsystem_inputs_cls, n=n, action_type=action[0])
            )
        operation_plans_subsystems.append(
            list(product(*combs))
        )        
        # Add the null operation option (tuple([0]*n), tuple([0]*n))
        operation_plans_subsystems[-1].append(
            tuple([tuple([0]*action[1]) for action in actions])
        ) 
        
    # Combine the operation plans for each subsystem
    operation_plans = list(product(*operation_plans_subsystems))
    
    # Filter out plans where the first subsystem never starts up, the second should never start up
    # operation_plans = [
    #     plan for plan in operation_plans
    #     if any(startup[0] != 0 for startup in plan)
    # ]
    
    return operation_plans
    

    # # Output the combined combinations
    # for startup, shutdown in operation_plans:
    #     print(f"Startup: {startup}, Shutdown: {shutdown}")

def filter_out_non_operation_duplicates(int_dec_vars_list: list[IntegerDecisionVariables], operation_min_duration: timedelta) -> list[IntegerDecisionVariables]:
    """ Function that removes integer decision variables from a list of them
        where there are multiples alternatives with a null operation duration """
    
    len0 = len(int_dec_vars_list)

    for null_idx, int_dec_vars in enumerate(int_dec_vars_list):
        if np.sum([int_var_series.sum() for int_var_series in asdict(int_dec_vars).values()]) < 1:
            break
            
    # Remove the null operation plan from the list
    int_dec_vars_list = [int_dec_vars for idx, int_dec_vars in enumerate(int_dec_vars_list) 
                         if int_dec_vars.get_total_active_duration() > operation_min_duration or idx==null_idx]
                
    if len0 - len(int_dec_vars_list) > 0:
        logger.info(f"Removed {len0 - len(int_dec_vars_list)} out of {len0} non-operation plans from the list since active duration was less than {operation_min_duration.total_seconds()/3600:.1f} hours. Remaining: {len(int_dec_vars_list)}")

    return int_dec_vars_list
 
def adjust_timestamps(data: OperationUpdateDatetimesType) -> OperationUpdateDatetimesType:
    """ Adjusts the timestamps in the operation update datetimes to ensure they are unique.
    This is done by adding a small timedelta (1 second) to any duplicate timestamps.
    Args:
        data (OperationUpdateDatetimesType): The operation update datetimes to adjust.
    Returns:
        OperationUpdateDatetimesType: The adjusted operation update datetimes with unique timestamps.
    """
    for subsystem, actions in data.items():
        seen_timestamps = set()  # Track unique timestamps
        for action, timestamps in actions:
            for i, ts in enumerate(timestamps):
                # Check if the timestamp is already in the set (collision)
                while ts in seen_timestamps:
                    # Add a small timedelta (1 second) to resolve the conflict
                    ts += timedelta(seconds=1)
                # Update the timestamp in place
                timestamps[i] = ts
                # Add the (now unique) timestamp to the set
                seen_timestamps.add(ts)
    return data
  
@dataclass
class OperationPlanner:
    """

    Example:
        operation_planner = OperationPlanner.initialize(operation_actions)
        print(operation_planner)
        
        # Expected output
        > Generated 100 operation plans
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 1, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (0, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (1, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (1, 1, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (1, 1, 1) -> (0, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (1, 1, 1) -> (1, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (1, 1, 1) -> (1, 1, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,), med: (0, 0, 0) -> (0, 0, 0) -> (0,) -> (0,)
        > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (0, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 0, 1) -> (1, 1, 0) -> (1,) -> (0,)
        > sfts: (0, 0, 1) -> (1, 0, 0) -> (1,) -> (0,), med: (0, 1, 1) -> (0, 0, 0) -> (1,) -> (0,)
        ...    
        """
    plans: list[tuple[tuple[int], ...]]
    operation_actions: OperationActionType
    irradiance_thresholds: IrradianceThresholds
    operation_datetimes: Optional[OperationUpdateDatetimesType] = None
    
    @classmethod
    def initialize(cls, operation_actions: OperationActionType, irradiance_thresholds: IrradianceThresholds, ) -> "OperationPlanner":
        plans = generate_operation_combos(operation_actions)
        return cls(plans, operation_actions, irradiance_thresholds)
    
    def __str__(self) -> str:
        output = [
            f"Generated {len(self.plans)} operation plans",
            f"Operation actions: {self.operation_actions}",
            f"Irradiance thresholds: {self.irradiance_thresholds}",
        ]
        for plan_idx, plan in enumerate(self.plans):
            output_parts = [f"{plan_idx:03d} |"]
            for subsystem_idx, (subsystem_id, _) in enumerate(self.operation_actions.items()):
                transition_str = " -> ".join(map(str, plan[subsystem_idx]))
                output_parts.append(f"\n{subsystem_id}:  \t{transition_str} |")
            output.append("".join(output_parts))
        return "\n".join(output)
    
    def generate_decision_series(self, I: pd.Series, 
                                 operation_min_duration: timedelta = timedelta(seconds=1),
                                 initial_dec_var_values: Optional[InitialDecVarsValues] = None) -> list[IntegerDecisionVariables]:
        """ Generate the decision series for the given irradiance data
        Args:
            I (pd.Series): timeseries of irradiance. Should be the irradiance data given to the optimization layer resampled to the model sample time.
        Returns:
            list[IntegerDecisionVariables]: list of decision variables for each plan
        """

        # update_datetimes = []
        # for subsystem_id, subsystem_actions in self.operation_actions.items():
        #     # initial_action = subsystem_actions[0]
        #     current_day = I.index[0].day
        #     dts = []
        #     for action_idx, action_tuple in enumerate(subsystem_actions):
        #         action_type = action_tuple[0]
        #         n_updates = action_tuple[1]
        #         # Check if we should move to the next date
        #         if action_idx > 0 and action_tuple[0] == "startup": # Repeated startup action, move to the next day
        #             current_day += 1
        #         dts.append(
        #             generate_update_datetimes(
        #                 I=I.loc[I.index.day==current_day], 
        #                 n=n_updates, 
        #                 action_type=action_type, 
        #                 irradiance_thresholds=self.irradiance_thresholds
        #             )
        #         )
        #         # print(f"subsystem {subsystem_id} | day {current_day} | action: {action_type} | n: {n_updates} | result: {I.loc[dts[-1]]}")
        #     update_datetimes.append(dts)
        
        operation_datetimes = generate_operation_datetimes(I, self.operation_actions, self.irradiance_thresholds)
        self.operation_datetimes = operation_datetimes
        operation_datetimes_flat = [
            flatten_list([action_tuple[1] for action_tuple in action_tuples])
            for action_tuples in operation_datetimes.values()
        ] # -> list[list[datetime]]

        dec_var_ids = [
            infer_attribute_name(
                IntegerDecisionVariables, 
                SubsystemDecVarId[subsystem_id.upper()].value
            ) 
            for subsystem_id in self.operation_actions.keys()
        ]

        dec_vars_list: list[IntegerDecisionVariables] = []
        for plan in self.plans:
            dec_vars_dict = {
                dec_var_id: pd.Series(
                    index=operation_datetimes_flat[subsystem_idx],
                    data=np.concatenate(plan[subsystem_idx]),
                    dtype=int
                )
                for subsystem_idx, dec_var_id in enumerate(dec_var_ids)
            }
            dec_vars = IntegerDecisionVariables(**dec_vars_dict)
            # print("before", dec_vars.sfts_mode)
            
            if initial_dec_var_values is not None:
                dec_vars.add_initial_values(initial_dec_var_values)
                # print("after", dec_vars)
                
            dec_vars_list.append(dec_vars)
            
        # Filter out duplicate inactive operation plans
        dec_vars_list = filter_out_non_operation_duplicates(
            int_dec_vars_list=dec_vars_list, 
            operation_min_duration=operation_min_duration
        )
            
        return dec_vars_list
    
    
def build_archipielago(
    problems: list[BaseNlpProblem], 
    algo_params: AlgoParams, 
    x0: Optional[list[np.ndarray]] = None, 
    fitness0: Optional[list[float]] = None,
    topology: PygmoArchipelagoTopologies = "unconnected",
) -> pg.archipelago:
    
    if x0 is not None:
        assert len(problems) == len(x0), f"Number of initial populations ({len(x0)}) should match number of problems ({len(problems)})"
        assert fitness0 is not None, "Initial fitness should be provided if initial populations are provided"
    
    archi = pg.archipelago(t=getattr(pg, topology)())
    for problem_idx, problem in enumerate(problems):
        # logger.debug(f"Adding {problem_idx=} / {len(problems)-1} to archipielago")
        
        # Initialize problem instance
        prob = pg.problem(problem)
        
        # Initialize population
        pop = pg.population(prob, size=algo_params.pop_size, seed=0)
        if x0 is not None and x0[problem_idx] is not None:
            if fitness0[problem_idx] is None:
                pop.set_x(0, x0[problem_idx])
            else:
                pop.set_xf(0, x0[problem_idx], [fitness0[problem_idx]])
        
        algo = pg.algorithm(getattr(pg, algo_params.algo_id)(**algo_params.params_dict))
        algo.set_verbosity( algo_params.log_verbosity )
        
        # 6. Build up archipielago
        archi.push_back(
            # Setting use_pool=True results in ever-growing memory footprint for the sub-processes
            # https://github.com/esa/pygmo2/discussions/168#discussioncomment-10269386
            pg.island(udi=pg.mp_island(use_pool=False), algo=algo, pop=pop, )
        )
        
    return archi

