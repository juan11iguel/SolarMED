from dataclasses import fields, field
from typing import Any, Optional
from itertools import product
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from loguru import logger

from solarmed_optimization import IntegerDecisionVariables, SubsystemId, SubsystemDecVarId, IrradianceThresholds
from solarmed_optimization.utils import infer_attribute_name
    
class ActionType(Enum):
    STARTUP = auto()
    SHUTDOWN = auto()

def generate_pause_update_datetimes(I: pd.Series, t_min: int) -> list[datetime]:
	""" Function that given a timeseries of irradiance returns two candidates for operation pause/restart within the given data span (should be from current datetime+margin up to programmed shutdown-margin)
	The first candidate is placed in the irradiance minimum minus some margin and the second one is located in the irradiance maximum in the range:
	 (1st candidate + t_min, 1st candidate + 2·t_min)
	"""
	...

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
    irradiance_thresholds.max = min(irradiance_thresholds.max, I.max())
    
    if n==1:
        irradiance_levels = np.array( [(irradiance_thresholds.min + irradiance_thresholds.max)/2] ) 
    else:
        irradiance_levels = np.linspace(irradiance_thresholds.min, irradiance_thresholds.max, n)

    candidates = [
        I[I >= level].index[0] 
        # if len(I[I > level]) > 0 else I.idxmax() # If no entry is found, use the maximum
        for level in irradiance_levels
    ]
    
    # Avoid having the operation change right at the start, messes up
    # with the problem initialization since some initial value is set
    # from the initial sample
    if candidates[0] - I.index[0] < I.index.freq:
        candidates[0] += I.index.freq

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
    # If the max threshold is too high, cap the irradiance threshold to the 
    # available data to make sure we have at least one value
    irradiance_thresholds.max = min(irradiance_thresholds.max, I.max())
    
    if n==1:
        irradiance_levels = np.array( [(irradiance_thresholds.min + irradiance_thresholds.max)/2] ) 
    else:
        irradiance_levels = np.linspace(irradiance_thresholds.min, irradiance_thresholds.max, n)
        
    I_reversed = I[::-1]
    candidates = sorted([
        I_reversed[I_reversed >= level].index[0] 
        # if len(I_reversed[I_reversed > level]) else I_reversed.idxmax() # If no entry is found, use the maximum
        for level in irradiance_levels
    ])
    
    # Avoid having the operation change right at the start, messes up
    # with the problem initialization since some initial value is set
    # from the initial sample
    if candidates[0] - I.index[0] < I.index.freq:
        candidates[0] += I.index.freq
    
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
    operation_actions: Optional[dict[str, list[tuple[str, int]]]] = None
    irradiance_thresholds: IrradianceThresholds = field(default_factory=IrradianceThresholds)
    
    @classmethod
    def initialize(cls, operation_actions: dict[str, list[tuple[str, int]]], irradiance_thresholds: IrradianceThresholds) -> "OperationPlanner":
        plans = generate_operation_combos(operation_actions)
        return cls(plans, operation_actions, irradiance_thresholds)
    
    def __str__(self) -> str:
        output = [
            f"Generated {len(self.plans)} operation plans",
            f"Operation actions: {self.operation_actions}",
            f"Irradiance thresholds: {self.irradiance_thresholds}",
        ]
        for plan_idx, plan in enumerate(self.plans):
            output_parts = [f"{plan_idx} |"]
            for subsystem_idx, (subsystem_id, _) in enumerate(self.operation_actions.items()):
                transition_str = " -> ".join(map(str, plan[subsystem_idx]))
                output_parts.append(f" {subsystem_id}: {transition_str} |")
            output.append("".join(output_parts))
        return "\n".join(output)
    
    def generate_decision_series(self, I: pd.Series | list[pd.Series]) -> list[IntegerDecisionVariables]:
        
        I = list(I) if not isinstance(I, list) else I
        
        # Generate update datetimes
        update_datetimes = []
        for subsystem_actions in self.operation_actions.values():
            # initial_action = subsystem_actions[0]
            day_cnt = 0
            dts = []
            for action_idx, action_tuple in enumerate(subsystem_actions):
                # Check if we should move to the next date
                if action_idx > 0 and action_tuple[0] == "startup": # initial_action[0]:
                    # Repeated action, use the next entry in I
                    day_cnt += 1
                dts.append(
                    generate_update_datetimes(I=I[day_cnt], n=action_tuple[1], action_type=action_tuple[0],
                                              irradiance_thresholds=self.irradiance_thresholds)
                )
                # print(f"day {day_cnt} | action: {action_tuple[0]} | n: {action_tuple[1]} | thresh. {self.irradiance_thresholds} | result: {I[day_cnt].loc[dts[-1]]}")
            update_datetimes.append(dts)
        
        dec_vars_list = []
        for plan in self.plans:
            dec_vars = {}
            for subsystem_idx, subsystem_name in enumerate(self.operation_actions.keys()):
                subsystem_id = SubsystemId(subsystem_name).name
                dec_var_type = SubsystemDecVarId[subsystem_id].value
                dec_var_id = infer_attribute_name(IntegerDecisionVariables, dec_var_type)
                
                dec_vars[dec_var_id] = pd.Series(index=np.concatenate(update_datetimes[subsystem_idx]), 
                                                 data=np.concatenate(plan[subsystem_idx])) 
            dec_vars_list.append( IntegerDecisionVariables(**dec_vars) )
            
        return dec_vars_list