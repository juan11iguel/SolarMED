from typing import Self
import numpy as np
import pandas as pd
import numpy.typing as npt
import re
from pydantic import BaseModel, ConfigDict, model_validator
import time
from loguru import logger
import pygad

from solarMED_modeling import SupportedStatesType, MedState
from solarMED_modeling.solar_med import SolarMED


def convert_to_state(state: str, state_cls: SupportedStatesType = MedState) -> SupportedStatesType:
    return getattr(state_cls, state)


class EnvVarsSolarMED(BaseModel):
    """
    Simple class to make sure that the required environment variables are passed
    """
    Tmed_c_in: npt.NDArray[np.float64]  # Seawater temperature
    Tamb: npt.NDArray[np.float64]  # Ambient temperature
    I: npt.NDArray[np.float64]  # Solar radiation
    wmed_f: npt.NDArray[np.float64] = None  # Seawater flow rate

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def check_no_nans_in_fields(self) -> Self:
        for field_name in self.model_fields:
            field_values = getattr(self, field_name)
            if np.isnan(field_values).any():
                raise ValueError(f'No NaNs are allowed in the input data. Found {np.sum(np.isnan(field_values))} NaNs in {field_name}')

        return self

    def model_dump_at_index(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.dict().items()}


class CostVarsSolarMED(BaseModel):
    """
    Simple class to make sure that the required cost variables are passed
    """
    costs_w: npt.NDArray[np.float64]  # Water cost
    costs_e: npt.NDArray[np.float64]  # Electricity cost

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DecVarsSolarMED(BaseModel):
    """
    Simple class to make sure that the correct decision variables are passed.

    It should be updated if any change is made to the SolarMED step method signature.

    mts_src: float,  # Thermal storage decision variables
    - Tsf_out: float,  # Solar field decision variables
    - mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float, med_vacuum_state: int[0,1,2] | MedVacuumState[OFF,LOW,HIGH] #
    """
    mts_src: npt.NDArray[np.float64]  # Thermal storage recirculation flow rate
    Tsf_out: npt.NDArray[np.float64]  # Solar field outlet temperature
    mmed_s: npt.NDArray[np.float64]  # MED heat source flow rate
    mmed_f: npt.NDArray[np.float64]  # MED feedwater flow rate
    Tmed_s_in: npt.NDArray[np.float64]  # MED heat source inlet temperature
    Tmed_c_out: npt.NDArray[np.float64]  # MED condenser outlet temperature
    med_vacuum_state: npt.NDArray[np.int8]  # MED vacuum system state (0: OFF, 1: LOW, 2: HIGH)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump_at_index(self, idx: int | tuple[int, int]) -> dict:

        if isinstance(idx, int):
            return {k: v[idx] for k, v in self.dict().items()}
        elif isinstance(idx, tuple):
            return {k: v[idx[0]:idx[1]] for k, v in self.dict().items()}


def fitness_function(ga_instance: pygad.GA, dec_vars: np.ndarray, solution_idx: int) -> float:  # acumulated profit

    # Np = env_vars.shape[1]
    # Nc = dec_vars.shape[1] # 0: rows, 1: columns
    Np: int = ga_instance.additional_vars["Np"]
    Nc: int = ga_instance.additional_vars["Nc"]
    episode_idx: int = ga_instance.additional_vars["episode_idx"]
    samples_opt: npt.NDArray[bool] = ga_instance.additional_vars["samples_opt"]
    model_copy: SolarMED = SolarMED( **ga_instance.additional_vars["model"].model_dump_instance() )

    # Checks
    # costs_w = costs_w if isinstance(costs_w, np.ndarray) else np.ones(1, Np) * costs_w
    # costs_e = costs_e if isinstance(costs_e, np.ndarray) else np.ones(1, Np) * costs_e
    costs_w: npt.NDArray[bool] = ga_instance.additional_vars["cost_vars"].costs_w
    costs_e: npt.NDArray[bool] = ga_instance.additional_vars["cost_vars"].costs_e
    env_vars: EnvVarsSolarMED = ga_instance.additional_vars["env_vars"]

    if not ga_instance.gene_names:
        raise ValueError("Gene names must be defined in the GA instance")
    num_dec_vars = int(len(ga_instance.gene_names) / Nc)
    # Get the decision variables names by removing the step index suffixes (everything after the last '_')
    base_gene_names = [ re.sub('_[^_]*$', '', gene_name) for gene_name in ga_instance.gene_names[0:num_dec_vars] ]

    # if isinstance(samples_opt, int):
    #     # Sample rate specified, build vector of samples where decision variables are updated
    #     samples_opt = np.zeros(1, Np)
    #     samples_opt[::samples_opt] = True

    if np.sum(samples_opt) != Nc:
        raise ValueError(
            "There should be as many True elements in samples_opt as number of samples to update the decision variables")

    # Initialization
    dec_vars_idx = -1
    acum_profit = np.zeros((Nc, 1))
    current_dec_vars: dict | None = None

    # Simulate
    df = pd.DataFrame()  # TODO: Add dimensions? Is it more efficient than just appending sequentially?
    # Time the loop

    start_time = time.time()
    for idx in range(Np):
        # Update decision variables values
        if samples_opt.take(idx) == True:
            dec_vars_idx += 1
            if dec_vars_idx >= Nc:
                raise ValueError(
                    "The number of samples to update the decision variables is greater than the number of available "
                    "decision variables updates")

            span: tuple[int, int] = (dec_vars_idx*num_dec_vars, (dec_vars_idx+1)*num_dec_vars)
            # IMPORTANT! Keep the order of the decision variables as they are in the gene_names
            current_dec_vars = {k: v for k, v in zip(base_gene_names, dec_vars[span[0]:span[1]])}

        model_copy.step(
            **current_dec_vars,
            **env_vars.model_dump_at_index(idx),
        )

        acum_profit[dec_vars_idx] += model_copy.evaluate_fitness_function(cost_e=costs_e.take(idx),
                                                                          cost_w=costs_w.take(idx))

        df = model_copy.to_dataframe(df)

        # logger.debug(f"Simulation step {idx} completed for solution {solution_idx}")

    exec_time = time.time() - start_time
    fitness = np.sum(acum_profit)
    best_fitness = np.max( ga_instance.previous_generation_fitness ) if ga_instance.generations_completed > 0 else -np.inf

    # Save results if best candidate
    # Can't we use some property from ga_instance instead of declaring our own `solution_fitness`?
    # UPDATE: This is not about the solutions, but about saving the simultation results
    if "best_fitness" not in ga_instance.additional_outputs:
        raise ValueError("best_fitness must be initialized with the GA instance (and set to a very high negative value, e.g. -np.inf)")

    #
    if fitness > ga_instance.additional_outputs["best_fitness"]:
        logger.debug(
            f"New best solution found with fitness {fitness:.2f} for generation {ga_instance.generations_completed} and solution {solution_idx}. Saving simulation results")

        ga_instance.additional_outputs["best_sol_df"] = df
        ga_instance.additional_outputs["solution_idx"] = solution_idx
        ga_instance.additional_outputs["best_fitness"] = fitness


    # Log progress
    if solution_idx is None:
        # Vaya cosa más rara
        solution_idx = 0
    total_expected_runtime = (ga_instance.num_generations * ga_instance.sol_per_pop * exec_time) / 3600
    total_runtime_left = ((ga_instance.num_generations - ga_instance.generations_completed) * ga_instance.sol_per_pop * exec_time) / 3600
    total_runtime_left -= (exec_time * (solution_idx + 1)) / 3600 # Remove the time already computed up to the current solution
    rel_progress = (total_expected_runtime - total_runtime_left) / total_expected_runtime * 100

    logger.debug(f"[Episode progress {episode_idx/Nc*100:.0f} %] - Gen.Gene {ga_instance.generations_completed}.{solution_idx} ({ga_instance.sol_per_pop}) - fitness: {fitness:.2f} ({best_fitness:.2f}) - Comp. cost: {exec_time:.0f} sec/episode - {exec_time/Np:.3f} sec/step - Time remaining: {total_runtime_left:.1f}h from a total of {total_expected_runtime:.2f}h ({rel_progress:.2f} %)")

    return fitness

# def on_fitness(ga_instance: pygad.GA, population_fitness: np.ndarray):
#     """
#     Callback that is evaluated after the fitness function is evaluated for all the population
#     Here we already know the best candidate, since computing the prediction horizon is expensive,
#     can we retrieve its outputs
#     UPDATE: Implemented on the fitness function itself
#     """
#     pass