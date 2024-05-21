import numpy as np
import numpy.typing as npt
import pandas as pd
from solarMED_modeling.solar_med import SolarMED
from solarMED_optimization import EnvVarsSolarMED, CostVarsSolarMED, DecVarsSolarMED
from solarMED_optimization.utils import timer_decorator


@timer_decorator
def simulate_episode(model_instance: SolarMED, env_vars: EnvVarsSolarMED, cost_vars: CostVarsSolarMED, dec_vars: DecVarsSolarMED, samples_opt: npt.NDArray[bool]) -> pd.DataFrame:

    # Create a copy to not modify the original model instance
    model_copy: SolarMED = SolarMED(**model_instance.model_dump_instance())

    Nc = np.sum(samples_opt)
    Np = env_vars.Tmed_c_in.shape[0]
    # num_dec_vars = len(DecVarsSolarMED.model_fields)

    if Nc > len(dec_vars.mts_src):
        raise ValueError(f"The number of samples to update the decision variables ({Nc}) is greater than the number of available decision variables updates ({len(dec_vars.mts_src)})")

    # Initialization
    dec_vars_idx = -1
    current_dec_vars: dict | None = None

    # Simulate
    df = pd.DataFrame()  # TODO: Add dimensions? Is it more efficient than just appending sequentially?

    for idx in range(Np):
        # Update decision variables values
        if samples_opt.take(idx) == True:
            dec_vars_idx += 1
            if dec_vars_idx >= Nc:
                raise ValueError(
                    "The number of samples to update the decision variables is greater than the number of available "
                    "decision variables updates")

            # span: tuple[int, int] = (dec_vars_idx * num_dec_vars, (dec_vars_idx + 1) * num_dec_vars)
            current_dec_vars = dec_vars.model_dump_at_index(idx=dec_vars_idx)

        model_copy.step(
            **current_dec_vars,
            **env_vars.model_dump_at_index(idx),
        )

        # Could also be done like the others by implementing `model_dump_at_index`
        model_copy.evaluate_fitness_function(cost_e=cost_vars.costs_e.take(idx), cost_w=cost_vars.costs_w.take(idx))

        df = model_copy.to_dataframe(df)

    return df