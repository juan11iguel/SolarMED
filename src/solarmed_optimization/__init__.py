from typing import Self
import numpy.typing as npt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from loguru import logger


from solarmed_modeling import SupportedStatesType, MedState


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

    def to_dataframe(self):
        return pd.DataFrame(self.dict())


class CostVarsSolarMED(BaseModel):
    """
    Simple class to make sure that the required cost variables are passed
    """
    costs_w: npt.NDArray[np.float64]  # Water cost
    costs_e: npt.NDArray[np.float64]  # Electricity cost

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dataframe(self):
        return pd.DataFrame(self.dict())


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

    def to_dataframe(self):
        return pd.DataFrame(self.dict())

# def on_fitness(ga_instance: pygad.GA, population_fitness: np.ndarray):
#     """
#     Callback that is evaluated after the fitness function is evaluated for all the population
#     Here we already know the best candidate, since computing the prediction horizon is expensive,
#     can we retrieve its outputs
#     UPDATE: Implemented on the fitness function itself
#     """
#     pass