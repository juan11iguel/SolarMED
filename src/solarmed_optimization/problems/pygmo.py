import numpy as np
from . import BaseMinlpProblem, get_bounds, evaluate_fitness


class MinlpProblem(BaseMinlpProblem):
    
    def get_bounds(self, readable_format: bool = False) -> np.ndarray[float | int] | tuple[np.ndarray[float | int], np.ndarray[float | int]]:
        return get_bounds(self, readable_format=readable_format)
    
    def fitness(self, x: np.ndarray[float | int]) -> list[float]:
        return evaluate_fitness(self, x)
    
    def get_nic(self) -> int:
        """ Get number of inequality constraints """
        return self.n_dec_vars if self.use_inequality_contraints else 0

    def get_nix(self) -> int:
        """ Get integer dimension """
        return sum([getattr(self.dec_var_updates, var_id) for var_id in self.dec_var_int_ids])
