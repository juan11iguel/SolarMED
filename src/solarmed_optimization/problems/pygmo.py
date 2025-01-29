import numpy as np
from solarmed_optimization.utils import flatten_list
from solarmed_optimization.problems import BaseMinlpProblem, evaluate_fitness_minlp

class MinlpProblem(BaseMinlpProblem):
    
    def get_bounds(self) -> tuple[np.ndarray[float | int], np.ndarray[float | int]]:
        
        # output = (self.box_bounds_lower, self.box_bounds_upper)
        # output = (
        #     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0], order='C'),
        #     np.array([ 0,  0,  1,  1,  0, 1,  0,  0,  0,  0,  0,0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1], order='C')
        # )
        # print(len(output[0]), len(output[1]), self.size_dec_vector)
        # output = (
        #     np.zeros((self.size_dec_vector, ), order='C'),# * np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0.]),
        #     np.ones((self.size_dec_vector, ), order='C') * 2# * np.array([ 0. ,  0. ,  2.0,  2.0,  0. , 2.,  0. ,  0. ,  0. ,  0. ,  0. ,0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  1. ,  0. ,  0. ,  0. ,  1. ])
        # )
        # print(output)
        return flatten_list(self.box_bounds_lower), flatten_list(self.box_bounds_upper)
        
    
    def fitness(self, x: np.ndarray[float | int], store_x: bool = True) -> list[float]:
        # return [0.0]
        
        output = evaluate_fitness_minlp(self, x)
        
        # Store decision vector
        if store_x:
            self.x_evaluated.append(x.tolist())
            self.fitness_history.append(output[0])
        
        # return evaluate_fitness(self, x)
        return output
    
    # def batch_fitness(self, dvs: np.ndarray[float | int], store_x: bool = True) -> list[float]:
        
    #     # Convert a batch of decision vectors, dvs, stored contiguously
    #     # to a list of decision vectors: [dv1, dv2, ..., dvn] -> [[dv1], [dv2], ..., [dvn]]
    #     x: list[np.ndarray[float | int]] = []
    #     for idx_start in range(0, len(dvs)-1, step=self.size_dec_vector):
    #         x.append(dvs[idx_start:idx_start+self.size_dec_vector])
        
    #     if store_x:
    #         self.x_evaluated.extend(x)
    #         # TODO: Add fitness
            
    #     output_list = evaluate_fitness(self, x)
        
    #     # Return a contiguous array by converting the list of outputs:
    #     # [[out1], [out2], ..., [outn]] -> [out1, out2, ..., outn] 
    #     return np.array(output_list).flatten()
    
    def get_nic(self) -> int:
        """ Get number of inequality constraints """
        return self.n_dec_vars if self.use_inequality_contraints else 0

    def get_nix(self) -> int:
        """ Get integer dimension """
        return sum([getattr(self.dec_var_updates, var_id) for var_id in self.dec_var_int_ids])
    
    # def gradient(self, x: np.ndarray[float | int]) -> list[float]:
    #     return pg.estimate_gradient(lambda x: self.fitness(x), x)
