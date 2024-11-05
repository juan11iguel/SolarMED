from dataclasses import dataclass
import math
import numpy as np
import pygmo as pg
from solarmed_optimization.utils import timer_decorator

@dataclass
class sphere_function:
    dim: int = 5 # To make the problem scalable

    def fitness(self, x: np.ndarray[float | int]) -> list[float] | np.ndarray[float]:
        """Fitness method

        Args:
            x (np.ndarray):  is called decision vector or chromosome, and is made of real numbers and integers

        Returns:
            np.ndarray: fitness of the input decision vector (concatenating the objectives, the equality and the inequality constraints)
        """

        return [sum(x*x)]

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """This method will return the box-bounds of the problem. 
        Infinities in the bounds are allowed.

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower bounds, upper bounds)
        """

        return ([-1] * self.dim, [1] * self.dim)

    def get_name(self) -> str:
        """ Problem’s name """
        return "Sphere Function"

    def get_extra_info(self) -> str:
        """ Problem’s extra info. """
        return "\tDimensions: " + str(self.dim)
    
def fitness(x: np.ndarray | int) -> np.ndarray[float] | list[float]:
    obj = 0
    for i in range(3):
        obj += (x[2*i-2]-3)**2 / 1000. - (x[2*i-2]-x[2*i-1]) + math.exp(20.*(x[2*i - 2]-x[2*i-1]))

    ce1 = 4*(x[0]-x[1])**2+x[1]-x[2]**2+x[2]-x[3]**2
    ce2 = 8*x[1]*(x[1]**2-x[0])-2*(1-x[1])+4*(x[1]-x[2])**2+x[0]**2+x[2]-x[3]**2+x[3]-x[4]**2
    ce3 = 8*x[2]*(x[2]**2-x[1])-2*(1-x[2])+4*(x[2]-x[3])**2+x[1]**2-x[0]+x[3]-x[4]**2+x[0]**2+x[4]-x[5]**2
    ce4 = 8*x[3]*(x[3]**2-x[2])-2*(1-x[3])+4*(x[3]-x[4])**2+x[2]**2-x[1]+x[4]-x[5]**2+x[1]**2+x[5]-x[0]
    ci1 = 8*x[4]*(x[4]**2-x[3])-2*(1-x[4])+4*(x[4]-x[5])**2+x[3]**2-x[2]+x[5]+x[2]**2-x[1]
    ci2 = -(8*x[5] * (x[5]**2-x[4])-2*(1-x[5]) +x[4]**2-x[3]+x[3]**2 - x[4])

    return [obj, ce1,ce2,ce3,ce4,ci1,ci2]

def get_bounds() -> tuple[np.ndarray, np.ndarray]:
    return ([-5]*6,[5]*6)

def gradient(x: np.ndarray[float]) -> np.ndarray[float]:
    return pg.estimate_gradient_h(lambda x: fitness(x), x)

@dataclass
class my_constrained_udp:

    def fitness(self, x: np.ndarray | int) -> np.ndarray[float] | list[float]:
        """Fitness method

        Args:
            x (np.ndarray): Decision vector or chromosome, and is made of real numbers and integers

        Returns:
            np.ndarray: fitness of the input decision vector (concatenating the objectives, the equality and the inequality constraints)
        """
        return fitness(x)        


    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Method that returns the box-bounds of the problem

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower bounds, upper bounds)
        """
        return get_bounds()

    def get_nic(self) -> int:
        """Method that returns the number of inequality constraints of the problem
        
        Returns:
            int: number of inequality constraints
        """
        return 2

    def get_nec(self) -> int:
        """Method that returns the number of equality constraints of the problem
        
        Returns:
            int: number of equality constraints
        """
        return 4

    def gradient(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return gradient(x)
    
    def get_name(self) -> str:
        """ Problem’s name """
        return "NLP Constrained UDP"
    
@dataclass
class my_minlp:
    def fitness(self, x: np.ndarray | int) -> np.ndarray[float] | list[float]:
        return fitness(x)
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Method that returns the box-bounds of the problem

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower bounds, upper bounds)
        """
        return ([-5]*6,[5]*6)
    
    def get_nec(self) -> int:
        return 0
    
    def get_nic(self) -> int:
        return 6
    
    def get_nix(self):
        """ Number of integer decision variables """
        return 2

    def get_name(self) -> str:
        """ Problem’s name """
        return "MINLP"
    
    
@timer_decorator
def evaluate(problem: pg.problem, algo: pg.algorithm, pop: pg.population = None)-> pg.population:
    if pop is None:
        pop = pg.population(problem,10)
    
    pop = algo.evolve(pop)
    
    return pop
    
    
if __name__ == "__main__":
    
    # Example 1. Simple user-defined problem
    # prob = pg.problem(sphere_function())
    # print(prob)
    # algo = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))
    # pop = evaluate(prob, algo)
    # print(pop.champion_f)
    
    # Example 2. Constrained user-defined problem
    prob = pg.problem(my_constrained_udp())
    print(prob)
    
    # Method A: Using the augmented Lagrangian method
    algo = pg.algorithm(uda = pg.nlopt('auglag'))
    algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
    algo.set_verbosity(200) # in this case this correspond to logs each 200 objevals
    pop = pg.population(prob = my_constrained_udp(), size = 1)
    pop.problem.c_tol = [1E-6] * 6
    pop = algo.evolve(pop) 
    print(f"Fitness evaluations: {pop.problem.get_fevals()}") 
    print(f"Gradient evaluations: {pop.problem.get_gevals()}") 
    print(f"{pop.champion_f=}")
    
    # Method B: Using 
    #METHOD B
    algo = pg.algorithm(uda = pg.mbh(pg.nlopt("slsqp"), stop = 20, perturb = .2))
    algo.set_verbosity(1) # in this case this correspond to logs each 1 call to slsqp
    pop = pg.population(prob = my_constrained_udp(), size = 1)
    pop.problem.c_tol = [1E-6] * 6
    pop = algo.evolve(pop) 
    print(f"Fitness evaluations: {pop.problem.get_fevals()}") 
    print(f"Gradient evaluations: {pop.problem.get_gevals()}")
    print(f"{pop.champion_f=}")
    
    # Example 3. Constrained MINLP
    prob = pg.problem(my_minlp())
    print(prob)
    prob.c_tol = [1e-8]*6
    algo = pg.ipopt()
    
    # We run 20 instances of the optimization in parallel via a default archipelago setup
    archi: pg.archipelago = pg.archipelago(n = 20, algo = algo, prob = prob, pop_size=1)
    archi.evolve(2)
    archi.wait()

    # We get the best of the parallel runs
    a = archi.get_champions_f()
    a2 = sorted(archi.get_champions_f(), key = lambda x: x[0])[0]
    best_isl_idx = [(el == a2).all() for el in a].index(True)
    x_best = archi.get_champions_x()[best_isl_idx]
    f_best = archi.get_champions_f()[best_isl_idx]

    print("Best relaxed solution, x: {}".format(x_best)) 
    print("Best relaxed solution, f: {}".format(f_best)) 