from dataclasses import dataclass
import math
import numpy as np
import pygmo as pg

pop_size: int = 10
def fitness(x: np.ndarray[float] | np.ndarray[int]) -> list[float]:
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
class my_minlp:
    def fitness(self, x: np.ndarray[float] | np.ndarray[int]) -> list[float]:
        print(f"fitness was called! {x=}")
        
        return fitness(x)
    
    def get_bounds(self, readable_format: bool = False, debug: bool = False) -> np.ndarray[float | int] | tuple[np.ndarray[float | int], np.ndarray[float | int]]:
        """Method that returns the box-bounds of the problem

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower bounds, upper bounds)
        """
        print("get_bounds was called!")
        
        return [-5]*6, [5]*6
    
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
    
prob = pg.problem(my_minlp())
print(prob)

algo = pg.algorithm(pg.gaco(gen=10, ker=pop_size))
print(f"Running {algo.get_name()}")
algo.set_verbosity(1) # regulates both screen and log verbosity

# Initialize population and evolve population
pop = pg.population(prob, size=pop_size)
print(f"Initial population: {pop}\nStarting evolution...")
pop = algo.evolve(pop)

# Extract results of evolution
uda=algo.extract(pg.gaco)
print(f"Completed evolution, best fitness: {pop.champion_f[0]}, \nbest decision vector: {pop.champion_x}")

# print(uda.get_log())
for iter_log in uda.get_log():
    print(iter_log)
