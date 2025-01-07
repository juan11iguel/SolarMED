# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy==2.2.1", 
#     "pygmo>=2.19.5",
# ]
# ///

# "numpy==2.2.1" "1.26.4"

import pygmo as pg
import numpy as np

np.set_printoptions(precision=2)

class sphere_function:
    x_history: list = None
    fitness_history: list = None
    
    def __init__(self):
        self.x_history = []
        self.fitness_history = []
    
    def fitness(self, x):
        cost = sum(x * x)
        # print(f"FITNESS EVAL: x = {x}, cost = {cost}")
        
        self.x_history.append(x)
        self.fitness_history.append(cost)
        
        return [cost]

    def get_bounds(self):
        return ([-1] * 3, [1] * 3)

prob = pg.problem(sphere_function())
pop = pg.population(prob, 30)
print(pop)

algo = pg.algorithm(pg.gaco(gen = 400, ker=30))
algo.evolve(pop)

print(f"x_history = {pop.problem.extract(object).x_history}")
print(f"fitness history = {pop.problem.extract(object).fitness_history}")


