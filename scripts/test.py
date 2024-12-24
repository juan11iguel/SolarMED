# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy==2.2.1", 
#     "pygmo>=2.19.5",
# ]
# ///

# "numpy==2.2.1" "1.26.4"

import pygmo as pg

class sphere_function:
    def fitness(self, x):
        cost = sum(x * x)
        print(f"FITNESS EVAL: x = {x}, cost = {cost}")
        return [cost]

    def get_bounds(self):
        return ([-1] * 3, [1] * 3)

prob = pg.problem(sphere_function())
pop = pg.population(prob, 5)
print(pop)