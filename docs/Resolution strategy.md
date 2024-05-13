

# Pendiente / ideas

- En cada iteración, aunque solo se usen los resultados inmediatamente próximos para evaluar la siguiente iteración, habría que guardar todos los datos para después poder usarlos en la visualización (ver datos pasados, la decisión siguiente, pero también en gris ver los futuros valores y predicción). Hacer una animación estilo do-mpc?

# Problem definition and characterization


%% Diagrama para visualizar árbol, no hacer como un árbol realmente %%
![](attachments/solarMED_optimization-Decision%20tree.svg)

Esto quiere decir que el algoritmo de optimización deberá proveer una matriz con tantas filas como variables de decisión, y tantas columnas como números de muestreos en el horizonte de control.


**WIP visualización de evolución de modos de operación**

![](attachments/newplot.png)

Update:
![](attachments/newplot%201.png)

# Problem resolution alternatives


## Two sample rates

Using this strategy has one main advantage and one main drawback. As drawback it requires solving two optimization problems, the first one with the slower sample rate, and a second one with the faster one and considering only solar field and thermal storage. Compared to just one evaluation for the single (fast) sample rate alternative. 

However, the first evaluation needs to provide a much reduced solution space (less number of steps), and so does the second one (this time with less decision variables), while also being initialized with the solution of the previous evaluation, facilitating significantly the resolution of the second more complex problem. 

All in all, this strategy reduces significantly the complexity of the problem.

$n_{vars, all} \times t_{s,slow} \rightarrow n_{vars, SF-TS} \times t_{s,fast}$

## Complete problem or a number of predefined paths problems

One of the most important considerations when attempting to solve the optimization problem, is whether the paths to take (operating modes evolution in the prediction horizon) are pre-computed or if on the other hand, it's left to the optimization algorithm to explore the tree and generate its own trajectories.


### One complete problem

In this scenario, the optimization 

### N predefined path problems

Here the path or trajectory to follow is pre-defined, in order to avoid the tree explosion (exponential growth of options), a number of total options / branches / trajectories / paths is first defined. The idea is that starting from the complete tree of options, filter out branches applying different criteria and an iterative process until the set number of options is reached. Then each problem needs to be solved separately.

**Methodology**

1. Build the complete tree (by considering each possible transition, as defined by the FSM, at each step)
2. Apply [initial non-iterative filtering](#Initial%20non-iterative%20filtering)
3. Apply [iterative filtering](#Iterative%20filtering) until a set number of paths is reached. <- Does it really need to be iterative? Or could just the X most promising be selected directly?


#### Filtering options

A bibliography research should be carried out, but for now, just laying out some ideas:

##### Initial non-iterative filtering

- The most important variable (important as in being the one than most determines the operation possibilities of the system) for this process is the solar irradiation. Periods where it does not vary significantly or has low values, the state changes could be restricted.
- If only one sample rate is considered (the faster one), then the MED state could be restricted to change in periods not multiple of its sample rate.

##### Iterative filtering

After the initial non-iterative filtering, a decision tree with some groups is obtained.

```pseudocode
# Some kind of recursion function
def build_tree():

	def add_states():
		for valid_state in new_states:
			new_states = add_states()
			return new_states

	n_paths = 0
	for step_idx in prediction_horizon]):
		new_states = FSM[step_idx].get_valid_transitions()
	
	
while n_paths >= max_n_trajectories:

	# Apply filtering methodology
	
```

- As a first step, the tree could be evaluated up to some point (half the three for example). A cost can then be obtained for the path, and based on this information, make a selection of the most promising paths. From there solve the reduced number of paths up to the prediction horizon, and once again, selected only the most promising.

- Another option would be to solve a reduced simplified tree, where the operating modes can only change at reduced rates (e.g. if the sample rate is one hour in a 24 h prediction horizon - 24 potential operating mode changes, the operating modes can only change every 4 h - 6 operating mode changes). Solved this decision tree, the operating modes are fixed for those solved points and a new iteration is performed where the gaps between them are then solved, an so on.

## Summary

Depending on the considerations mentioned above, different computational and problems structures are obtained. In particular, the number of decision variables to solve for:

|          Problem type / Subsystems considered | SF-TS | SF-TS + MED |
| --------------------------------------------: | :---: | :---------: |
|          Complete (includes binary variables) |   4   |     15      |
| N predefined paths (without binary variables) |   2   |      7      |

On the other hand, depending on the sample rate consideration, the number of steps:

|              | Sample rate (seg) | N of steps |
| :----------: | :---------------: | :--------: |
| $t_{s,fast}$ |        60         |    1440    |
| $t_{s,slow}$ |      30 · 60      |     48     |

The optimization algorithm will need to provide a matrix solution , and there will be as many degrees of freedom as elements in this matrix that the algorithm will need to solve for:

|                                                        Resolution strategy | Resulting problem<br>structure |
| -------------------------------------------------------------------------: | :----------------------------- |
|                                          complete SF-TS+MED ($t_{s,fast}$) | 15 x 1440                      |
|                                      n predefined SF-TS+MED ($t_{s,fast}$) | n(7 x 1440)                    |
|         complete SF-TS+MED ($t_{s,slow}$) -> complete SF-TS ($t_{s,fast}$) | 15 x 48 -> 4 x 1440            |
|     complete SF-TS+MED ($t_{s,slow}$) -> n predefined SF-TS ($t_{s,fast}$) | 15 x 48 -> n(2 x 1440)         |
|     n predefined SF-TS+MED ($t_{s,slow}$) -> complete SF-TS ($t_{s,fast}$) | n(7 x 48) -> 4 x 1440          |
| n predefined SF-TS+MED ($t_{s,slow}$) -> n predefined SF-TS ($t_{s,fast}$) | n(7 x 48) -> n(2 x 1440)       |

Auxiliary table

|                        | $t_{s,fast}$ | $t_{s,slow}$ |
| ---------------------: | :----------: | :----------: |
|         complete SF-TS |   4 x 1440   |    4 x 48    |
|     complete SF-TS+MED |  15 x 1440   |   15 x 48    |
|     n predefined SF-TS | n(2 x 1440)  |  n(2 x 48)   |
| n predefined SF-TS+MED | n(7 x 1440)  |  n(7 x 48)   |


%% Diagrama para visualizar las distintas alternativas de resolución propuestas %%
```mermaid
flowchart TD
    RS[Resolution strategy] --> SLOW((Ts,slow))
    RS --> FAST((Ts,fast))

    FAST -->|x1440| ApAs(∀paths, ∀subsystems)
    FAST -->|x1440| NpAs(Npaths, ∀subsystems)
    ApAs -->|15x1440| End((end))
    NpAs -->|n 7x1440| End((end))

    SLOW -->|x48| ApAs2(∀paths, ∀subsystems)
    SLOW -->|x48| NpAs2(Npaths, ∀subsystems)
    ApAs2 -->|15x48| FAST2((Ts,fast))
    NpAs2 -->|n 7x48| FAST3((Ts,fast))

    FAST2 -->|15x48 ⭢ x1440| ApSFTS(∀paths, SF-TS)
    FAST3 -->|n 7x48 ⭢ x1440| NpSFTS(Npaths, SF-TS)
    ApSFTS -->|15x48 ⭢ 4x1440| End((end))
    NpSFTS -->|n 7x48 ⭢ n 2x1440| End((end))
```


# Other considerations

Artificially modifying the problem with unnecessary restrictions or by modifying the problem itself should be avoided as much as possible. Two examples:

- Trying to set as objective maximizing the operation time. If for example the optimization algorithm in one iteration tries values that invalidate the system midway through the evaluation and does not recover, the model automatically is going to penalize this behavior with a worse cost function value. And so eventually the most fit individuals or solutions will be the ones that prolong the operation for **as long as it makes *economical* sense** for the given horizon. 
- Restricting the state changes. The model itself will penalize constant state changes since it has associated transient non-productive states (*generating vacuum, starting-up, shutting-down*) between steady states (*off, idle, active*). This will result in a worse cost obtained at the end of the path evaluation.

## Variable sample rate

Create a function that takes a number of samples to assign, and distributes them based on some criteria:

```
samples_opt = samples_assigner(number_of_samples: int, ratio: int, method: Literal["by_distance", "by_variability"], estimator_values: array = None)
```

Where `samples_opt` will have the shape $[\cdot]_{1\times N_p}$ being $N_p$  the prediction horizon (number of samples from the evaluation of the model). Its elements will be one/True when the decision variables should be updated and 0/False when not.

Method:
- By distance, based on some function (linear, quadratic, exponential) spread the samples with an increasing distance between them as the distance from the origin increases.
- By variability of some reference variable. Using some process / environment variable or another estimator variable (`estimator_values`), and based on its variance, assign more samples where the variance is higher and less where not.

Note: the first element will always be one.


## Fitness function

In order to evaluate the model given any sample rate, variable or not a function like this can be used:

```
def model_interface(
	dec_vars: array[Lc x Nc], 
	env_vars: array[Lenv x Np], 
	samples_opt: array[1 x Np] | int, 
	costs_w: array[1 x Np] | float, 
	costs_e: array[1 x Np] | float
) -> acum_cost: float

	# Checks
	costs_w = costs_w if isarray(costs_w) else np.ones(1, Np)*costs_w
	costs_e = costs_e if isarray(costs_e) else np.ones(1, Np)*costs_e

	if isinstance(samples_opt, int):
		# Sample rate specified, build vector of samples where decision variables are updated
		samples_opt = np.arange(start=0, stop=env_vars.shape[0?], step=samples_opt)
	else:
		# Samples specified
		if np.sum(samples_opt) != Nc:
			raise ValueError(There should be as many True elements in samples_opt as number of samples to update the decision variables)

	# Initialization
	dec_vars_idx = -1
	current_dec_vars = None
	acum_cost = zeros(1, Nc)

	# Simulate
	for idx=0:1:Np
		if samples_opt[idx] == True:
			dec_vars_idx += 1
			current_dec_vars = dec_vars[dec_vars_idx]
			
		 model.step(
			 current_dec_vars,
			 env_vars[idx],
			 costs_w[idx],
			 costs_e[idx]
		 )
		 acum_cost[dec_vars_idx] += model.evaluate_cost()
	
	return np.sum(acum_cost)
```

Where Lx represents number of variables and Nx number of samples.