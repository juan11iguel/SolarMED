"""
Simulate one episode with multiple days with the following procedure:

current_day = 0
WHILE (n_days_in_episode-current_day) >= 1:
1. Define paths (data, outputs)
3. Define episode (`date_str`, etc)
4. Define parameters (algorithm, problem, initial states)
5. Get problem data (problem samples, initialize model instance, read environment data, etc)

7. Evaluate Operation Plan - Startup
    1. Generate startup candidates
	1. Initialize candidate problems
	2. Pack candidate problems in a `pygmo.archipielago` and evaluate them
	3. Repeat for each considered forecast scenario
	4. Choose the best average candidate -> (operation_start, operation_end0)
	5. If computation takes more than `op_plan_computation_time` x 3 raise a warning
    6. Generate results (OptimizationResults, visualizations, etc) and save them
9. Evaluate operation optimization every X steps from operation start to operation end earliest candidate - op_optim_computation_time
    1. Evaluate Operation optimization
    2. Simulate until next decision variables update
10. Evaluate Operation Plan - Shutdown // Evaluate operation optimization
	1. Generate candidate problems
	2. Pack candidate problems in a `pygmo.archipielago` and evaluate them -> operation end
    If it takes longer than `op_optim_computation_time` raise a warning 
    3. In parallel (//) evaluate operation optimization
11. Move to the next day
END
"""
