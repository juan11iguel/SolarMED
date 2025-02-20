from dataclasses import fields
from typing import get_args
from dataclasses import asdict
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger
import pygmo as pg

from solarmed_modeling.solar_med import SolarMED
from solarmed_modeling.fsms import SolarMedState

from solarmed_optimization.problems import BaseNlpProblem
from solarmed_optimization import (RealDecVarsBoxBounds, 
                                   DecisionVariablesUpdates,
                                   EnvironmentVariables,
                                   InitialDecVarsValues,
                                   DecisionVariables,
                                   IntegerDecisionVariables,
                                   RealDecisionVariablesUpdatePeriod,
                                   RealDecisionVariablesUpdateTimes,
                                   RealLogicalDecVarDependence)
from solarmed_optimization.utils.evaluation import evaluate_model_multi_day
# from solarmed_optimization.utils.evaluation import evaluate_idle_thermal_storage
# from solarmed_optimization.utils.operation_plan import get_start_and_end_datetimes

def generate_real_dec_vars_times(real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod, 
                                 int_dec_vars_list: list[IntegerDecisionVariables]) -> RealDecisionVariablesUpdateTimes:
	"""Generate the real decision variables times for each integer decision variables element in the list.

	Args:
		real_dec_vars_update_period (RealDecisionVariablesUpdatePeriod): Real decision variables update period
		int_dec_vars_list (list[IntegerDecisionVariables]): List of integer decision variables

	Returns:
		RealDecisionVariablesUpdateTimes: Real decision variables times
  
	Example:
		real_dec_vars_times = generate_real_dec_vars_times(
			real_dec_vars_update_period=RealDecisionVariablesUpdatePeriod(),
			int_dec_vars_list=int_dec_vars
		)

		print(f"{int_dec_vars[0].sfts_mode=}")
		print(f"{int_dec_vars[0].med_mode=}")
		print(f"{real_dec_vars_times[0].qsf=}")
		print(f"{real_dec_vars_times[0].qmed_s=}")
	"""
 
	real_dec_var_times = []
	for int_dec_vars in int_dec_vars_list: # For each operation plan
		temp_dict = {}
		for real_dec_var_id, update_period in asdict(real_dec_vars_update_period).items():
			# The integer decision vars are used to mark the start and end of the datetimes
			aux_int_dec_var_id = RealLogicalDecVarDependence[real_dec_var_id].value
			aux_int_dec_var_series = getattr(int_dec_vars, aux_int_dec_var_id)
			
			# Generate the index for the real dec var series
			unique_days = aux_int_dec_var_series.index.normalize().unique()
			# Generate separate np.arange arrays for each day and concatenate
			daily_ranges = []

			for day in unique_days:
				# Get the range for the current day
				day_start = day
				day_end = day + pd.Timedelta(days=1)  # End of the day
				
				# Filter data for the current day
				daily_data = aux_int_dec_var_series[
					(aux_int_dec_var_series.index >= day_start) & 
					(aux_int_dec_var_series.index < day_end)
				]
				# Generate np.arange for the day's data 
				# From when the system is activated
				# Until it is deactivated
				start_idx = np.argmax(daily_data != 0)
				end_idx = len(daily_data) - np.argmax(daily_data[::-1] != 0) # - 1
				# print(f"{start_idx=}, {end_idx=}")
				# daily_range = np.arange(
				#     start=daily_data.index[ start_idx ],
				#     stop=daily_data.index[ end_idx if end_idx < len(daily_data)-1 else len(daily_data)-1 ],
				#     step=np.timedelta64(update_period, 's')
				# )
				# Generate the timezone-aware datetime index
				daily_range = pd.date_range(
					start=daily_data.index[start_idx],
					end=daily_data.index[min(end_idx, len(daily_data)-1)],  # Ensure valid index
					freq=pd.Timedelta(seconds=update_period),
					tz=daily_data.index.tz
				)
				daily_ranges.append(daily_range)

			temp_dict[real_dec_var_id] = np.concatenate(daily_ranges)
			# temp_dict[real_dec_var_id] = np.arange(start=aux_int_dec_var_series.index[0], 
			#                                        stop=aux_int_dec_var_series.index[-1], 
			#                                        step=np.timedelta64(update_period, 's'))
		real_dec_var_times.append(RealDecisionVariablesUpdateTimes(**temp_dict))

	return real_dec_var_times


class Problem(BaseNlpProblem):
	sample_time_mod: int # Model sample time
	sample_time_ts: int  # Thermal storage sample time
	env_vars: EnvironmentVariables # Environment variables
	initial_values: InitialDecVarsValues
	int_dec_vars: IntegerDecisionVariables
	dec_var_updates: DecisionVariablesUpdates # Decision variables updates
	size_dec_vector: int # Size of the decision vector
	real_dec_vars_box_bounds: RealDecVarsBoxBounds
	real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod
	real_dec_vars_times: RealDecisionVariablesUpdateTimes
	episode_range: tuple[datetime, datetime] # Episode range
	initial_state: SolarMedState # System initial state
	model_dict: dict # SolarMED model dumped instance
	box_bounds_lower: list[np.ndarray[float]] # Lower bounds for the decision variables (in list of arrays format).
	box_bounds_upper: list[np.ndarray[float]] # Upper bounds for the decision variables (in list of arrays format).
	x_evaluated: list[list[float]] # Decision variables vector evaluated (i.e. sent to the fitness function)
	fitness_history: list[float] # Fitness record of decision variables sent to the fitness function
    
	def __init__(self, int_dec_vars: IntegerDecisionVariables, 
              	 initial_dec_vars_values: InitialDecVarsValues, 
                 env_vars: EnvironmentVariables, 
                 real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod,
                 model: SolarMED,
                 sample_time_ts: int,
                 store_x: bool = False,
                 store_fitness: bool = True,
                ) -> None:
     
		int_dec_vars = IntegerDecisionVariables(**asdict(int_dec_vars)) # Avoid modifying the original instance

		self.sample_time_mod = model.sample_time
		self.sample_time_ts = sample_time_ts
		self.env_vars = env_vars.resample(f"{self.sample_time_mod}s", origin="start")
		self.initial_values = initial_dec_vars_values
		self.real_dec_vars_update_period = real_dec_vars_update_period
		self.int_dec_vars_pre_resample= IntegerDecisionVariables(**asdict(int_dec_vars))
		self.store_x = store_x
		self.store_fitness = store_fitness

		self.episode_range = env_vars.I.index[0], env_vars.I.index[-1]
		self.dec_var_ids, self.dec_var_dtypes = zip(*[(field.name, get_args(field.type)[0]) for field in fields(DecisionVariables)])
		self.dec_var_int_ids: list[str] = [var_id for var_id, var_type in zip(self.dec_var_ids, self.dec_var_dtypes) if var_type in [bool, int]]
		self.dec_var_real_ids: list[str] = [var_id for var_id, var_type in zip(self.dec_var_ids, self.dec_var_dtypes) if var_type is float]
        
        # Store the model instance
		self.model_dict = model.dump_instance()
		# Add fields not included in the dump_instance method
		self.model_dict.update(dict(
			# Initial states
			## FSM states
			fsms_internal_states=model.fsms_internal_states,
				med_state=model.med_state,
				sf_ts_state=model.sf_ts_state,
		))
		self.initial_state: SolarMedState = model.get_state()
			
		# Initialize real decision variables times
		self.real_dec_vars_times = generate_real_dec_vars_times(
			real_dec_vars_update_period=real_dec_vars_update_period,
			int_dec_vars_list=[int_dec_vars]
		)[0]
		self.size_dec_vector = sum([len(times) for times in asdict(self.real_dec_vars_times).values()])
		self.dec_var_updates = DecisionVariablesUpdates(
			**{name: values.size for name, values in asdict(int_dec_vars).items()},
			**{name: len(values) for name, values in asdict(self.real_dec_vars_times).items()},
        )
  
  		# Get real variables limits
		self.real_dec_vars_box_bounds = RealDecVarsBoxBounds.initialize(
			fmp=model.fixed_model_params, 
			Tmed_c_in=env_vars.Tmed_c_in.mean()
		)
		self.box_bounds_lower, self.box_bounds_upper = generate_bounds(self, int_dec_vars)

		# Initialize decision vector history
		self.x_evaluated = []
		self.fitness_history = []

		# Setup integer decision variables
		for int_dec_var_id, int_dec_var_values in asdict(int_dec_vars).items():
			# from env_vars.index[0] to int_dec_var.index[0] -> some initial provided value
			int_dec_var_values = pd.concat([
				pd.Series(index=[self.episode_range[0]], 
						  data=[getattr(initial_dec_vars_values, int_dec_var_id)]), 
				int_dec_var_values
			])
			# From int_dec_var.index[-1] to env_vars.index[-1] the last value is kept
			# Actually, we don't keep simulating after the last integer decision variable value is set to zero
   			# setattr(
			# 	int_dec_vars, 
			# 	int_dec_var_id, 
			# 	pd.concat([
           	# 		int_dec_var_values.iloc[-1],
        	# 		pd.Series(index=[int_dec_var_values.index[-1]], data=getattr(initial_dec_vars_values, int_dec_var_id)), 
            #   	])
			# )
			# Resample integer decision variables to model sample time. Do it here to avoid doing it every time in the fitness function
			setattr(
				int_dec_vars, 
				int_dec_var_id, 
				int_dec_var_values.resample(f"{self.sample_time_mod}s").ffill()
			)
		self.int_dec_vars = int_dec_vars

	def __post_init__(self, ) -> None:
		logger.info(f"""{self.get_name()} initialized.
					{self.get_extra_info()}""")
  
		# 4 MED real decision varibales
		if self.initial_state == SolarMedState.sf_IDLE_ts_IDLE_med_OFF:
			logger.warning(
				f"Some updates in the decision vector are unnecessary since once the \
				MED is started, it takes some time (mainly to generate vacuum). In practice \
				this is limited to {self.model_dict['fsms_params']['med']['vacuum_duration_time']//3600 * 4 }\
        		out of {self.size_dec_vector} updates."
			)

	def get_name(self) -> str:
		""" Get problem’s name """
		return "SolarMED NLP problem"

	def get_extra_info(self) -> str:
		""" Get problem’s extra info. """
		
		return f"""
	-\t Size of decision vector: {self.size_dec_vector} elements
	-\t Decision variable ids: {self.dec_var_ids}
	-\t Decision variable types: {self.dec_var_dtypes}
	-\t Model sample time: {self.sample_time_mod} seconds
	-\t System initial state: {self.initial_state.name}
	-\t Lower bounds: {self.box_bounds_lower}
	-\t Upper bounds: {self.box_bounds_upper}"""
 
	def get_nic(self) -> int:
		""" Get number of inequality constraints """
		return 0

	def get_nix(self) -> int:
		""" Get integer dimension """
		return 0

	def decision_vector_to_decision_variables(self, x: np.ndarray[float], resample: bool = True) -> DecisionVariables:
		""" Convert decision vector to decision variables """
  
		decision_dict: dict[str, np.ndarray] = {}
		cnt = 0
		for var_id in self.dec_var_real_ids:
			num_updates = getattr(self.dec_var_updates, var_id)
			dec_var_times = getattr(self.real_dec_vars_times, var_id)
			
			# from env_vars.index[0] to real_dec_var_time.index[0] -> some initial provided value
			data = np.insert(x[cnt:cnt+num_updates], 0, getattr(self.initial_values, var_id))
			data = np.append(data, data[-1]) # Keep the last value until the end of the episode
			
			index = np.insert(dec_var_times, 0, self.episode_range[0])
			index = np.append(index, self.episode_range[-1])
			decision_dict[var_id] = pd.Series(data=data, index=index)

			if resample:
				decision_dict[var_id] = decision_dict[var_id].resample(f"{self.sample_time_mod}s", origin="start").ffill()
    
			cnt += num_updates
			
		int_dec_var_index = list(asdict(self.int_dec_vars).values())[0].index # precioso
		assert int_dec_var_index[0] == self.episode_range[0], \
			f"Initial integer decision variable time should be the same as the episode start: {int_dec_var_index[0]=}, {self.episode_range[0]=}"
		# assert int_dec_var_index[-1] >= self.episode_range[-1] - pd.Timedelta(seconds=self.sample_time_mod), \
		# 	f"Last integer decision variable time should span until the episode end: {int_dec_var_index[-1]=}, {self.episode_range[-1]=}"
   
	
   
		return DecisionVariables(
			**decision_dict,
			**asdict(self.int_dec_vars)
		)
  
	def get_bounds(self, ) -> tuple[np.ndarray, np.ndarray]:
		return np.concatenate(self.box_bounds_lower), np.concatenate(self.box_bounds_upper)
		
	def fitness(self, x: np.ndarray[float], debug_mode: bool = False) -> list[float]:
            
		# Model initialization logic after external evaluation of thermal storage
		model: SolarMED = SolarMED(**self.model_dict)

		# Sanitize decision vector, sometimes float values are negative even though they are basically zero (float precision?)
		x[np.abs(x) < 1e-6] = 0

		# Convert complete decision (whole episode) vector to decision variables with model sample time
		dec_vars_episode: DecisionVariables = self.decision_vector_to_decision_variables(x=x)

		fitness_total = evaluate_model_multi_day(
			model=model,
			dec_vars=dec_vars_episode,
			env_vars=self.env_vars,
			evaluation_range=self.episode_range,
			mode="optimization",
			dec_var_int_ids=self.dec_var_int_ids,
			sample_time_ts=self.sample_time_ts,
   			debug_mode=debug_mode
		)
    
		# Store decision vector and fitness value
		if self.store_x:
			self.x_evaluated.append(x.tolist())
		if self.store_fitness:
			self.fitness_history.append(fitness_total)
  
		if debug_mode:
			print(f"Evaluation {len(self.fitness_history)} | {fitness_total=}, qsf={np.unique(self.decision_vector_to_decision_variables(x=x, resample=False).qsf)}")
  
		return [fitness_total]

	def gradient(self, x) -> np.ndarray:
		return pg.estimate_gradient(lambda x: self.fitness(x), x)


def generate_bounds(problem_instance: Problem, int_dec_vars: IntegerDecisionVariables) -> tuple[np.ndarray, np.ndarray]:
		# Lower box-bounds
		lb: list[np.ndarray[float]] = [
			np.full((len(timestamps), ), np.nan, dtype=float) for timestamps in asdict(problem_instance.real_dec_vars_times).values()
		]
		# Upper box-bounds
		ub: list[np.ndarray[float]] = lb.copy()
		
		for var_idx, var_id in enumerate(problem_instance.dec_var_real_ids):
			lower_limit = getattr(problem_instance.real_dec_vars_box_bounds, var_id)[0],
			upper_limit = getattr(problem_instance.real_dec_vars_box_bounds, var_id)[1],
			update_period = getattr(problem_instance.real_dec_vars_update_period, var_id) # seconds
			update_times_index = pd.DatetimeIndex(getattr(problem_instance.real_dec_vars_times, var_id))
			
			# Get integer decision variable values
			int_var_id = RealLogicalDecVarDependence[var_id].value
   			# Either zeros or ones, otherwise some logic should be included to translate the values
			int_var_values: pd.Series = getattr(int_dec_vars, int_var_id)
   
			lb_ = []
			ub_ = []
			for day in int_var_values.index.normalize().unique(): # For each day
				# Get the range for the current day
				op_start = update_times_index[update_times_index.normalize() == day][0]
				op_end = update_times_index[update_times_index.normalize() == day][-1]				
				
    			# Resample integer decision variable to real decision variable sample time
				int_var_day_data_resampled = (
					int_var_values
					.resample(f"{update_period}s", origin=op_start)
					.ffill()
					# .bfill()
				)
				# Filter data for the current operation day
				aux_logical_var_values = int_var_day_data_resampled[
					(int_var_day_data_resampled.index >= op_start) & 
					(int_var_day_data_resampled.index <= op_end)
				].values
				lb_.extend(lower_limit * aux_logical_var_values)
				ub_.extend(upper_limit * aux_logical_var_values)

			lb[var_idx] = np.array(lb_)
			ub[var_idx] = np.array(ub_)
   
		return lb, ub