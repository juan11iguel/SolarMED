from dataclasses import fields, dataclass, is_dataclass
from typing import get_args, Optional
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from loguru import logger
import pygmo as pg
import tempfile
from pathlib import PurePosixPath, Path
import gzip
import shutil

from solarmed_modeling.solar_med import SolarMED
from solarmed_modeling.fsms import SolarMedState

from solarmed_optimization.problems import BaseNlpProblem
from solarmed_optimization import (
    RealDecVarsBoxBounds, 
	DecisionVariablesUpdates,
	EnvironmentVariables,
	InitialDecVarsValues,
	DecisionVariables,
	IntegerDecisionVariables,
	RealDecisionVariablesUpdatePeriod,
	RealDecisionVariablesUpdateTimes,
	RealLogicalDecVarDependence,
	ProblemParameters,
	ProblemsEvaluationParameters,
	AlgoParams,
	OpPlanActionType
)

from solarmed_optimization.utils import (resolve_dataclass_type, 
                                         condition_result_dataframe,
                                         downsample_by_segments)
from solarmed_optimization.utils.evaluation import (evaluate_model_multi_day,
													evaluate_optimization_nlp)

def get_scenario_id(scenario_idx: int) -> str:
	return f"scenario_{scenario_idx:02d}"

@dataclass
class OperationPlanResults:
	date_str: str # Date in YYYYMMDD format
	action: OpPlanActionType # Operation plan action identifier
	x: pd.DataFrame # Decision vector (columns) for each problem (rows)
	# int_dec_vars: list[pd.DataFrame] # Integer decision variables for each problem
	fitness: pd.Series # Fitness values for each problem
	fitness_history: pd.DataFrame # Optimization algorithm evolution fitness history (rows) for each problem (columns)
	# environment_df: pd.DataFrame # Environment data
	scenario_idx: int = 0 # Uncertainty scenario index, zero by default
	metadata_flds: tuple[str, ...] = ("date_str", "action", "scenario_idx", "evaluation_time", "best_problem_idx", "algo_params", "problems_eval_params", "problem_params")
	best_problem_idx: Optional[int] = None # Index of the best performing problem
	results_df: Optional[pd.DataFrame] = None # Simulation timeseries results for the best performing problem
	evaluation_time: Optional[float] = None # Time, in seconds, taken to evaluate layer
	algo_params: Optional[AlgoParams] = None # Algorithm parameters
	problems_eval_params: Optional[ProblemsEvaluationParameters] = None
	problem_params: Optional[ProblemParameters] = None # Problem parameters
	env_vars: Optional[EnvironmentVariables] = None # Environment variables
	
	def __post_init__(self):
		if self.best_problem_idx is None:
			self.best_problem_idx = int(self.fitness.idxmin())
		self.set_environment_variables()
   
	def set_environment_variables(self, env_vars: Optional[EnvironmentVariables] = None) -> None:
		if self.env_vars is not None:
			return
		
		# Else
		if env_vars is not None:
			self.env_vars = env_vars
			
		elif self.env_vars is None and self.results_df is not None:
			self.env_vars = EnvironmentVariables.from_dataframe(self.results_df)
		
	def evaluate_best_problem(self, problems: list[BaseNlpProblem] | BaseNlpProblem, model: SolarMED) -> pd.DataFrame:
		print("best problem idx", self.best_problem_idx, "best x:", self.x.iloc[self.best_problem_idx].values)
		results_df = evaluate_optimization_nlp(
			x=self.x.iloc[self.best_problem_idx].values, 
			problem=problems[self.best_problem_idx] if isinstance(problems, list) else problems,
			model=SolarMED(**model.dump_instance())
		)
  		# Remove NaNs, add columns for decision variables
		self.results_df = condition_result_dataframe(results_df)
		self.set_environment_variables()
   
		return self.results_df
	
	def get_hdf_base_path(self, date_str: str = None, action: OpPlanActionType = None, scenario_idx: int = None) -> str:
		date_str = self.date_str if date_str is None else date_str
		action = self.action if action is None else action
		scenario_idx = self.scenario_idx if scenario_idx is None else scenario_idx
		
		return f'/{date_str}/{action}/{get_scenario_id(scenario_idx)}'
	
	def export(self, output_path: Path, compress: bool = True, reduced: bool = False) -> None:
		""" Export results to a file. """
		if not output_path.exists():
			output_path.parent.mkdir(parents=True, exist_ok=True)
		
		temp_path = output_path.with_suffix(".h5")
		if output_path.with_suffix(".gz").exists():
			# Uncompress the gzip file into a temporary .h5 file
			with gzip.open(output_path.with_suffix(".gz"), 'rb') as f_in:
				with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f_out:
					shutil.copyfileobj(f_in, f_out)
					temp_path = Path(f_out.name)

		# if reduced:
		#     self = copy.deepcopy(self)  # To avoid modifying the original object
		#     self.results_df = None

		with pd.HDFStore(temp_path, mode='a') as store:
			path_str = self.get_hdf_base_path()

			# Add the results dataframes to the file
			store.put(f'{path_str}/x', self.x)
			store.put(f'{path_str}/fitness', self.fitness)
			store.put(f'{path_str}/fitness_history', self.fitness_history)
			store.put(f'{path_str}/results', self.results_df)

			# Add metadata attributes to the file
			storer = store.get_storer(f"{path_str}/results")
			storer.attrs.description = (
				f"Evaluation results for the SolarMED optimal coupling. "
				f"Operation plan layer - {self.action} for day {self.date_str}"
			)
			for fld_id in self.metadata_flds:
				fld_val = getattr(self, fld_id)
				if is_dataclass(fld_val):
					fld_val = asdict(fld_val)
				setattr(storer.attrs, fld_id, fld_val)

		# Compress the .h5 file using gzip
		if compress:
			suffix = ".gz"
			with open(temp_path, 'rb') as f_in, gzip.open(output_path.with_suffix(suffix), 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
			temp_path.unlink()  # Remove temporary file .h5 file
		else:
			suffix = ".h5"
			# Copy the uncompressed file to the output path
			if temp_path != output_path.with_suffix(suffix):
				shutil.copy(temp_path, output_path.with_suffix(suffix))
				temp_path.unlink()  # Remove temporary file .h5 file
		
		logger.info(f"Exported results to {output_path.with_suffix(suffix)} / {path_str}")

	@classmethod
	def initialize(cls, input_path: Path, date_str: str, action: OpPlanActionType, scenario_idx: int = 0, log: bool = True) -> "OperationPlanResults":
		""" Initialize an OperationPlanResults object from a file. """
		
		if not isinstance(input_path, Path):
			input_path = Path(input_path)
  
		if input_path.suffix == ".gz":
			# Uncompress the gzip file into a temporary .h5 file
			with gzip.open(input_path, 'rb') as f_in:
				with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f_out:
					shutil.copyfileobj(f_in, f_out)
					temp_path = Path(f_out.name)
		else:
			temp_path = input_path

		try:
			base_path_str = cls.get_hdf_base_path(None, date_str=date_str, action=action, scenario_idx=scenario_idx)
		
			with pd.HDFStore(temp_path, mode='r') as store:
				# Load dataframes
				try:
					results_df = store.get(f'{base_path_str}/results')
				except KeyError:
					# results_df = None  # In case results_df was not saved (reduced export)
					all_keys = store.keys()
					# Find unique base paths (/date_str/action/scenario_idx)
					unique_base_paths = {
						str(PurePosixPath(k).parents[0]) for k in all_keys
					}
					unique_base_paths = sorted(unique_base_paths)
					
					raise KeyError(f"Could not find {base_path_str}. Available evaluation results {len(unique_base_paths)}: {unique_base_paths}")

					
				x = store.get(f'{base_path_str}/x')
				fitness = store.get(f'{base_path_str}/fitness')
				fitness_history = store.get(f'{base_path_str}/fitness_history')

				# Load metadata attributes
				storer = store.get_storer(f'{base_path_str}/results')
				# for attr_name in storer.attrs._v_attrnames:
				# 	value = getattr(storer.attrs, attr_name)
				# 	print(f"{attr_name} = {value}")

				metadata_flds_dict = {}
				for fld_id in cls.metadata_flds:
					# print(f"{fld_id=}")
					fld_def = cls.__dataclass_fields__[fld_id]
					fld_type = fld_def.type
					value = getattr(storer.attrs, fld_id, None)

					dataclass_type = resolve_dataclass_type(fld_type)
					if dataclass_type and value is not None:
						value = dataclass_type(**value)

					metadata_flds_dict[fld_id] = value
				# action = getattr(storer.attrs, 'action', None)
				# evaluation_time = getattr(storer.attrs, 'evaluation_time', None)
				# best_problem_idx = getattr(storer.attrs, 'best_problem_idx', None)

		finally:
			if input_path.suffix == ".gz":
				temp_path.unlink()  # Clean up temp .h5 file

		# Create the OperationPlanResults object
		op_plan_results = OperationPlanResults(
			x=x,
			fitness=fitness,
			fitness_history=fitness_history,
			results_df=results_df,

			**metadata_flds_dict
		)

		if log:
			logger.info(f"Initialized OperationPlanResults from {input_path} / {base_path_str}")

		return op_plan_results

def batch_export(output_path: Path, op_plan_results_list: list[OperationPlanResults], compress: bool = True) -> None:
	""" Export multiple OperationPlanResults objects to a single file. """

	if not isinstance(output_path, Path):
		output_path = Path(output_path)

	# First export to .h5
	temp_h5_path = output_path.with_suffix(".h5")

	for op_plan_results_ in op_plan_results_list:
		op_plan_results_.export(output_path=temp_h5_path, compress=False)

	# Compress once after all are written
	if compress:
		suffix = ".gz"
		with open(temp_h5_path, 'rb') as f_in, gzip.open(output_path.with_suffix(".gz"), 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)
		temp_h5_path.unlink()  # Remove uncompressed .h5 file
	else:
		suffix = ".h5"
		# Copy the uncompressed file to the output path
		# shutil.copy(temp_h5_path, output_path.with_suffix(suffix))
	
	logger.info(f"Exported {len(op_plan_results_list)} operation plan results to {output_path.with_suffix(suffix)}")

def generate_real_dec_vars_times(real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod, 
								 int_dec_vars_list: list[IntegerDecisionVariables]) -> list[RealDecisionVariablesUpdateTimes]:
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
			sample_time = pd.Timedelta(seconds=update_period)
			
			# Generate the index for the real dec var series
			unique_days = aux_int_dec_var_series.index.normalize().unique().day
			# Generate separate np.arange arrays for each day and concatenate
			daily_ranges = []

			for day in unique_days:
				# Get the range for the current day
				# day_start = day
				# day_end = day + pd.Timedelta(days=1)  # End of the day
				
				# Filter data for the current day
				# daily_data = aux_int_dec_var_series[
				# 	(aux_int_dec_var_series.index >= day_start) & 
				# 	(aux_int_dec_var_series.index < day_end)
				# ]
				# # Generate np.arange for the day's data 
				# From when the system is activated
				# Until it is deactivated
				# start_idx = np.argmax(daily_data != 0)
				# end_idx = len(daily_data) - np.argmax(daily_data[::-1] != 0) # - 1
				# print(f"{start_idx=}, {end_idx=}")
				# daily_range = np.arange(
				#     start=daily_data.index[ start_idx ],
				#     stop=daily_data.index[ end_idx if end_idx < len(daily_data)-1 else len(daily_data)-1 ],
				#     step=np.timedelta64(update_period, 's')
				# )
				start, end = int_dec_vars.get_start_and_end_datetimes(day=day, var_id=aux_int_dec_var_id)
				if start == end:
					continue
				elif start == getattr(int_dec_vars, aux_int_dec_var_id).index[0]:
					# Not an actual start of operation, but a continuation from a previous active state
					# An initial value should be provided so skip the initial sample
					start += sample_time
				# Generate the timezone-aware datetime index
				daily_ranges.append(pd.date_range(
					start=start,
					end=end,
					freq=sample_time,
					tz=timezone.utc
				))

			if len(daily_ranges) == 0:
				temp_dict[real_dec_var_id] = np.array([start])
			else:
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
	int_dec_vars: IntegerDecisionVariables
	dec_var_updates: DecisionVariablesUpdates # Decision variables updates
	size_dec_vector: int # Size of the decision vector
	real_dec_vars_box_bounds: RealDecVarsBoxBounds
	real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod
	real_dec_vars_times: RealDecisionVariablesUpdateTimes
	episode_range: tuple[datetime, datetime] # Episode range
	initial_state: SolarMedState # System initial state
	initial_values: InitialDecVarsValues # Initial values for the (real) decision variables
	model_dict: dict # SolarMED model dumped instance
	box_bounds_lower: list[np.ndarray[float]] # Lower bounds for the decision variables (in list of arrays format).
	box_bounds_upper: list[np.ndarray[float]] # Upper bounds for the decision variables (in list of arrays format).
	x_evaluated: list[list[float]] # Decision variables vector evaluated (i.e. sent to the fitness function)
	fitness_history: list[float] # Fitness record of decision variables sent to the fitness function
	operation_span: tuple[datetime, datetime] # Operation start and end datetimes for the first day
	daily_operation_spans: list[tuple[datetime, datetime]] # Operation start and end datetimes for each day
	
	def __init__(self, int_dec_vars: IntegerDecisionVariables, 
				 env_vars: EnvironmentVariables, 
				 real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod,
				 initial_values: InitialDecVarsValues,
				 model: SolarMED,
				 sample_time_ts: int,
				 store_x: bool = False,
				 store_fitness: bool = True,
				) -> None:

		# print("duplicates in int_dec_vars: ", int_dec_vars.med_mode.index[int_dec_vars.med_mode.index.duplicated()], int_dec_vars.sfts_mode.index[int_dec_vars.sfts_mode.index.duplicated()])
		int_dec_vars = IntegerDecisionVariables(**asdict(int_dec_vars)) # Avoid modifying the original instance

		self.sample_time_mod = model.sample_time
		self.sample_time_ts = sample_time_ts
		self.initial_values = initial_values
		self.env_vars = env_vars.resample(f"{self.sample_time_mod}s", origin="start")
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

		# Compute operation span
		self.operation_span = int_dec_vars.get_start_and_end_datetimes(self.episode_range[0].day)
		days = [self.episode_range[0].day + day_cnt for day_cnt in range((self.episode_range[1] - self.episode_range[0]).days+1)]
		self.daily_operation_spans = [int_dec_vars.get_start_and_end_datetimes(day) for day in days]
		# Setup integer decision variables
		# Off-loaded from here, we should receive already configured IntegerDecisionVariables,
		# including possible intitial values 
		# for int_dec_var_id, int_dec_var_values in asdict(int_dec_vars).items():
		# 	# from env_vars.index[0] to int_dec_var.index[0] -> some initial provided value
		# 	# print(f"{int_dec_var_id} \t| {self.episode_range[0]} | {int_dec_var_values.index[0]}")
		# 	if self.episode_range[0] < int_dec_var_values.index[0]:
		# 		int_dec_var_values = pd.concat([
		# 			pd.Series(index=[self.episode_range[0]], 
		# 					data=[getattr(initial_dec_vars_values, int_dec_var_id)]), 
		# 			int_dec_var_values
		# 		])
		# 	else:
		# 		raise ValueError(f"The first update shouldn't coincide with the operation start, only come some time after: {self.episode_range[0]=}, {int_dec_var_values.index[0]=}")
		# 	# From int_dec_var.index[-1] to env_vars.index[-1] the last value is kept
		# 	# Actually, we don't keep simulating after the last integer decision variable value is set to zero
   		# 	# setattr(
		# 	# 	int_dec_vars, 
		# 	# 	int_dec_var_id, 
		# 	# 	pd.concat([
		#    	# 		int_dec_var_values.iloc[-1],
		# 	# 		pd.Series(index=[int_dec_var_values.index[-1]], data=getattr(initial_dec_vars_values, int_dec_var_id)), 
		# 	#   	])
		# 	# )
		# 	setattr(
		# 		int_dec_vars, 
		# 		int_dec_var_id, 
		# 		int_dec_var_values.resample(f"{self.sample_time_mod}s").ffill()
		# 	)
		# Add a value for the end of the episode, needs to be done here since it's only here where we know the episode end
		int_dec_vars = IntegerDecisionVariables(**{
			name: pd.concat([value, pd.Series(index=[self.episode_range[1]], data=[0.0])]) 
			for name, value in int_dec_vars.to_dict().items()
			if value.index[-1] <= self.episode_range[1]
		})
		# Resample integer decision variables to model sample time. Do it here to avoid doing it every time in the fitness function
		# logger.info(int_dec_vars.sfts_mode.index[0:3])
		self.int_dec_vars = int_dec_vars.resample(self.sample_time_mod)
		# self.int_dec_vars = int_dec_vars
		# print("")

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
			
			if num_updates <= 1: # No estoy seguro de si debería ser solo cuando no haya ninguna actualización (num_updates == 0)
				initial_value = getattr(self.initial_values, var_id)
				data = initial_value.values if isinstance(initial_value, pd.Series) else initial_value
				index = np.array([self.episode_range[0]]) if not isinstance(initial_value, pd.Series) else initial_value.index
				
    			# Add an additional (null) value for the end of the episode
				data = np.append(data, 0.0) 
				index = np.append(index, self.episode_range[1])

				decision_dict[var_id] = pd.Series(data=data, index=index)
			else:
				# Initialize real decision variable array with provided data for active periods
				data = np.array(x[cnt:cnt+num_updates])
				index = np.array(dec_var_times)

				# Set initial value. From self.episode_range[0] to self.operation_span[0] -> some initial provided value
				data = np.insert(data, 0, getattr(self.initial_values, var_id))
				index = np.insert(index, 0, self.episode_range[0])
	
				# Set final value for operation_span[1] (should be done for each day in a multi-day scenario)
				insert_pos = np.searchsorted(index, self.operation_span[1])
				if insert_pos == len(index) or index[insert_pos] != self.operation_span[1]:
					data = np.insert(data, insert_pos, 0.0)
					index = np.insert(index, insert_pos, self.operation_span[1])
				else:
					# Last decision coincides with the operation span end, value is overriden
					data[insert_pos] = 0.0
				
				# Keep the last value until the end of the operation
				# UPDATE: fkn donkey, if there are no more updates is because the system is inactive, the value is zero		
				if index[-1] != self.daily_operation_spans[-1][1]:
					data = np.append(data, 0.0) 
					index = np.append(index, self.daily_operation_spans[-1][1])
				else:
					# Last decision coincides with the operation span end, value is overriden
					data[-1] = 0.0
     
				# Finally, add an additional value for the end of the episode
				if index[-1] != self.episode_range[1]:
					data = np.append(data, 0.0) 
					index = np.append(index, self.episode_range[1])
				else:
					# Last decision coincides with the operation span end, value is overriden
					data[-1] = 0.0
	
				decision_dict[var_id] = pd.Series(data=data, index=index)
				# print(f"{decision_dict[var_id]=}")

			# check for duplicates
			duplicated = decision_dict[var_id].index[ decision_dict[var_id].index.duplicated() ]
			if len(duplicated) > 0:
				logger.error(f"Here comes the error: {var_id} | {duplicated}")

			if resample:
				# new_index = pd.date_range(
				# 	start=decision_dict[var_id].index[0],
				# 	end=self.episode_range[1],
				# 	freq=f"{self.sample_time_mod}s",
				# 	tz=timezone.utc
				# )
				# # Reindex and forward fill
				# decision_dict[var_id] = decision_dict[var_id].reindex(new_index).ffill()
				decision_dict[var_id] = decision_dict[var_id].resample(f"{self.sample_time_mod}s", origin="start", ).ffill()
				# print(f"{var_id} | {decision_dict[var_id].index[-3:]}")
			cnt += num_updates
			
		int_dec_var_index = list(asdict(self.int_dec_vars).values())[0].index # precioso
		assert int_dec_var_index[0] == self.episode_range[0], \
			f"Initial integer decision variable time should be the same as the episode start: {int_dec_var_index[0]=}, {self.episode_range[0]=}"
		# assert int_dec_var_index[-1] >= self.episode_range[-1] - pd.Timedelta(seconds=self.sample_time_mod), \
		# 	f"Last integer decision variable time should span until the episode end: {int_dec_var_index[-1]=}, {self.episode_range[-1]=}"
   
		int_dec_vars = self.int_dec_vars if resample else self.int_dec_vars_pre_resample
   
		return DecisionVariables(
			**decision_dict,
			**asdict(int_dec_vars)
		)
  
	def decision_variables_to_decision_vector(self, dec_vars: DecisionVariables) -> np.ndarray[float]:
		""" Convert decision variables to decision vector """
		
		# 1. Extract active operation
		dec_vars_ = dec_vars.dump_in_span(self.daily_operation_spans[0], return_format="series")
		for span in self.daily_operation_spans[1:]:
			dec_vars_ = dec_vars_.append( dec_vars.dump_in_span(span, return_format="series") )

		# 2. Extract the values for each decision variable given their number of updates
		dec_vars_dict = dec_vars_.to_dict()
		dec_var_updates_dict = self.dec_var_updates.to_dict()

		x = np.concatenate([
			downsample_by_segments(
				dec_vars_dict[var_id], 
				dec_var_updates_dict[var_id], 
				dtype=float
			) 
			for var_id in self.dec_var_real_ids
		])
  
		return x
  
  
	def get_bounds(self, ) -> tuple[np.ndarray, np.ndarray]:
		return np.hstack(self.box_bounds_lower), np.hstack(self.box_bounds_upper)
		
	def fitness(self, x: np.ndarray[float], debug_mode: bool = False) -> list[float]:
			
		# Model initialization logic after external evaluation of thermal storage
		model: SolarMED = SolarMED(**self.model_dict)

		# Sanitize decision vector, sometimes float values are negative even though they are basically zero (float precision?)
		x[np.abs(x) < 1e-6] = 0

		# Convert complete decision (whole episode) vector to decision variables with model sample time
		try:
			dec_vars_episode: DecisionVariables = self.decision_vector_to_decision_variables(x=x)
		except ValueError as e: 
			# ValueError: cannot reindex on an axis with duplicate labels
			print(f"mal todo: med {self.int_dec_vars.med_mode.index} | sfts {self.int_dec_vars.sfts_mode.index}")
			raise e

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

def generate_bounds(problem_instance: Problem, int_dec_vars: IntegerDecisionVariables) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
		# print(f"{var_id}, {update_times_index=}")
		
		# print(int_var_values.index[int_var_values.index.duplicated()])
		if len(update_times_index) <= 1:
			# System is not activated throught horizon
			lb[var_idx] = np.array(0)
			ub[var_idx] = np.array(0)
			continue
		
		# Else
		lb_ = []
		ub_ = []
		for day in int_var_values.index.normalize().unique(): # For each day
			# Get the range for the current day
			if not np.any(update_times_index.normalize() == day):
				# Not a single update in the day
				continue

			op_start = update_times_index[update_times_index.normalize() == day][0]
			op_end = update_times_index[update_times_index.normalize() == day][-1]				
			# print(f"{op_start=}, {op_end=}")
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

