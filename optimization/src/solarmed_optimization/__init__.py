import math
from typing import get_args, NamedTuple, Literal, Optional, Any, Type
from dataclasses import dataclass, fields, field, is_dataclass, asdict
import copy
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pygmo as pg
from loguru import logger

from solarmed_modeling.solar_med import (SolarMED, 
										 EnvironmentParameters,
										 ModelParameters,
										 FixedModelParameters,
										 FsmParameters,
										 FsmInternalState,
										 InitialStates)
from solarmed_modeling.fsms import MedState, SfTsState, MedVacuumState
from solarmed_modeling.fsms.med import (FsmParameters as MedFsmParams,
										FsmInputs as MedFsmInputs)
from solarmed_modeling.fsms.sfts import (FsmParameters as SftsFsmParams,
										 FsmInputs as SfTsFsmInputs)
# Example of OperationActionType:
# dict[subsystem_id, list[tuple[ActionType, n_updates]]]
# {'sfts': [('startup', 3), ('shutdown', 2), ('startup', 1), ('shutdown', 1)],
#  'med': [('startup', 3), ('shutdown', 2), ('startup', 1), ('shutdown', 1)]}
OperationActionType = dict[str, list[tuple[str, int]]]
# Example of ActionUpdatesType:
# dict[subsystem_id, list[tuple[ActionType, list[datetime]]]]
# {'sfts': [('startup', [datetime1, ..., datetimeN]]), ('shutdown', [datetime1, ..., datetimeN]), ('startup', [datetime1, ..., datetimeN]), ('shutdown', [datetime1, ..., datetimeN])],
#  'med': [('startup', [datetime1, ..., datetimeN]), ('shutdown', [datetime1, ..., datetimeN]), ('startup', [datetime1, ..., datetimeN]), ('shutdown', [datetime1, ..., datetimeN])]}
OperationUpdateDatetimesType = dict[str, list[tuple[str, list[datetime]]]]

OpPlanActionType = Literal["startup", "shutdown"]
PygmoArchipelagoTopologies = Literal["unconnected", "ring", "fully_connected"]
 
def prepend(obj_cls: Type[Any], obj: Any, prepend_object: Any) -> Any:
	""" Prepend the current decision variables with another set of decision variables.
		This instance will be prepended with the provided instance up until this instance first index."""
	
	output = {}
	for name, value in asdict(obj).items():
		if value is None:
			continue
		elif not isinstance(value, pd.Series):
			raise TypeError(f"All attributes must be pd.Series for datetime indexing. Got {type(value)} instead.")

		pval = getattr(prepend_object, name)
		pval = pval[pval.index < value.index[0]]

		combined = pd.concat([pval, value])

		# Preserve frequency if both parts have matching freq
		# if value.index.freq == pval.index.freq and value.index.freq is not None:
		# 	combined.index.freq = value.index.freq

		output[name] = combined

	return obj_cls(**output)

def append(obj_cls: Type[Any], obj: Any, append_object: Any) -> Any:
	""" Append the current decision variables with another set of decision variables.
		This instance will be appended with the provided instance starting from this instance last index."""
	
	output = {}
	for name, value in asdict(obj).items():
		if value is None:
			continue
		elif not isinstance(value, pd.Series):
			raise TypeError(f"All attributes must be pd.Series for datetime indexing. Got {type(value)} instead.")

		aval = getattr(append_object, name)
		aval = aval[aval.index > value.index[-1]]

		combined = pd.concat([value, aval])

		# Preserve frequency if both parts have matching freq
		# if value.index.freq == aval.index.freq and value.index.freq is not None:
		# 	combined.index.freq = value.index.freq

		output[name] = combined

	return obj_cls(**output)


def dump_in_span(
    vars_dict: dict,
    span: tuple[int, Optional[int]] | tuple[datetime, Optional[datetime]],
    return_format: Literal["values", "series"] = "values",
    align_first: bool = False,
    resampling_method: Optional[Literal["nearest", "ffill", "bfill"]] = None, # "interpolate"
) -> dict:
    """
    Dump variables within a given span, optionally aligning the first value and reindexing.

    Args:
        vars_dict: A dictionary containing the variables to dump.
        span: A tuple representing the range (indices or datetimes).
        return_format: Format of the returned values ("values" or "series").
        align_first: If True, aligns the first value exactly to span[0] using interpolation or nearest method.
        resampling_method: If provided, reindexes using original frequency after alignment. Options: "interpolate", "nearest", etc.

    Returns:
        A new dictionary containing the filtered data.
    """
    if isinstance(span[0], datetime):
        dt_start, dt_end = span
        dt_end = dt_end if dt_end is not None else list(vars_dict.values())[0].index[-1]

        span_vars_dict = {}
        for name, series in vars_dict.items():
            if series is None:
                continue
            if not isinstance(series, pd.Series):
                raise TypeError(f"All attributes must be pd.Series for datetime indexing: {name} is {type(series)}")

            sliced = series[(series.index >= dt_start) & (series.index <= dt_end)]

            if align_first and dt_start not in series.index:
                if series.index[0] < dt_start < series.index[-1]:
                    # if resampling_method == "interpolate":
                    #     aligned_val = series.reindex(series.index.union([dt_start])).sort_index().interpolate("time").loc[dt_start]
                    # else:
                    aligned_val = series.loc[:dt_start].iloc[-1] if resampling_method == "ffill" else \
                                series.loc[dt_start:].iloc[0] if resampling_method == "bfill" else \
                                series.reindex(series.index.union([dt_start])).sort_index().asof(dt_start) if resampling_method == "nearest" else \
                                None
                    if aligned_val is not None:
                        # Insert aligned first value
                        sliced = pd.concat([pd.Series([aligned_val], index=[dt_start]), sliced])
                        sliced = sliced[~sliced.index.duplicated(keep="first")].sort_index()

            # Optional reindexing with original frequency
            if resampling_method and series.index.freq is not None:
                try:
                    freq = series.index.freq
                    new_index = pd.date_range(start=dt_start, end=dt_end, freq=freq)
                    if resampling_method == "interpolate":
                        sliced = sliced.reindex(new_index).interpolate("time")
                    else:
                        sliced = sliced.reindex(new_index, method=resampling_method)
                except Exception as e:
                    raise RuntimeError(f"Failed to reindex {name}: {e}")

            span_vars_dict[name] = sliced

    else:
        # Numeric span
        idx_start, idx_end = span
        idx_end = -1 if idx_end is None else idx_end

        span_vars_dict = {
            name: value[idx_start:idx_end]
            if isinstance(value, (pd.Series, np.ndarray)) else value
            for name, value in vars_dict.items()
        }

    if return_format == "values":
        span_vars_dict = {
            name: value.values if isinstance(value, pd.Series) else value
            for name, value in span_vars_dict.items()
        }

    return span_vars_dict


class MedMode(Enum):
	""" Possible decisions for MED operation modes.
	Given this, the FSM inputs are deterministic """
	OFF = 0
	# IDLE = 1
	ACTIVE = 1
	
med_fsm_inputs_table: dict[tuple[MedMode, MedState], MedFsmInputs] = {
	# med_mode = OFF
	(MedMode.OFF, MedState.OFF):               MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
	(MedMode.OFF, MedState.GENERATING_VACUUM): MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
	(MedMode.OFF, MedState.IDLE):              MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
	(MedMode.OFF, MedState.STARTING_UP):       MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
	(MedMode.OFF, MedState.SHUTTING_DOWN):     MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
	(MedMode.OFF, MedState.ACTIVE):            MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
	
	# med_mode = IDLE
	# (MedMode.IDLE, MedState.OFF):               MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
	# (MedMode.IDLE, MedState.GENERATING_VACUUM): MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
	# (MedMode.IDLE, MedState.IDLE):              MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
	# (MedMode.IDLE, MedState.STARTING_UP):       MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
	# (MedMode.IDLE, MedState.SHUTTING_DOWN):     MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
	# (MedMode.IDLE, MedState.ACTIVE):            MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.LOW),
	
	# med_mode = ACTIVE
	(MedMode.ACTIVE, MedState.OFF):               MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
	(MedMode.ACTIVE, MedState.GENERATING_VACUUM): MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.HIGH),
	(MedMode.ACTIVE, MedState.IDLE):              MedFsmInputs(med_active=True,  med_vacuum_state=MedVacuumState.LOW),
	(MedMode.ACTIVE, MedState.STARTING_UP):       MedFsmInputs(med_active=True,  med_vacuum_state=MedVacuumState.LOW),
	(MedMode.ACTIVE, MedState.SHUTTING_DOWN):     MedFsmInputs(med_active=False, med_vacuum_state=MedVacuumState.OFF),
	(MedMode.ACTIVE, MedState.ACTIVE):            MedFsmInputs(med_active=True,  med_vacuum_state=MedVacuumState.LOW),
}

class SfTsMode(Enum):
	""" Possible decisions for Solar field and Thermal storage modes.
	Given this, the associated FSM inputs are deterministic """
	OFF = 0
	# SF_HEATING_UP = 1
	ACTIVE = 1
	
sfts_fsm_inputs_table: dict[tuple[SfTsMode, SfTsState], SfTsFsmInputs] = {
	# sfts_mode = OFF
	(SfTsMode.OFF, SfTsState.IDLE): SfTsFsmInputs(sf_active=False, ts_active=False),
	(SfTsMode.OFF, SfTsState.HEATING_UP_SF): SfTsFsmInputs(sf_active=False, ts_active=False),
	(SfTsMode.OFF, SfTsState.SF_HEATING_TS): SfTsFsmInputs(sf_active=False, ts_active=False),
	
	# sfts_mode = ACTIVE
	(SfTsMode.ACTIVE, SfTsState.IDLE): SfTsFsmInputs(sf_active=True, ts_active=True),
	(SfTsMode.ACTIVE, SfTsState.HEATING_UP_SF): SfTsFsmInputs(sf_active=True, ts_active=True),
	(SfTsMode.ACTIVE, SfTsState.SF_HEATING_TS): SfTsFsmInputs(sf_active=True, ts_active=True),
}

class SubsystemId(Enum):
	SFTS = "sfts"
	MED = "med"
	
class SubsystemDecVarId(Enum):
	SFTS = SfTsMode
	MED = MedMode
	
@dataclass
class IrradianceThresholds:
	""" Irradiance thresholds (W/m²)"""
	lower: float = 300.
	upper: float = 600.
 
	
@dataclass
class EnvironmentVariables:
	"""
	Simple class to make sure that the required environment variables are passed
	
	All the variables should be 1D arrays with as many elements as the horizon of the optimization problem
	"""
	# n_horizon: int
	
	Tmed_c_in: float | np.ndarray[float] | pd.Series  # Seawater temperature
	Tamb: float | np.ndarray[float] | pd.Series # Ambient temperature
	I: float | np.ndarray[float] | pd.Series# Solar radiation
	
	cost_w: float | np.ndarray[float] | pd.Series # Cost of water, €/m³ 
	cost_e: float | np.ndarray[float] | pd.Series # Cost of electricity, €/kWhe
	
	wmed_f: float | np.ndarray[float] | pd.Series = None # Seawater salinity
	
	# def __post_init__(self):
	#     # Validate that all the environment variables have the same length
	#     assert all(
	#         len(getattr(self, var_id)) == len(self.Tmed_c_in) 
	#         for var_id in ["Tamb", "I", "cost_w", "cost_e"]
	#     ), "All variables must have the same length (optim_window_size // sample_time_mod)"
	@classmethod
	def from_dataframe(cls, df: pd.DataFrame, cost_w: float = None, cost_e: float = None) -> "EnvironmentVariables":
		if "cost_w" not in df.columns:
			cost_w=pd.Series(
				data=np.ones((len(df.index),)) * cost_w,
				index=df.index,
			)
		else:
			cost_w = df["cost_w"]
		if "cost_e" not in df.columns:
			cost_e=pd.Series(
				data=np.ones((len(df.index),)) * cost_e,
				index=df.index,
			)
		else:
			cost_e = df["cost_e"]
			
		return cls(
			I=df["I"],
			Tamb=df["Tamb"],
			Tmed_c_in=df["Tmed_c_in"],
			cost_w=cost_w,
			cost_e=cost_e
		)
  
	def get_date_str(self) -> str:
		""" Returns date string in YYYYMMDD format for the first index entry """
		return list(asdict(self).values())[0].index[0].strftime("%Y%m%d")
  
	def copy(self) -> "EnvironmentVariables":
		""" Create a deep copy of this EnvironmentVariables" instance. """
		return copy.deepcopy(self)

	def dump_at_index(self, idx: int, return_dict: bool = False) -> "EnvironmentVariables":
		"""
		Dump instance at a given index.

		Parameters:
		- idx: Integer index to extract.

		Returns:
		- A dictionary.
		"""
		dump =  {name: np.asarray(value)[idx] for name, value in asdict(self).items() if value is not None}
		
		return dump if return_dict else EnvironmentVariables(**dump)
	
	def dump_in_span(self, span: tuple[int, Optional[int]] | tuple[datetime, Optional[datetime]], return_format: Literal["values", "series"] = "values", **kwargs) -> 'EnvironmentVariables':
		""" Dump environment variables within a given span """
		
		vars_dict = dump_in_span(vars_dict=asdict(self), span=span, return_format=return_format, **kwargs)
		return EnvironmentVariables(**vars_dict)
	
	def resample(self, *args, **kwargs) -> "EnvironmentVariables":
		""" Return a new resampled environment variables instance """
		
		output = {}
		for name, value in asdict(self).items():
			if value is None:
				continue
			elif not isinstance(value, pd.Series):
				raise TypeError(f"All attributes must be pd.Series for datetime indexing. Got {type(value)} instead.")
			
			target_freq = int(float(args[0][:-1]))
			try:
				current_freq = value.index.freq.n
			except AttributeError as e:
				logger.error(f"Series index has no valid freq for resampling: {value.index}")
				raise e
			
			value = value.resample(*args, **kwargs)
			if  target_freq > current_freq: # Downsample
				value = value.first()
			else: # Upsample
				value = value.interpolate()
			output[name] = value
			
		return EnvironmentVariables(**output)
	
	def to_dataframe(self) -> pd.DataFrame:
		"""
		Convert environment variables into a pandas DataFrame.
		"""
		data = {
			name: pd.Series(value) if not isinstance(value, pd.Series) else value
			for name, value in asdict(self).items()
		}
		return pd.DataFrame(data)

	def __len__(self) -> int | ValueError:
		""" Check if all values have equal length and return that length. Otherwise, raise a ValueError. """
		lengths = {len(v) for v in asdict(self).values() if v is not None}
		if len(lengths) > 1:
			raise ValueError("Length check is unsupported when attributes have different lengths.")
		return lengths.pop() if lengths else 0

	def prepend(self, dec_vars: "EnvironmentVariables") -> "EnvironmentVariables":
		""" Prepend the current decision variables with another set of decision variables.
  			This instance will be prepended with the provided instance up until this instance first index."""
			
		return prepend(EnvironmentVariables, self, dec_vars)
	
	def append(self, dec_vars: "EnvironmentVariables") -> "EnvironmentVariables":
		""" Append the current decision variables with another set of decision variables.
  			This instance will be appended with the provided instance starting from this instance last index"""
			
		return append(EnvironmentVariables, self, dec_vars)
	
@dataclass
class DecisionVariables:
	"""
	Simple class to make sure that the required decision variables are passed
	to the model instance with the correct type
	"""
	# Real
	qsf: float | np.ndarray[float] | pd.Series #  Solar field flow -> Actual optimization output will be the outlet temperature (`Tsf,out`) after evaluating the inverse solar field model.
	qts_src: float | np.ndarray[float] | pd.Series #  Thermal storage recharge flow.
	qmed_s: float | np.ndarray[float] | pd.Series #  MED heat source flow.
	qmed_f: float | np.ndarray[float] | pd.Series #  MED feed flow.
	Tmed_s_in: float | np.ndarray[float] | pd.Series #  MED heat source inlet temperature.
	Tmed_c_out: float | np.ndarray[float] | pd.Series #  MED condenser outlet temperature.
	# Logical / integers
	sfts_mode: int | np.ndarray[int] | pd.Series #  Solar field and thermal storage mode (off, active)
	# sf_active: bool | np.ndarray[bool]#  Solar field state (off, active)
	# ts_active: bool | np.ndarray[bool]#  Thermal storage state (off, active)
	# med_active: bool | np.ndarray[bool]#  MED heat source state (off, active)
	# med_vac_state: int | np.ndarray[int] | pd.Series #  MED vacuum system state (off, low, high)
	med_mode: int | np.ndarray[int] | pd.Series #  MED operation mode (off, active)
	
	# def __post_init__(self) -> None:
	#     # Ensure attributes are of correct type
	#     for fld in fields(self):
	#         value: np.ndarray | bool | float | int | pd.Series = getattr(self, fld.name)
	#         _type = get_args(fld.type)[0]
	#         if isinstance(value, np.ndarray):
	#             setattr(self, fld.name, value.astype(_type))
	#         elif isinstance(value, pd.Series):
	#             setattr(self, fld.name, value.values.astype(_type))
	#         else:
	#             setattr(self, fld.name, _type(value))
 
	@classmethod
	def from_dataframe(cls, df: pd.DataFrame, ) -> "DecisionVariables":
			
		return cls(**{fld.name: df[f"dec_var_{fld.name}"]  for fld in fields(cls) })

	def copy(self) -> "DecisionVariables":
		""" Create a deep copy of this DecisionVariables instance. """
		return copy.deepcopy(self)
	
	def dump_at_index(self, idx: int, return_dict: bool = False) -> "DecisionVariables":
		"""
		Dump instance at a given index.

		Parameters:
		- idx: Integer index to extract.

		Returns:
		- A dictionary.
		"""
		dump =  {name: np.asarray(value)[idx] for name, value in asdict(self).items() if value is not None}
		
		return dump if return_dict else DecisionVariables(**dump)
	
	def dump_in_span(self, span: tuple[int, Optional[int]] | tuple[datetime, Optional[datetime]], return_format: Literal["values", "series"] = "values", **kwargs) -> 'DecisionVariables':
		""" Dump decision variables within a given span """
		
		vars_dict = dump_in_span(vars_dict=asdict(self), span=span, return_format=return_format, **kwargs)
		return DecisionVariables(**vars_dict)
	
	def to_dict(self) -> dict:
		"""
		Convert decision variables into a dictionary.
		"""
		return asdict(self)
 
	def to_dataframe(self) -> pd.DataFrame:
		"""
		Convert decision variables into a pandas DataFrame.
		"""
		data = {
			name: pd.Series(value) if not isinstance(value, pd.Series) else value
			for name, value in asdict(self).items()
		}
		return pd.DataFrame(data)

	def add_dec_vars_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Add decision variables to a pandas DataFrame.
		"""
		dec_vars_df = self.to_dataframe().rename(columns={
			name: f"dec_var_{name}" for name in self.__dict__.keys()
		})
  
		return pd.concat([df, dec_vars_df], axis=1)

	def prepend(self, dec_vars: "DecisionVariables") -> "DecisionVariables":
		""" Prepend the current decision variables with another set of decision variables.
  			This instance will be prepended with the provided instance up until this instance first index."""
			
		return prepend(DecisionVariables, self, dec_vars)
	
	def append(self, dec_vars: "DecisionVariables") -> "DecisionVariables":
		""" Append the current decision variables with another set of decision variables.
  			This instance will be appended with the provided instance starting from this instance last index"""
			
		return append(DecisionVariables, self, dec_vars)

	# def __add__(self, other: "DecisionVariables") -> "DecisionVariables":
	# 	""" Concatenate two DecisionVariables instances. 
	# 		The object on the left will be prepended to the object on the right up 
   	# 		until the first index of it. """
      
	# 	if not isinstance(other, DecisionVariables):
	# 		raise TypeError(f"Cannot add DecisionVariables with {type(other)}")

	# 	output = {}
	# 	for name, value in asdict(self).items():
	# 		if value is None:
	# 			output[name] = getattr(other, name)
	# 		elif not isinstance(value, pd.Series):
	# 			raise TypeError(f"All attributes must be pd.Series for datetime indexing. Got {type(value)} instead.")
	# 		else:
	# 			other_value = getattr(other, name)
	# 			if other_value is not None:
	# 				other_value = other_value[other_value.index < value.index[0]]
	# 				value = pd.concat([other_value, value])
	# 			output[name] = value

	# 	return DecisionVariables(**output)

	def __eq__(self, other: "DecisionVariables") -> bool:
		""" Check equality of two DecisionVariables instances by comparing all attributes. """
		if not isinstance(other, DecisionVariables):
			return False

		for name, value in asdict(self).items():
			other_value = getattr(other, name)
			if isinstance(value, pd.Series) and isinstance(other_value, pd.Series):
				if not value.equals(other_value):
					return False
			elif value != other_value:
				return False

		return True

	def __len__(self) -> int | ValueError:
		""" Check if all values have equal length and return that length. Otherwise, raise a ValueError. """
		lengths = {len(v) for v in asdict(self).values()}
		if len(lengths) > 1:
			raise ValueError("Length check is unsupported when attributes have different lengths.")
		return lengths.pop() if lengths else 0
  
  				
def dump_at_index_dec_vars(dec_vars: DecisionVariables, idx: int, return_dict: bool = False) -> DecisionVariables | dict:
	""" Dump decision variables at a given index """
	
	import warnings
	
	warnings.warn("This function is deprecated. Use DecisionVariables.dump_at_index instead.", DeprecationWarning)
	
	# Precioso, equivale a: {field.name: field.type( field.value[idx] )}
	dump = {field.name: get_args(field.type)[0]( getattr(dec_vars, field.name)[idx] ) for field in fields(dec_vars)}
	
	if return_dict:
		return dump
	return DecisionVariables(**dump)
	
@dataclass
class InitialDecVarsValues:
	sfts_mode: int | pd.Series = 0
	med_mode: int | pd.Series = 0
	qsf: float | pd.Series = 0.0
	qts_src: float | pd.Series = 0.0
	qmed_s: float | pd.Series = 0.0
	qmed_f: float | pd.Series = 0.0
	Tmed_s_in: float | pd.Series = 0.0
	Tmed_c_out: float | pd.Series = 0.0
	
@dataclass
class IntegerDecisionVariables:
	sfts_mode: SfTsMode | pd.Series
	med_mode: MedMode | pd.Series
	
	@classmethod
	def from_dec_vars(cls, dec_vars: DecisionVariables) -> "IntegerDecisionVariables":
		""" Initialize integer decision variables from the DecisionVariables instance """
		return cls(
			sfts_mode=dec_vars.sfts_mode,
			med_mode=dec_vars.med_mode
		)
	
	def to_dataframe(self) -> pd.DataFrame:
		"""
		Convert integer decision variables into a pandas DataFrame.
		"""
		data = {
			name: pd.Series(value) if not isinstance(value, pd.Series) else value
			for name, value in asdict(self).items()
		}
		return pd.DataFrame(data)

	def copy(self) -> "IntegerDecisionVariables":
		""" Create a deep copy of this IntegerDecisionVariables" instance. """
		return copy.deepcopy(self)
	
	def get_start_and_end_datetimes(self, day: Optional[int] = None, var_id: Optional[Literal["sfts_mode", "med_mode"]] = None) -> tuple[datetime, datetime]:
		""" Get start and end datetimes of the decision variables """
		
		# To avoid circular imports
		from solarmed_optimization.utils import get_start_and_end_datetimes
		
		if day is None:
			day = self.sfts_mode.index[0].day
			
		var_values_list = asdict(self).values() if var_id is None else [asdict(self)[var_id]]
		# [
		# 	print(f"{var_values.index.day=} >= {day=}")
		# 	for var_values in var_values_list
		# ]
		daily_data = [
			var_values[(var_values.index.day >= day) & (var_values.index.day < day+1)]
			for var_values in var_values_list
		]
			
		return get_start_and_end_datetimes(series=daily_data)
	
	def get_total_active_duration(self, ) -> timedelta:
		days = np.unique( list(asdict(self).values())[0].index.day ) # Preciosidad
		
		active_duration = timedelta(seconds=0)
		for day in days:
			start, end = self.get_start_and_end_datetimes(day=day)
			if start is None:
				active_duration += timedelta(seconds=0)
			else:
				active_duration += end-start
		
		return active_duration
	
	def add_initial_values(self, initial_dec_vars: InitialDecVarsValues) -> None:
		""" Add initial values to the integer decision variables """
		
		for var_id in asdict(self):
			if var_id in asdict(initial_dec_vars):
				initial_value = getattr(initial_dec_vars, var_id)
				current_value = getattr(self, var_id)
				if isinstance(initial_value, pd.Series) and isinstance(current_value, pd.Series):
					value = pd.concat([current_value, initial_value]).sort_index()
				else:
					raise ValueError(f"Initial value for {var_id} should be a pd.Series")
				setattr(self, var_id, value)    
				
	def resample(self, target_freq_sec: int) -> "IntegerDecisionVariables":
		""" Return a new resampled object instance """
		output = {}
		for name, value in asdict(self).items():
			if value is None:
				continue
			elif not isinstance(value, pd.Series):
				raise TypeError(f"All attributes must be pd.Series for datetime indexing. Got {type(value)} instead.")
			
			# target_freq = int(float(args[0][:-1]))
			# current_freq = value.index.freq.n
			
			value = value.resample(f"{target_freq_sec}s")
			# if  target_freq > current_freq: # Downsample
			# 	value = value.first()
			# else: # Upsample
			value = value.ffill()
			output[name] = value
			
		return IntegerDecisionVariables(**output)

	def dump_in_span(self, span: tuple[int, Optional[int]] | tuple[datetime, Optional[datetime]], return_format: Literal["values", "series"] = "values", **kwargs) -> 'IntegerDecisionVariables':
		""" Dump decision variables within a given span """
		
		vars_dict = dump_in_span(vars_dict=asdict(self), span=span, return_format=return_format, **kwargs)
		return IntegerDecisionVariables(**vars_dict)
	
	def to_dict(self) -> dict:
		"""
		Convert decision variables into a dictionary.
		"""
		return asdict(self)
	
@dataclass
class DecisionVariablesUpdates:
	""" Number of decision variable updates in the optimization window """
	
	# sf_active: int # 
	# ts_active: int # 
	sfts_mode: int #
	med_mode: int #
	# med_active: int # 
	# med_vac_state: int # 
	qsf: int # 
	qts_src: int # 
	qmed_s: int # 
	qmed_f: int # 
	Tmed_s_in: int # 
	Tmed_c_out: int # 

	def to_dict(self) -> dict:
		"""
		Convert decision variables updates into a dictionary.
		"""
		return asdict(self)
	
	# def __post_init__(self):
		# Validate that SfTs FSM related decision varaiables have the same number of updates
		# assert self.sf_active == self.ts_active, "Solar field and thermal storage logical variables should have the same number of updates"
		
		# Validate that MED FSM related decision variables have the same number of updates
		# assert self.med_active == self.med_vac_state, "MED logical variables should have the same number of updates"
		
		# TODO: Would be good to validate that the number of updates is within:
		# 1 <= n_uptes <= optim_window_size / sample_time_mod (=n_evals_mod)

# Check errors during development
# assert [field.name for field in fields(DecisionVariables)] == [field.name for field in fields(DecisionVariablesUpdates)], \
#     "Attributes of DecisionVariables should exactly match attributes in DecisionVariableUpdates"

class OptimToFsmsVarIdsMapping(NamedTuple):
	"""
	Examples:
	# Convert from optim id to fsm id
	print(f"optim_id: sfts_mode -> model id: {OptimToFsmsVarIdsMapping.sfts_mode.value}")

	# Convert from fsm id to optim id
	print(f"fsm id: qts_src -> optim_id: {OptimToFsmsVarIdsMapping('qts_src').name}")
	"""
	sfts_mode: tuple = ("sf_active", "ts_active")
	med_mode: tuple  = ("med_active", "med_vacuum_state")
	
# class FsmstoOptimVarIdsMapping:
#     sf_active = "sf_active" 
#     ts_active = "ts_active"
#     med_mode = ""
	
class OptimVarIdstoModelVarIdsMapping(Enum):
	"""
	Mapping between optimization decision variable ids and model variable ids.
	Using an Enum allows for bi-directional lookups compared to a dictionary
	
	Maybe we could include here only variables that differ, and by default assume
	that the variable ids are the same in the optimization and the model
	
	Structure:
	optim_var_id = model_var_id
	
	Examples:
	# Convert from optim id to model id
	print(f"optim_id: qsf -> model id: {OptimVarIdstoModelVarIdsMapping.qsf.value}")

	# Convert from model id to optim id
	print(f"model id: qts_src -> optim_id: {OptimVarIdstoModelVarIdsMapping('qts_src').name}")
	"""
	sf_active = "sf_active"
	ts_active = "ts_active"
	med_active = "med_active"
	med_vac_state = "med_vacuum_state"
	qsf = "qsf"
	qts_src = "qts_src"
	qmed_s = "qmed_s"
	qmed_f = "qmed_f"
	Tmed_s_in = "Tmed_s_in"
	Tmed_c_out = "Tmed_c_out"

class RealLogicalDecVarDependence(Enum):
	""" Utility class that defines dependence relationship between real 
	decision variables and operation modes / integer ones """
	
	qsf = "sfts_mode"
	qts_src = "sfts_mode"
	qmed_s = "med_mode"
	qmed_f = "med_mode"
	Tmed_s_in = "med_mode"
	Tmed_c_out = "med_mode"
	
@dataclass
class RealDecVarsBoxBounds:
	""" Real decision variables box bounds, as in: (lower bound, upper bound)"""
	qsf: tuple[float, float]
	qts_src: tuple[float, float]
	qmed_s: tuple[float, float]
	qmed_f: tuple[float, float]
	Tmed_s_in: tuple[float, float]
	Tmed_c_out: tuple[float, float]
	
	@classmethod
	def initialize(cls, fmp: FixedModelParameters, Tmed_c_in: float) -> 'RealDecVarsBoxBounds':
	
		return cls(
			qts_src = (fmp.ts.qts_src_min, fmp.ts.qts_src_max),
			qsf = (fmp.sf.qsf_min, fmp.sf.qsf_max),
			Tmed_s_in = (fmp.med.Tmed_s_min, fmp.med.Tmed_s_max),
			Tmed_c_out = (Tmed_c_in+2, Tmed_c_in+10),
			qmed_s = (fmp.med.qmed_s_min, fmp.med.qmed_s_max),
			qmed_f = (fmp.med.qmed_f_min, fmp.med.qmed_f_max),
		)

@dataclass
class RealDecisionVariablesUpdatePeriod:
	qsf: int = 1800
	qts_src: int = 1800
	qmed_s: int = 3600
	qmed_f: int = 3600
	Tmed_s_in: int = 3600
	Tmed_c_out: int = 3600
	
@dataclass
class RealDecisionVariablesUpdateTimes:
	qsf: list[datetime]
	qts_src: list[datetime]
	qmed_s: list[datetime]
	qmed_f: list[datetime]
	Tmed_s_in: list[datetime]
	Tmed_c_out: list[datetime]
@dataclass
class FsmData:
	metadata: dict
	paths_df: pd.DataFrame
	valid_inputs: list[list[list[float]]]
	
@dataclass
class ProblemSamples:
	# Times to samples transformation
	n_evals_mod_in_hor_window: int # Number of model evalations along the optimization window
	n_evals_mod_in_opt_step: int # Number of model evaluations in one optimization step
	episode_samples: int # Number of samples in one episode
	optim_window_samples: int # Number of samples in the optimization window
	max_opt_steps: int # Max number of steps for the optimization scheme
	span: int # Number of previous samples to keep track of
	default_n_dec_var_updates: int # Default number of decision variable updates in the optimization window (depends on sample_time_opt)
	max_dec_var_updates: int # Maximum number of decision variable updates in the optimization window (depends on sample_time_mod)
	min_dec_var_updates: int  = 1 # Minimum number of decision variable updates in the optimization window
	
# @dataclass
# class OptimizationParamters:
# 	operation_actions: Optional[OperationActionType] = None # Optional for MINLP, required in nNLP alternative. Defines the operation actions/updates for each subsystem
# 	real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod = field(default_factory=lambda: RealDecisionVariablesUpdatePeriod()) # nNLP
# 	initial_dec_vars_values: InitialDecVarsValues = field(default_factory=lambda: InitialDecVarsValues())  # nNLP
# 	irradiance_thresholds: IrradianceThresholds = field(default_factory=lambda: IrradianceThresholds()) # nNLP
# 	op_optim_computation_time = timedelta(minutes=15)
# 	op_optim_eval_period = timedelta(minutes=30)

@dataclass
class ProblemParameters:
	sample_time_mod: int = 400 # Model sample time, seconds
	sample_time_opt: int = int(3600 * 0.8) # Optimization evaluations period, seconds
	sample_time_ts: int = 3600 # Thermal storage sample time, seconds (used in nNLP alternative)
	optim_window_time: int = 3600 * 8 # Optimization window size, seconds
	episode_duration: Optional[int] = None # By default use len(df)
	idx_start: Optional[int] = None # By default estimate from sf fixed_mod_params.delay_span
	env_params: EnvironmentParameters = field(default_factory=lambda: EnvironmentParameters())
	fixed_model_params: FixedModelParameters = field(default_factory=lambda: FixedModelParameters())
	model_params: ModelParameters = field(default_factory=lambda: ModelParameters())
	on_limits_violation_policy: Literal['raise_error', 'clip', 'penalize'] = "clip" # Policy to apply when inputs result in outputs outside their operating limits (model.evaluate_fitness_function)
	fsm_params: FsmParameters = field(default_factory=lambda: FsmParameters(
		med=MedFsmParams(
			vacuum_duration_time = 1*3600, # 1 hour
			brine_emptying_time = 30*60,   # 30 minutes
			startup_duration_time = 20*60, # 20 minutes
			off_cooldown_time = 12*3600,   # 12 hours
			active_cooldown_time = 4*3600, # 3 hours
		),
		sf_ts=SftsFsmParams(
			recirculating_ts_enabled = False,
			idle_cooldown_time = 1*3600,   # 1 hour
		)
	))
	fsm_internal_states: FsmInternalState = field(default_factory=lambda: FsmInternalState())
	fsm_valid_sequences: dict[ str, list[list[int]] ] = field(default_factory=lambda: {
		'MED': [
			[MedState.IDLE.value, MedState.STARTING_UP.value, MedState.ACTIVE.value],
			[MedState.GENERATING_VACUUM.value, MedState.STARTING_UP.value, MedState.ACTIVE.value],
		],
		'SFTS': [
			[SfTsState.HEATING_UP_SF.value, SfTsState.SF_HEATING_TS.value],
		]
	})
	dec_var_updates: Optional[DecisionVariablesUpdates] = None # Set automatically in utils.initialization.problem_initialization if not manually defined
	optim_window_days: Optional[int] = None # Automatically computed from optim_window_time
	initial_states: Optional[InitialStates] = None # Optional, if specified model will be initialized with these states
	system_shutdown_duration: timedelta = timedelta(hours=1) # Approximate time to shutdown all subsystems in the system
	operation_min_duration: timedelta = timedelta(hours=2) # Mininum exptected operation time in order for the system to be activated
	operation_actions: Optional[OperationActionType] = None # Optional for MINLP, required in nNLP alternative. Defines the operation actions/updates for each subsystem
	real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod = field(default_factory=lambda: RealDecisionVariablesUpdatePeriod()) # nNLP
	initial_dec_vars_values: InitialDecVarsValues = field(default_factory=lambda: InitialDecVarsValues())  # ... (nNLP)
	irradiance_thresholds: IrradianceThresholds = field(default_factory=lambda: IrradianceThresholds()) # ... (nNLP)
	op_optim_computation_time: timedelta = timedelta(minutes=15)
	op_optim_eval_period: timedelta = timedelta(minutes=30)
	op_plan_startup_computation_time: timedelta = timedelta(hours=3)
	op_plan_shutdown_computation_time: timedelta = timedelta(minutes=30)
	
	def __post_init__(self):
		""" Make convenient to initialize this dataclass from dumped instances """
		for fld in fields(self):
			if is_dataclass(fld.type):
				value = getattr(self, fld.name)
				if isinstance(value, dict):
				# if not isinstance(value, fld.type) and value is not None:
					setattr(self, fld.name, fld.type(**value))
					
		self.optim_window_days = math.ceil(self.optim_window_time / (24*3600))
		
	@classmethod
	def initialize(cls, problem_type: Literal["MINLP", "nNLP"], **kwargs) -> "ProblemParameters":
		
		if problem_type == "nNLP":
			assert kwargs.get("operation_actions", None) is not None, "operation_actions must be specified for nNLP problem type"
		
		return cls(**kwargs)    

@dataclass
class ProblemData:
	df: pd.DataFrame
	problem_params: ProblemParameters
	problem_samples: ProblemSamples
	model: SolarMED
	# optim_params: Optional[OptimizationParamters] = None
 
	def copy(self, ) -> "ProblemData":
		return copy.deepcopy(self)

# Legacy MINLP
@dataclass
class AlgorithmParameters:
	pop_size: int = 32
	n_gen: int = 80
	seed_num: int = 23
	
# Legacy MINLP
@dataclass
class PopulationResults:
	pop_per_gen: list[list[float|int]] # (gen, individual, dec.variable)
	fitness_per_gen: list[list[float]] # (gen, individual)
	time_per_gen: list[float] # (gen, )
	time_total: float
	best_idx_per_gen: list[int] # (gen, )
	worst_idx_per_gen: list[int] # (gen, )
	
	# def __post_init__(self, ):
	#     # Check type of attributes, if they are numpy arrays, make them into lists

	@classmethod
	def initialize(cls, problem, pop_size: int, n_gen: int, elapsed_time: int) -> 'PopulationResults':
	
		x_evaluated = problem.x_evaluated
		fitness_record = problem.fitness_history
		x_history: list[list[list[int | float]]] = []
		fitness_history: list[list[float]] = []
		best_idx: list[int] = []
		worst_idx: list[int] = []
		for idx in range(0, len(x_evaluated)-1, pop_size):
			x_history.append( x_evaluated[idx:idx+pop_size] )
			fitness_history.append( fitness_record[idx:idx+pop_size] )
			best_idx.append( int(np.argmin(fitness_history[-1])) )
			worst_idx.append( int(np.argmax(fitness_history[-1])) )
				
		return cls(
			pop_per_gen=x_history,
			fitness_per_gen=fitness_history,
			best_idx_per_gen=best_idx,
			worst_idx_per_gen=worst_idx,
			time_per_gen=elapsed_time/n_gen,
			time_total=elapsed_time
		)


@dataclass
class AlgoParams:
	algo_id: str = "sea"
	max_n_obj_fun_evals: int = 1_000 # When debugging, change to a lower value
	max_n_logs: int = 300
	pop_size: int = 1

	params_dict: Optional[dict] = None
	log_verbosity: Optional[int] = None
	gen: Optional[int] = None

	def __post_init__(self, ):

		if self.algo_id in ["gaco", "sga", "pso_gen"]:
			self.gen = self.max_n_obj_fun_evals // self.pop_size
			self.params_dict = {
				"gen": self.gen,
			}
		elif self.algo_id == "simulated_annealing":
			self.gen = self.max_n_obj_fun_evals // self.pop_size
			self.params_dict = {
				"bin_size": self.pop_size,
				"n_T_adj": self.gen
			}
		else:
			self.pop_size = 1
			self.gen = self.max_n_obj_fun_evals
			self.params_dict = { "gen": self.max_n_obj_fun_evals // self.pop_size }

		if self.log_verbosity is None:
			self.log_verbosity = math.ceil( self.gen / self.max_n_logs)
   
	def copy(self) -> "AlgoParams":
		""" Create a deep copy of this AlgorithmParameters instance. """
		return copy.deepcopy(self)


@dataclass
class ProblemsEvaluationParameters:
	"""
	Parameters for the problems evaluation process.
	"""
	drop_fraction: float = 0.3 # Fraction of problems to drop per update (0 to 1)
	max_n_obj_fun_evals: int = 1_000 # Total maximum number of objective function evaluations
	max_n_parallel_problems: int = 50 # Maximum number of problems to evaluate in parallel
	n_updates: Optional[int] = None # Number of (problem drop) updates to perform
	n_obj_fun_evals_per_update: Optional[int] = None # Number of objective function evaluations between updates
	archipelago_topology: Optional[PygmoArchipelagoTopologies] = "unconnected"
	n_instances: int = 1 # Number of problem instances (islands) evolving in parallel in a connected archipelago. Used for op.optim 

	def __post_init__(self):
		if self.archipelago_topology != "unconnected":
			assert self.n_instances >= 1, "If using a connected archipelago topology, n_instances must be greater or equal to 1"
			self.n_updates = 1
		assert self.n_updates is not None or self.n_obj_fun_evals_per_update is not None, "Either n_updates or n_obj_fun_evals_per_update must be provided"
		assert self.drop_fraction >= 0 and self.drop_fraction <= 1, "Fraction of problems to drop per update must be between 0 and 1"


		if self.n_obj_fun_evals_per_update is None:
			self.n_obj_fun_evals_per_update = self.max_n_obj_fun_evals // self.n_updates
		elif self.n_updates is None:
			self.n_updates = self.max_n_obj_fun_evals // self.n_obj_fun_evals_per_update

	def update_problems(self, problems_fitness: list[float]) -> tuple[list[int], list[int]]:
		"""
		Drop the worst performing problems based on the drop_fraction.
		Returns the indices of the problems to keep and to drop.
		NaN values in problems_fitness are ignored in the decision process,
		but their positions are preserved in index calculations.
		"""

		# Get the indices of non-NaN entries
		valid_indices = [i for i, val in enumerate(problems_fitness) if not math.isnan(val)]
		# Get the fitness values of non-NaN entries
		valid_fitness = [(i, problems_fitness[i]) for i in valid_indices]
		# Sort valid fitness values by value (descending = worst first)
		valid_fitness_sorted = sorted(valid_fitness, key=lambda x: x[1], reverse=True)

		# Determine number to drop
		n_to_drop = int(len(valid_fitness_sorted) * self.drop_fraction)
		# Extract the global indices to drop
		drop_indices = [i for i, _ in valid_fitness_sorted[:n_to_drop]]
		# Extract the global indices to keep (remaining non-NaNs not in drop)
		keep_indices = [i for i in valid_indices if i not in drop_indices]

		return keep_indices, drop_indices

	def copy(self) -> "ProblemsEvaluationParameters":
		""" Create a deep copy of this ProblemsEvaluationParameters instance. """
		return copy.deepcopy(self)

@dataclass
class OptimizationParams:
    algo_params: AlgoParams
    problems_eval_params: ProblemsEvaluationParameters
