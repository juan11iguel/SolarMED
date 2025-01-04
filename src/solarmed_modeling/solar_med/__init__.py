from dataclasses import dataclass, field, asdict, fields
import re
import numpy as np
import pandas as pd
from iapws import IAPWS97 as w_props
from loguru import logger
from enum import Enum
from typing import Any, Literal, TypeVar, get_args
import pickle
from pydantic import (BaseModel,
                      Field,
                      ConfigDict,
                      field_validator,
                      computed_field,
                      ValidationInfo,
                      PositiveFloat,
                      PrivateAttr,
                      field_serializer)
from pathlib import Path

from solarmed_modeling.data_validation import (within_range_or_zero_or_max, 
                                               within_range_or_min_or_max,
                                               conHotTemperatureType, 
                                               check_value_single)
from solarmed_modeling.solar_field import (solar_field_model, 
                                           ModelParameters as SfModParams, 
                                           FixedModelParameters as SfFixedModParams)
from solarmed_modeling.heat_exchanger import (heat_exchanger_model,
                                              calculate_heat_transfer_effectiveness,
                                              ModelParameters as HexModParams)
from solarmed_modeling.thermal_storage import (thermal_storage_two_tanks_model, 
                                               ModelParameters as TsModParams,
                                               FixedModelParameters as TsFixedModParams)
from solarmed_modeling.three_way_valve import three_way_valve_model
from solarmed_modeling.med import (FixedModelParameters as MedFixedModParams,
                                   MedModel)
from solarmed_modeling.heat_gen_and_storage import heat_generation_and_storage_subproblem
from solarmed_modeling.power_consumption import Actuator
from solarmed_modeling.fsms import (MedState, 
                                    MedVacuumState, 
                                    SfTsState, 
                                    ThermalStorageState, 
                                    SolarFieldState, 
                                    SolarMedState)
from solarmed_modeling.fsms.med import (MedFsm,
                                        FsmParameters as MedFsmParams,
                                        FsmInternalState as MedFsmInternalState,
                                        FsmInputs as MedFsmInputs)
from solarmed_modeling.fsms.sfts import (SolarFieldWithThermalStorageFsm,
                                         FsmParameters as SfTsFsmParams,
                                         FsmInternalState as SfTsFsmInternalState,
                                         FsmInputs as SfTsFsmInputs,
                                         get_sfts_state,
                                         get_sf_ts_individual_states)
"""
    TODO: 
    - [x] Integrate new models deprecating the inverse approach
    - [x] Modify MED FSM to only use qsf instead of Tsf_out
    - [ ] Add cooldown times support to FSMs
    - [ ] Partial initialization of FSMs
"""

# logger.disable(__name__)

# logger.disable("solarMED_modeling")
np.set_printoptions(precision=2) # Set numpy to print only 2 decimal places
dot = np.multiply # Alias for element-wise multiplication
Nsf_max: int = 100  # Maximum number of steps to keep track of the solar field flow rate, should be higher than the maximum expected delay

SupportedAlternativesLiteral: TypeVar = Literal["standard", "constant-water-props"]
supported_eval_alternatives: tuple[str] = get_args(SupportedAlternativesLiteral)
""" 
    - standard: intializes two water properties objects for low and high temperatures at every step 
    - constant-water-props: only does it once at initialization
"""

class ModelVarType(Enum):
    TIMESERIES = 0
    PARAMETER  = 1
    INITIAL_STATE = 2

states_sf_ts: list[SfTsState] = [state for state in SfTsState]
states_med: list[MedState] = [state for state in MedState]

@dataclass
class ModelParameters:
    sf: SfModParams   = field(default_factory=lambda: SfModParams())
    ts: TsModParams   = field(default_factory=lambda: TsModParams())
    hex: HexModParams = field(default_factory=lambda: HexModParams())
    
    def __post_init__(self):
        """ Make it convenient to initialize this dataclass from dumped instances """
        
        for fld in fields(self):
            value = getattr(self, fld.name)
            # if not isinstance(value, fld.type):
            if isinstance(value, dict):
                setattr(self, fld.name, fld.type(**value))
                
@dataclass
class FixedModelParameters:
    med: MedFixedModParams = field(default_factory=lambda: MedFixedModParams())
    sf: SfFixedModParams = field(default_factory=lambda: SfFixedModParams())
    ts: TsFixedModParams = field(default_factory=lambda: TsFixedModParams())
    
    def __post_init__(self):
        """ Make it convenient to initialize this dataclass from dumped instances """
        
        for fld in fields(self):
            value = getattr(self, fld.name)
            # if not isinstance(value, fld.type):
            if isinstance(value, dict):
                setattr(self, fld.name, fld.type(**value))
    
@dataclass
class FsmParameters:
    med: MedFsmParams = field(default_factory=lambda: MedFsmParams())
    sf_ts: SfTsFsmParams = field(default_factory=lambda: SfTsFsmParams())
    
    def __post_init__(self):
        """ Make it convenient to initialize this dataclass from dumped instances """
        
        for fld in fields(self):
            value = getattr(self, fld.name)
            # if not isinstance(value, fld.type):
            if isinstance(value, dict):
                setattr(self, fld.name, fld.type(**value))
    
@dataclass
class FsmInternalState:
    med: MedFsmInternalState = field(default_factory=lambda: MedFsmInternalState())
    sf_ts: SfTsFsmInternalState = field(default_factory=lambda: SfTsFsmInternalState())
    
    def __post_init__(self):
        """ Make it convenient to initialize this dataclass from dumped instances """
        
        for fld in fields(self):
            value = getattr(self, fld.name)
            # if not isinstance(value, fld.type):
            if isinstance(value, dict):
                setattr(self, fld.name, fld.type(**value))

    
@dataclass
class EnvironmentParameters:
    cost_w: float = 3 # Cost of water, €/m³ 
    cost_e: float = 0.05 # Cost of electricity, €/kWhe
    
@dataclass
class ActuatorsMaping:
    med: dict[str, str | Actuator | dict] = field(default_factory=lambda: {
        'qmed_b': "med_brine_pump", 
        'qmed_f': "med_feed_pump",
        'qmed_d': "med_distillate_pump", 
        'qmed_c': "med_cooling_pump",
        'qmed_s': "med_heatsource_pump",
    })
    sf: dict[str, str, str | Actuator | dict] = field(default_factory=lambda: {
        'qsf': "sf_pump",
    })
    ts: dict[str, str, str | Actuator | dict] = field(default_factory=lambda: {
         'qts_src': "ts_src_pump"
    })


class SolarMED(BaseModel):
    """
    Model of a Multi-effect distillation pilot plant, static collectors solar field  and thermal storage system
    located at Plataforma Solar de Almería.

    See more at: https://www.psa.es/es/instalaciones/desalacion/med.php (outdated?)

    How to use:
        1. Create an instance of the class, for initialization it requires: thermal storage initial state, and solar
        field initial temperature.
        2. Evaluate the system evolution using the step method, it requires the environmental conditions and the
        values of the decision variables.
        3. Terminate the model using the terminate method.

    Some notes:
        - For the flows, if the input value is below it's minimum operating value, it's set to zero.
        - If some decision variable produces an output outside operating limits, the variable is set to its saturated
        value
        - Decision variables have a suffix `_sp` to indicate that they are setpoints, some decision variables are not
        real inputs of the system, so even if they pass the validation, their values are subject to change, this
        change is reflected in the equivalent output variable. Tsf_out_sp < Tsf_out_min ->
        Tsf_out = Tsf_out(k-1) - losses to the environment
        - Systems can be activated or deactivated depending on the decision variables and conditions. The different
        operating modes are defined in the SolarMED_states.
        - To export the model state to a dataframe, use the `to_dataframe` method.
        - To export (serialize) the model as a dictionary, use pydantic's built in method `solar_med.model_dump()` (being solar an
        instance of the class `solar_med = SolarMED(...)`)  to produce a dictionary representation of the model.
        By default, fields are configured to be exported or not, but this can be overided with arguments to the
        method, check its [docs](https://docs.pydantic.dev/latest/concepts/serialization/).
        - Alternatively, the model can be serialized as JSON using the `solar_med.model_dump_json()` method.
    """

    # Parameters
    # Important to define first, so that they are available for validation
    fixed_model_params: FixedModelParameters = Field(FixedModelParameters(), titile="Fixed model parameters",
                                                     description="Fixed model parameters", json_schema_extra={"var_type": ModelVarType.PARAMETER})
    model_params: ModelParameters = Field(ModelParameters(), titile="Model parameters",
                                                description="Component models parameters", json_schema_extra={"var_type": ModelVarType.PARAMETER})
    fsms_params: FsmParameters = Field(FsmParameters(), titile="FSM parameters",
                                            description="Finite State Machine parameters", json_schema_extra={"var_type": ModelVarType.PARAMETER})
    fsms_internal_states: FsmInternalState = Field(FsmInternalState(), titile="FSM internal state",
                                                            description="Finite State Machine internal state", json_schema_extra={"var_type": ModelVarType.PARAMETER})
    env_params: EnvironmentParameters = Field(EnvironmentParameters(), titile="Environment parameters",
                                                description="Environment parameters", json_schema_extra={"var_type": ModelVarType.PARAMETER})
    ## General parameters
    use_models: bool = Field(True, title="use_models", json_schema_extra={"units": "-", "var_type": ModelVarType.PARAMETER},
                             description="Whether to evaluate the models for the components of the SolarMED system, if set to False, `use_finite_state_machines` must be set to True")
    use_finite_state_machine: bool = Field(False, title="FSM enabled", json_schema_extra={"units": "-", "var_type": ModelVarType.PARAMETER},
                                           description="Whether to enable or not the Finite State Machine (FSM) for the SolarMED system")
    sample_time: float = Field(60, description="Sample rate (seg)", title="sample rate",
                               json_schema_extra={"units": "s", "var_type": ModelVarType.PARAMETER})
    resolution_mode: SupportedAlternativesLiteral = Field("constant-water-props", repr=True, json_schema_extra={"var_type": ModelVarType.PARAMETER},
                   description="Mode of solving the model, either `standard` which intializes two water properties objects for low and high temperatures at every step, and constant-water-props, which only does it once at initialization")
    on_limits_violation_policy: Literal['raise_error', 'clip', 'penalize'] = Field('clip', title="On limits violation policy",
                                                                                    json_schema_extra={"var_type": ModelVarType.PARAMETER},
                                                                                    description="Policy to apply when inputs result in outputs outside their operating limits")
    penalty: float = Field(0, title="penalty", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "u.m."}, ge=0,
                           description="Penalty for undesired states or conditions", repr=False)
    
    ## Finite State Machine parameters
    # TODO: Add cooldown times / hystheresis? It would greatly delay/reduce the exponential growth/explosion of possible transitions
    # active_cooldown_time: int = Field(3*60, title="MED active,cooldown", json_schema_extra={"units": "s"},
    #                                   description="Minimum time than the MED system needs to stay active after transitioning to this state (seconds)")
    # off_cooldown_time: int = Field(5*60, title="MED off,cooldown", json_schema_extra={"units": "s"},
    #                                description="Minimum time than the MED system needs to stay off after transitioning to this state (seconds)")

    ## Common
    actuators_consumptions: ActuatorsMaping = Field(ActuatorsMaping(), title="Actuators consumption object",
                                                                  json_schema_extra={"var_type": ModelVarType.PARAMETER},
                                                                  description="Map of process variables to actuator objects/ids", repr=False)
    Jsf_ts: float = Field(None, title="Jsf,ts", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWe"},
                       description="Output. Solar field and thermal storage electrical power consumption (kWe)")
    ## MED
    Jvacuum: list[float] = Field(default_factory=lambda: [0., 1., 5.], title="Jvacuum", json_schema_extra={"var_type": ModelVarType.PARAMETER, "units": "kWe"},
                                 description="MED vacuum system electrical power consumption (kWe)", repr=False)

    # Variables (states, outputs, decision variables, inputs, etc.)
    ## General
    current_sample: int = Field(0, title="Current sample (s)", json_schema_extra={"units": "sec", "var_type": ModelVarType.TIMESERIES},
                                description="Current model sample (NOTE: not in sync with FSMs sample counters)")
    ## Environment (uncontrolled inputs)
    wmed_f: float = Field(35, title="wmed,f", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "g/kg"},
                          description="Environment. Seawater / MED feedwater salinity (g/kg)", gt=0)
    Tamb: float = Field(None, title="Tamb", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                        description="Environment. Ambient temperature (ºC)", ge=-15, le=50)
    I: float = Field(None, title="I", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "W/m2"},
                     description="Environment. Solar irradiance (W/m2)", ge=0, le=2000)
    Tmed_c_in: float = Field(None, title="Tmed,c,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                             description="Environment. Seawater temperature (ºC)", ge=10, le=28)

    ## Thermal storage
    ### Decision variables
    qts_src_sp: float = Field(None, title="qts,src*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                              description="Decision variable. Thermal storage recharge flow rate (m³/h)", ge=0)
    ### Outputs
    qts_src: float = Field(None, title="qts,src", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                           description="Output. Thermal storage recharge flow rate after validation / FSM evaluation (m³/h)", ge=0)
    qts_dis: float = Field(None, title="qts,dis", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                           description="Output. Thermal storage discharge flow rate (m³/h)", ge=0)
    Tts_h_in: conHotTemperatureType = Field(None, title="Tts,h,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                            description="Output. Thermal storage heat source inlet temperature, to top of hot tank == Thx_s_out (ºC)")
    Tts_c_in: conHotTemperatureType = Field(None, title="Tts,c,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                            description="Output. Thermal storage load discharge inlet temperature, to bottom of cold tank == Tmed_s_out (ºC)")
    Tts_h_out: conHotTemperatureType = Field(None, title="Tts,h,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                             description="Output. Thermal storage heat source outlet temperature, from top of hot tank == Tts_h_t (ºC)")
    Tts_h: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,h",
                                                                                   json_schema_extra={"var_type": ModelVarType.INITIAL_STATE, "units": "C"},
                                                                                   description="Output. Temperature profile in the hot tank (ºC)")
    Tts_c: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,c",
                                                                                   json_schema_extra={"var_type": ModelVarType.INITIAL_STATE, "units": "C"},
                                                                                   description="Output. Temperature profile in the cold tank (ºC)")
    Pth_ts_src: float = Field(None, title="Pth,ts,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                           description="Output. Thermal storage inlet power (kWth)")
    Pth_ts_dis: float = Field(None, title="Pth,ts,dis", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                           description="Output. Thermal storage outlet power (kWth)")
    Jts: float = Field(None, title="Jts,e", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWe"},
                       description="Output. Thermal storage electrical power consumption (kWe)")

    ## Solar field
    ### Decision variables
    qsf_sp: float = Field(None, title="qsf*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                       description="Decision variable. Solar field flow rate (m³/h)")
    ### Outputs
    qsf: float = Field(None, title="qsf", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                       description="Output. Solar field flow rate after validation / FSM evaluation (m³/h)")
    Tsf_out: conHotTemperatureType = Field(None, title="Tsf,out*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                              description="Output. Solar field outlet temperature (ºC)")
    Tsf_in: conHotTemperatureType = Field(None, title="Tsf,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                          description="Output. Solar field inlet temperature (ºC)")
    Tsf_in_ant: np.ndarray[conHotTemperatureType] = Field(..., title="Tsf,in_ant", json_schema_extra={"var_type": ModelVarType.INITIAL_STATE, "units": "C"},
                                                          description="Solar field inlet temperature in the previous Nsf_max steps (ºC)")
    qsf_ant: np.ndarray[float] = Field(..., repr=False, exclude=False, json_schema_extra={"var_type": ModelVarType.INITIAL_STATE},
                                       description='Solar field flow rate in the previous Nsf_max steps', )
    Tsf_out_ant: conHotTemperatureType = Field(None, title="Tsf,out,ant", json_schema_extra={"var_type": None, "units": "C"},
                                               description="Output. Solar field prior outlet temperature (ºC)")
    SEC_sf: float = Field(None, title="SEC_sf", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWhe/kWth"},
                          description="Output. Solar field conversion efficiency (kWhe/kWth)")
    Jsf: float = Field(None, title="Jsf,e", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kW"},
                       description="Output. Solar field electrical power consumption (kWe)")
    Pth_sf: float = Field(None, title="Pth_sf", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                       description="Output. Solar field thermal power generated (kWth)")

    ## MED
    ### Decision variables    
    qmed_s_sp: float = Field(None, title="mmed,s*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                             description="Decision variable. MED hot water flow rate (m³/h)")
    qmed_f_sp: float = Field(None, title="mmed,f*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                             description="Decision variable. MED feedwater flow rate (m³/h)")
    Tmed_s_in_sp: float = Field(None, title="Tmed,s,in*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                description="Decision variable. MED hot water inlet temperature (ºC)")
    Tmed_c_out_sp: float = Field(None, title="Tmed,c,out*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                 description="Decision variable. MED condenser outlet temperature (ºC)")
    ### Outputs
    qmed_s: float = Field(None, title="mmed,s", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED hot water flow rate after validation / FSM evaluation (m³/h)")
    qmed_f: float = Field(None, title="mmed,f", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED feedwater flow rate validation / FSM evaluation (m³/h)")
    Tmed_s_in: float = Field(None, title="Tmed,s,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                             description="Output. MED hot water inlet temperature validation / FSM evaluation (ºC)")
    Tmed_c_out: float = Field(None, title="Tmed,c,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                              description="Output. MED condenser outlet temperature validation / FSM evaluation (ºC)", ge=0)
    qmed_c: float = Field(None, title="mmed,c", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED condenser flow rate (m³/h)")
    Tmed_s_out: float = Field(None, title="Tmed,s,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                              description="Output. MED heat source outlet temperature (ºC)")
    qmed_d: float = Field(None, title="mmed,d", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED distillate flow rate (m³/h)")
    qmed_b: float = Field(None, title="mmed,b", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED brine flow rate (m³/h)")
    Jmed: float = Field(None, title="Jmed", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWe"},
                        description="Output. MED electrical power consumption (kW)")
    Pth_med: float = Field(None, title="Pmed", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                        description="Output. MED thermal power consumption ~= Pth_ts_out (kW)")
    STEC_med: float = Field(None, title="STEC_med", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWhth/m3"},
                            description="Output. MED specific thermal energy consumption (kWhth/m³)")
    SEEC_med: float = Field(None, title="SEEC_med", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWhe/m3"},
                            description="Output. MED specific electrical energy consumption (kWhe/m³)")

    ## Heat exchanger
    # Basically copies of existing variables, but with different names, no bounds checking
    Thx_p_in: conHotTemperatureType = Field(None, title="Thx,p,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                            description="Output. Heat exchanger primary circuit (hot side) inlet temperature == Tsf_out (ºC)")
    Thx_p_out: conHotTemperatureType = Field(None, title="Thx,p,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                             description="Output. Heat exchanger primary circuit (hot side) outlet temperature == Tsf_in (ºC)")
    Thx_s_in: conHotTemperatureType = Field(None, title="Thx,s,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                            description="Output. Heat exchanger secondary circuit (cold side) inlet temperature == Tts_c_out(ºC)")
    Thx_s_out: conHotTemperatureType = Field(None, title="Thx,s,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                             description="Output. Heat exchanger secondary circuit (cold side) outlet temperature == Tts_t_in (ºC)")
    qhx_p: float = Field(None, title="mhx,p", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                         description="Output. Heat exchanger primary circuit (hot side) flow rate == qsf (m³/h)")
    qhx_s: float = Field(None, title="mhx,s", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                         description="Output. Heat exchanger secondary circuit (cold side) flow rate == qts_src (m³/h)")
    Pth_hx_p: float = Field(None, title="Pth,hx,p", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                         description="Output. Heat exchanger primary circuit (hot side) power == Pth_sf (kWth)")
    Pth_hx_s: float = Field(None, title="Pth,hx,s", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                         description="Output. Heat exchanger secondary circuit (cold side) power == Pth_ts_in (kWth)")
    epsilon_hx: float = Field(None, title="εhx", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                              description="Output. Heat exchanger effectiveness (-)")

    ## Three-way valve
    # Same case as with heat exchanger
    R3wv: float = Field(None, title="R3wv", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                        description="Output. Three-way valve mix ratio (-)")
    q3wv_src: float = Field(None, title="m3wv,src", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                            description="Output. Three-way valve source flow rate == qts,dis (m³/h)")
    q3wv_dis: float = Field(None, title="m3wv,dis", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                            description="Output. Three-way valve discharge flow rate == mmed,s (m³/h)")
    T3wv_src: conHotTemperatureType = Field(None, title="T3wv,src", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                            description="Output. Three-way valve source temperature == Tts,h,t (ºC)")
    T3wv_dis_in: conHotTemperatureType = Field(None, title="T3wv,dis,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                               description="Output. Three-way valve discharge inlet temperature == Tmed,s,in (ºC)")
    T3wv_dis_out: conHotTemperatureType = Field(None, title="T3wv,dis,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                                description="Output. Three-way valve discharge outlet temperature == Tmed,s,out (ºC)")

    ## Others
    med_active: bool = Field(False, title="med_active", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                             description="Flag indicating if the MED is active", repr=True)
    sf_active: bool = Field(False, title="sf_active", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                            description="Flag indicating if the solar field is active", repr=True)
    ts_active: bool = Field(False, title="hx_active", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                            description="Flag indicating if the heat exchanger is transfering heat from solar field to thermal storage",
                            repr=True)
    # default_penalty: float = Field(1e6, title="penalty", json_schema_extra={"var_type": ModelVarType.timeseries, "units": "u.m."}, ge=0,
    #                                description="Default penalty for undesired states or conditions", repr=False,
    #                              )

    Jtotal: float = Field(None, title="Jtotal", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWe"},
                          description="Total electrical power consumption (kWe)")
    total_cost: float = Field(None, title="Total cost", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "u.m./h"},
                              description="Total cost (u.m./h)")
    total_income: float = Field(None, title="Total income", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "u.m./h"},
                                description="Total income (u.m./h)")
    net_profit: float = Field(None, title="Net profit", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "u.m."},
                                description="Net profit (u.m.)")
    net_loss: float = Field(None, title="Net loss", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "u.m."},
                                description="Net loss (u.m.)")


    ## Finite State Machine (FSM) states
    med_vacuum_state: MedVacuumState = Field(MedVacuumState.OFF, title="MEDvacuum,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                             description="Input. MED vacuum system state")
    med_state: MedState = Field(MedState.OFF, title="MED,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                description="Input/Output. MED state. It can be used to define the MED initial state, after it's always an output")
    sf_state: SolarFieldState = Field(SolarFieldState.IDLE, title="SF,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                      description="Input/Output. Solar field state. It can be used to define the Solar Field initial state, after it's always an output")
    ts_state: ThermalStorageState = Field(ThermalStorageState.IDLE, title="TS,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                          description="Input/Output. Thermal storage state. It can be used to define the Thermal Storage initial state, after it's always an output")
    sf_ts_state: SfTsState = Field(SfTsState.IDLE, title="SF_TS,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                     description="Output. Solar field with thermal storage state")
    current_state: SolarMedState = Field(None, title="state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                          description="Output. Current state of the SolarMED system")

    # Private attributes
    _water_props: tuple[w_props, w_props] = PrivateAttr(None) # Water properties objects used accross the model
    _MedModel: MedModel = PrivateAttr(None) #= Field(None, repr=False, description="MATLAB MED model instance")
    _med_fsm: MedFsm = PrivateAttr(None) # Finite State Machine object for the MED system. Should not be accessed/manipulated directly
    _sf_ts_fsm: SolarFieldWithThermalStorageFsm = PrivateAttr(None) # Finite State Machine object for the Solar Field with Thermal Storage system. Should not be accessed/manipulated directly
    _fmp: FixedModelParameters = PrivateAttr(None)  # Just a short alias for fixed_model_params
    _mp: FixedModelParameters = PrivateAttr(None)  # Just a short alias for model_params
    # _created_at: datetime = PrivateAttr(default_factory=datetime.datetime.now) # Should not be accessed/manipulated directly

    model_config = ConfigDict(
        validate_assignment=True,  # So that fields are validated, not only when created, but every time they are set
        arbitrary_types_allowed=True,
        extra='forbid',
        protected_namespaces = ()
    )

    @computed_field(title="Export fields df", description="Fields to export into a dataframe", json_schema_extra={"var_type": None})
    @property
    def export_fields_df(self) -> list[str]:
        # Fields to export into a dataframe
        # Choose variable to export based on explicit property, need to be single variables
        fields = []
        for fld in self.model_fields.keys():
            if 'var_type' not in self.model_fields[fld].json_schema_extra:
                logger.warning(f'Field {fld} has no `json_schema_extra.var_type` set, it should be if it needs to be included in the exports')
            else:
                if self.model_fields[fld].json_schema_extra.get('var_type', None) == ModelVarType.TIMESERIES:
                    field_value = getattr(self, fld)
                    if check_value_single(field_value, fld):
                        fields.append(fld)

        return fields

    @computed_field(title="Export fields model config", description="Fields to export into model parameters dict", json_schema_extra={"var_type": None})
    @property
    def export_fields_config(self) -> list[str]:
        # Fields to export into model parameters dict
        # return [field for field in self.__fields__.keys() if
        #         self.__fields__[field].json_schema_extra.get('var_type', None) == ModelVarType.PARAMETER]

        fields = []
        for fld in self.model_fields.keys():
            if 'var_type' in self.model_fields[fld].json_schema_extra:
                if self.model_fields[fld].json_schema_extra.get('var_type', None) == ModelVarType.PARAMETER:
                    fields.append(fld)
            else:
                logger.warning(
                    f'Field {fld} has no `json_schema_extra.var_type` set, it should be if it needs to be included in the exports')
        return fields
    
    @computed_field(title="Export fields model initialization", description="Initialization fields to export", json_schema_extra={"var_type": None})
    @property
    def export_fields_initial_states(self) -> list[str]:
        # Fields to export into a dataframe
        # Choose variable to export based on explicit property, need to be single variables
        fields = []
        for fld in self.model_fields.keys():
            if 'var_type' not in self.model_fields[fld].json_schema_extra:
                logger.warning(f'Field {fld} has no `json_schema_extra.var_type` set, it should be if it needs to be included in the exports')
            elif self.model_fields[fld].json_schema_extra.get('var_type', None) == ModelVarType.INITIAL_STATE:
                fields.append(fld)

    @field_validator("qmed_s_sp", "qmed_f_sp", "qts_src_sp", "qsf_sp", "qmed_c")
    @classmethod
    def validate_flow(cls, flow: float, info: ValidationInfo) -> PositiveFloat:
        field_name = info.field_name.removesuffix("_sp")
        if "med" in field_name:
            grp_name = "med"
        elif "sf" in field_name:
            grp_name = "sf"
        elif "ts" in field_name:
            grp_name = "ts"
        else:
            raise ValueError(f"Field {field_name} doesn't belong to any model component")
        component_fmp: FixedModelParameters = getattr(info.data["fixed_model_params"], grp_name)
        range_ = (getattr(component_fmp, f"{field_name}_min"), 
                  getattr(component_fmp, f"{field_name}_max"))

        return within_range_or_zero_or_max(flow, range=range_, var_name=info.field_name)
    
    @field_validator("Tmed_s_in_sp")
    @classmethod
    def validate_Tmed_s_in(cls, Tmed_s_in: float, info: ValidationInfo) -> float:
        # Lower limit set by pre-defined operational limit, if lower -> 0
        # Upper bound, take the lower between the hot tank top temperature and the pre-defined operational limit
        fmp: FixedModelParameters = info.data["fixed_model_params"]
        
        return within_range_or_zero_or_max(
            Tmed_s_in, range=(fmp.med.Tmed_s_min,
                              np.min([fmp.med.Tmed_s_max, info.data["Tts_h"][0]]),),
            var_name=info.field_name
        )

    @field_validator("Tmed_c_out_sp")
    @classmethod
    def validate_Tmed_c_out(cls, Tmed_c_out: float, info: ValidationInfo) -> float:
        # The upper limit is not really needed, its value is an output restrained already by mmed_c lower bound
        
        return within_range_or_min_or_max(Tmed_c_out, range=(info.data["Tmed_c_in"], np.inf),
                                          var_name=info.field_name)

    @field_serializer("fixed_model_params")
    def serialize_fixed_model_params(self, value: FixedModelParameters) -> dict:
        return asdict(value)
    
    @field_serializer("actuators_consumptions")
    def serialize_actuators_consumptions(self, value: ActuatorsMaping) -> dict:
        return {k: {var: actuator.id for var, actuator in v.items()} for k, v in asdict(value).items()}

    @field_serializer("med_vacuum_state")
    def serialize_med_vacuum_state(self, value: MedVacuumState) -> int:
        return value.value

    def model_post_init(self, ctx) -> None:
        """ Post initialization method, called after the model is created """
        # Aliases
        self._fmp = self.fixed_model_params
        self._mp  = self.model_params

        if self.sample_time > 800 and self.use_models:
            raise RuntimeWarning("Sample time is too high for the time-dependent models (i.e. solar field). "
                                 "This is likely to cause unfeasibilities in the model evaluation")

        if not self.use_models and not self.use_finite_state_machine:
            raise ValueError("At least one of `use_models` or `use_finite_state_machine` must be set to True")

        if self.use_models:
            # Initialize the MATLAB engine
            # self.init_matlab_engine()
            self._MedModel = MedModel(param_array=self._fmp.med.param_array)

        # Make sure thermal storage state is a numpy array
        self.Tts_h = np.array(self.Tts_h, dtype=float)
        self.Tts_c = np.array(self.Tts_c, dtype=float)

        # Validate that there are as many values of vacuum consumption as states
        assert len(self.Jvacuum) == len(MedVacuumState), "The number of vacuum consumption values must be equal to the number of vacuum states"

        # Validate and initialize actuators
        # if set(['ts', 'med', 'sf']) != set(self.actuators_consumptions.keys()):
            # raise ValueError(f"Actuators consumptions must have keys for 'ts', 'med', and 'sf', use an empty dict if there are no actuators for a category. Example: {default_actuators_consumptions_map}")
        for actuator_category in self.actuators_consumptions.__dict__.values():
            for var_id, actuator in actuator_category.items():
                if isinstance(actuator, str):
                    actuator_category[var_id] = Actuator(id=actuator)
                elif isinstance(actuator, dict):
                    actuator_category[var_id] = Actuator(**actuator)
                elif not isinstance(actuator, Actuator):
                    raise ValueError(f"Actuator {actuator} is not a valid actuator, should an id of supported actuator, a dict to initialize an Actuator instance, or an Actuator instance")

        # Initialize outlet temperature to the provided inlet initial temperature
        self.Tsf_in = self.Tsf_in_ant.take(-1)
        self.Tsf_out_ant = self.Tsf_in

        # Make a list of field names that are of type numeric (int, float, etc)
        # self.export_fields = [field for field in self.__fields__.keys() if isinstance(getattr(self, field), (int, float))]

        if self.use_finite_state_machine:

            # initial_sf_ts = SfTsState(str(self.sf_state.value) + str(self.ts_state.value))
            # initial_sf_ts = get_sfts_state(sf_state = self.sf_state, 
            #                                ts_state = self.ts_state)
            self.current_state = self.get_state()

            self._sf_ts_fsm: SolarFieldWithThermalStorageFsm = SolarFieldWithThermalStorageFsm(
                name='SF-TS', 
                initial_state=self.sf_ts_state, 
                sample_time=self.sample_time,
                params=self.fsms_params.sf_ts,
                internal_state=self.fsms_internal_states.sf_ts,
            )
            self._med_fsm: MedFsm = MedFsm(
                name='MED', 
                initial_state=self.med_state,
                sample_time=self.sample_time, 
                params=self.fsms_params.med,
                internal_state=self.fsms_internal_states.med,
            )
            
        if self.resolution_mode == 'constant-water-props':
            self._water_props: tuple[w_props, w_props] = (
                w_props(P=0.2, T=90 + 273.15), # P=2 bar  -> 0.2MPa, T in K, average working temperature of hot tank
                w_props(P=0.2, T=65 + 273.15)  # P=2 bar  -> 0.2MPa, T in K, average working temperature of cold tank
            )

        logger.info(f'''
        SolarMED model initialized with: 
            - Evaluating models: {self.use_models}
            - Evaluating finite state machines: {self.use_finite_state_machine}
            - Resolution mode: {self.resolution_mode}
            - On limits violation policy: {self.on_limits_violation_policy}
            - Sample time: {self.sample_time} s
            - MED actuators: {[actuator.id for actuator in self.actuators_consumptions.med.values()]}
            - Solar field actuators: {[actuator.id for actuator in self.actuators_consumptions.sf.values()]}
            - Thermal storage actuators: {[actuator.id for actuator in self.actuators_consumptions.ts.values()]}
            - Model parameters: {self.model_params}
            - Fixed model parameters: {self.fixed_model_params}
            - FSM parameters: {self.fsms_params},
            - FSM initial internal state: {self.fsms_internal_states}
            - Environment parameters: {self.env_params}
        ''')

    def step(
            self,
            qts_src: float,  # Thermal storage decision variables
            qsf: float, # Solar field decision variables
            qmed_s: float, qmed_f: float, Tmed_s_in: float, Tmed_c_out: float,  # MED decision variables
            Tmed_c_in: float, Tamb: float, I: float, wmed_f: float = None,  # Environment variables
            med_vacuum_state: int | MedVacuumState = 2,  # Optional, to provide the MED vacuum state (OFF, LOW, HIGH)
            compute_fitness: bool = False,
    ) -> None:

        """
        Update model outputs given current environment variables and decision variables

            Inputs:
                - Decision variables
                    MED
                    ---------------------------------------------------
                    + qmed_s (m³/h): Heat source flow rate
                    + qmed_f (m³/h): Feed water flow rate
                    + Tmed_s_in (ºC): Heat source inlet temperature
                    + Tmed_c_out (ºC): Cooling water outlet temperature

                    THERMAL STORAGE
                    ---------------------------------------------------
                    + qts_src (m³/h): Thermal storage heat source flow rate

                    SOLAR FIELD
                    ---------------------------------------------------
                    + Tsf_out (ºC): Solar field outlet temperature

                - Environment variables
                    + Tmed_c_in (ºC): Seawater temperature
                    + wmed_f (g/kg): Seawater salinity (optional)
                    + Tamb (ºC): Ambient temperature
                    + I (W/m²): Solar irradiance

        """

        self.current_sample += 1

        # Process inputs
        # Most of the validation is now done in the class definition
        if wmed_f is not None:
            self.wmed_f = wmed_f

        # Environment
        self.Tamb = Tamb
        self.I = I
        self.Tmed_c_in = Tmed_c_in

        # MED
        self.qmed_s_sp = qmed_s #if qmed_s is not None else self.qmed_s # Use the previous value
        self.qmed_f_sp = qmed_f
        self.Tmed_s_in_sp = Tmed_s_in
        self.Tmed_c_out_sp = Tmed_c_out
        self.med_vacuum_state = med_vacuum_state

        # Thermal storage
        self.qts_src_sp = qts_src

        # Solar field
        self.qsf_sp = qsf
        # logger.info(f"{self.qsf_sp=}")
        
        # Initialize variables
        self.penalty = 0

        # Set operating mode
        if self.use_finite_state_machine:
            # Save FSMs before updating them
            # sf_ts_fsm0 = copy.deepcopy(self._sf_ts_fsm)
            # med_fsm0 = copy.deepcopy(self._med_fsm)

            self._sf_ts_fsm.step(inputs=SfTsFsmInputs(sf_active= self.qsf_sp, ts_active= self.qts_src_sp))
            self._med_fsm.step(
                inputs=MedFsmInputs(
                    med_active= all(np.array([self.qmed_s_sp, self.qmed_f_sp, self.Tmed_s_in_sp, self.Tmed_c_out_sp]) > 0), 
                    med_vacuum_state=self.med_vacuum_state
                )
            )
                # qmed_s=self.qmed_s_sp, qmed_f=self.qmed_f_sp, Tmed_s_in=self.Tmed_s_in_sp,
                # Tmed_c_out=self.Tmed_c_out_sp,
            self.update_current_state()

            # If the finite state machines are used, they need to set the values of: sf_active, ts_active and med_active
            # before evaluating the step
            self.med_active = False if self.med_state != MedState.ACTIVE else True
            self.sf_active = False if self.sf_state != SolarFieldState.ACTIVE else True
            self.ts_active = False if self.ts_state != ThermalStorageState.ACTIVE else True
        else:
            # Check MED state
            self.med_active = True if self.qmed_s_sp > 0 and self.qmed_f_sp > 0 and self.Tmed_s_in_sp > 0 and self.Tmed_c_out_sp > 0 else False
            # Check solar field state
            self.sf_active = True if self.qsf_sp > 0 else False
            # Check heat exchanger state / thermal storage state
            self.ts_active = True if self.qts_src_sp > 0 else False
            
            self.med_state: MedState = MedState.ACTIVE if self.med_active else MedState.OFF 
            self.sf_state: SolarFieldState = SolarFieldState(self.sf_active)
            self.ts_state: ThermalStorageState = ThermalStorageState(self.ts_active) 
            self.sf_ts_state = get_sfts_state(sf_state=self.sf_state, ts_state=self.ts_state)

            self.update_current_state()
        logger.debug(f"SolarMED state after inputs validation: {self.current_state}")
        # After the validation, variables are either zero or within the limits (>0),
        # based on this, the step method in the individual state machines are called

        # Solve model for current step
        self.solve_step()
        
        if compute_fitness:
            self.evaluate_fitness_function()

        # Re-evaluate FSM once the models have been solved
        # Do we really want to be doing this?
        # Sporadic invalid inputs might change the operating mode which takes a long time to recover
        # if self.use_finite_state_machine and self.use_models:
        #     self._sf_ts_fsm = sf_ts_fsm0
        #     self._med_fsm = med_fsm0
        #
        #     self._sf_ts_fsm.step(Tsf_out=self.Tsf_out, qts_src=self.qts_src)
        #     self._med_fsm.step(qmed_s=self.qmed_s, qmed_f=self.qmed_f, Tmed_s_in=self.Tmed_s_in,
        #                        Tmed_c_out=self.Tmed_c_out, med_vacuum_state=self.med_vacuum_state)
        #
        #     self.update_current_state()
        #     logger.debug(f"SolarMED state after step evaluation: {self.current_state}")

    def solve_step(self) -> None:
        """
        Every individual model `solve` method considers either the active state of the component
        or the use_models logical variable to return its outputs. If any is false they are not evaluated
        but return either their values, or the values defined by the finite state machine. 
        """
        
        if self.resolution_mode != "constant-water-props":
            self._water_props: tuple[w_props, w_props] = (
                w_props(P=0.2, T=self.Tts_h[0] + 273.15), # P=2 bar  -> 0.2MPa, T in K, temperature of hot tank
                w_props(P=0.2, T=self.Tts_c[-1]+ 273.15)  # P=2 bar  -> 0.2MPa, T in K, temperature of cold tank
            )
            
        # 1st. MED
        self.solve_MED()

        # 2nd. Three-way valve
        self.solve_3wv()

        # 3rd. Solve solar field, heat exchanger and thermal storage subproblem
        ## If both the solar field is active and the thermal storage is being recharged
        ## Then the system is coupled, solve coupled subproblem
        if self.ts_active and self.sf_active:
            self.solve_coupled_subproblem()

        ## Otherwise, solar field and thermal storage are decoupled. Solve each system independently
        else:
            # The order is important, hex depends on Tsf_out which is set by the solar field
            # Solve solar field
            # self.qsf = self.solve_solar_field_inverse(Tsf_out=self.Tsf_out_sp)
            self.solve_solar_field()
            
            # Heat exchanger
            self.solve_heat_exchanger()
            
            # Solve thermal storage
            # self.Tts_h_in = self.Tts_c[-1] # Hot tank inlet temperature is the bottom temperature of the cold tank
            self.solve_thermal_storage()

            

            # Since they are decoupled, necessarily the outlet of the solar field becomes its inlet
            # self.Tsf_in = self.Tsf_out + 3  # Tener en cuenta de que esto no tiene sentido físico, pero se observa en los datos experimentales

        # Update prior values
        ## Solar field
        self.Tsf_out_ant = self.Tsf_out

        self.qsf_ant = np.roll(self.qsf_ant, -1)  # Shift all elements to the left
        self.qsf_ant[-1] = self.qsf  # Add the new value at the end

        self.Tsf_in_ant = np.roll(self.Tsf_in_ant, -1)  # Shift all elements to the left
        self.Tsf_in_ant[-1] = self.Tsf_in  # Add the new value at the end

        # Calculate additional outputs
        self.calculate_med_aux_outputs()
        self.calculate_sf_aux_outputs()
        self.calculate_ts_aux_outputs()
        self.calculate_hx_aux_outputs()
        # Total variables
        self.Jtotal = self.Jmed + self.Jts + self.Jsf
        self.Jsf_ts = self.Jsf + self.Jts

    def get_state(self, mode: Literal["default", "human_readable"] = 'default') -> SolarMedState | str:
        # state_code = self.generate_state_code(self._sf_ts_fsm.state, self._med_fsm.state)

        state_code = str(self.sf_state.value) + str(self.ts_state.value) + str(self.med_state.value)

        if mode == 'human_readable':
            state_str = SolarMedState(state_code).name
            # Replace _ by space and make everything minusculas
            state_str =  state_str.replace('_', ' ').lower()
            # Replace ts to TS, sf to SF and med to MED
            state_str = state_str.replace('ts', 'TS').replace('sf', 'SF').replace('med', 'MED')
            
            return state_str
        
        # Else
        return SolarMedState(state_code)

    def update_internal_states(self) -> None:
        # Sync internal states from fsm classes in fsm
        self.fsms_internal_states: FsmInternalState = FsmInternalState(
            med=self._med_fsm.internal_state,
            sf_ts=self._sf_ts_fsm.internal_state
        )
        self.med_state: MedState = self._med_fsm.get_state()
        self.sf_ts_state: SfTsState = self._sf_ts_fsm.get_state()
        # self.sf_state = SolarFieldState(int(self.sf_ts_state.value[0]))
        # self.ts_state = ThermalStorageState(int(self.sf_ts_state.value[1]))
        self.sf_state, self.ts_state = get_sf_ts_individual_states(sfts_state=self.sf_ts_state)

    def update_current_state(self) -> None:
        if self.use_finite_state_machine:
            self.update_internal_states()
        self.current_state = self.get_state()


    """
    Implementation details:
    
    - The inputs that determine the states of the different compoenents are *implicit*. An
    example: for the solar field there is not an explicit input such as `sf_active`, but it's the solar field flow
    that it's either zero (equivalent to the explicit `sf_active=False`) or has a value above zero.
    
    --
    
    Implementation "good" practices: 
    
    - solve_* methods only require as inputs, variables that might change within
    the model evaluation, to make sure they use the adequate value. They should try
    to have as few inputs as possible, to simplify their interface.
    
    - Decision variables, are named with the `sp` suffix.
    Since some of the decision variables are in reality process outputs (e.g. Tmed_c_out),
    once the particular subsystem is solved, a variable without the `sp` suffix is updated,
    reflecting the actual value of the decision variable. This is also the case
    for the rest of variables after validation / state machine evaluation.
    
    - Each _solve method needs to have a update_attrs argument, that if true, updates the
    attributes of the model with the calculated values, to simplify the interface.
    """

    def solve_MED(self, update_attrs: bool = True) -> tuple[float, float, float, float, float, float, float, float]:
        """ 
        Solve the MED model.
        """
        
        def override_model():
            
            # Process outputs
            qmed_d = 0
            qmed_c = 0
            qmed_b = 0
            Tmed_s_out = 0

            # Overiden decision variables
            qmed_s = 0
            qmed_f = 0
            Tmed_s_in = 0
            Tmed_c_out = self.Tmed_c_in # Or 0?
            
            if self.use_finite_state_machine:
                # FSM overrides
                st_cond = None
                if self.med_state == MedState.STARTING_UP:
                    st_cond = self.fsms_params.med.startup_conditions
                    
                elif self.med_state == MedState.SHUTTING_DOWN:
                    st_cond = self.fsms_params.med.shutdown_conditions
                    
                if st_cond is not None:
                    qmed_s = st_cond.qmed_s
                    qmed_f = st_cond.qmed_f
                    qmed_b = st_cond.qmed_b
                    qmed_c = st_cond.qmed_c
                    Tmed_s_in = st_cond.Tmed_s_in
            
            # Assign outputs    
            if update_attrs:
                self.qmed_s = qmed_s
                self.qmed_f = qmed_f
                self.Tmed_s_in = Tmed_s_in
                self.Tmed_c_out = Tmed_c_out
                self.qmed_d = qmed_d
                self.qmed_c = qmed_c
                self.qmed_b = qmed_b
                self.Tmed_s_out = Tmed_s_out
            
            return qmed_s, qmed_f, Tmed_s_in, Tmed_c_out, qmed_d, qmed_c, qmed_b, Tmed_s_out

        # Alias
        Tmed_c_out = self.Tmed_c_out_sp 
        Tmed_s_in = self.Tmed_s_in_sp
        qmed_s = self.qmed_s_sp
        qmed_f = self.qmed_f_sp
        Tmed_c_in = self.Tmed_c_in

        Tmed_c_out0 = Tmed_c_out
        med_model_solved: bool = False

        if not self.med_active or not self.use_models:
            return override_model()

        # MsIn = self._matlab.double([qmed_s / 3.6], size=(1, 1))  # m³/h -> L/s
        # TsinIn = self._matlab.double([Tmed_s_in], size=(1, 1))
        # MfIn = self._matlab.double([qmed_f], size=(1, 1))
        # TcwinIn = self._matlab.double([Tmed_c_in], size=(1, 1))
        # op_timeIn = self._matlab.double([0], size=(1, 1))
        # wf=wmed_f # El modelo sólo es válido para una salinidad así que ni siquiera
        # se considera como parámetro de entrada

        while not med_model_solved and (Tmed_c_in < Tmed_c_out < Tmed_s_in):

            # TcwoutIn = self._matlab.double([Tmed_c_out], size=(1, 1))
            # # try:
            # qmed_d, Tmed_s_out, qmed_c, _, _ = self._MED_model.MED_model(
            #     MsIn,  # L/s
            #     TsinIn,  # ºC
            #     MfIn,  # m³/h
            #     TcwoutIn,  # ºC
            #     TcwinIn,  # ºC
            #     op_timeIn,  # hours
            #     nargout=5
            # )
            inputs = np.array([[
                qmed_s,      # m³/h
                Tmed_s_in,   # ºC
                qmed_f,      # m³/h
                Tmed_c_out,  # ºC
                Tmed_c_in    # ºC
            ]])
            qmed_d, Tmed_s_out, qmed_c = self._MedModel.predict(inputs)[0, :]

            if qmed_c > self._fmp.med.qmed_c_max:
                Tmed_c_out += 1
            elif qmed_c < self._fmp.med.qmed_c_min:
                Tmed_c_out -= 1
            else:
                med_model_solved = True

        if not med_model_solved:
            self.med_active = False
            logger.warning('MED is not active due to unfeasible operation in the condenser, setting all MED outputs to 0')

            return override_model()

        # Else
        if abs(Tmed_c_out0 - Tmed_c_out) > 0.1:
            logger.debug(
                f"MED condenser flow was out of range, changed outlet temperature from {Tmed_c_out0:.2f} to {Tmed_c_out:.2f}"
            )

        ## Brine flow rate
        qmed_b = qmed_f - qmed_d  # m³/h
        
        # Assign outputs    
        if update_attrs:
            self.qmed_s = qmed_s
            self.qmed_f = qmed_f
            self.Tmed_s_in = Tmed_s_in
            self.Tmed_c_out = Tmed_c_out
            self.qmed_d = qmed_d
            self.qmed_c = qmed_c
            self.qmed_b = qmed_b
            self.Tmed_s_out = Tmed_s_out

        return qmed_s, qmed_f, Tmed_s_in, Tmed_c_out, qmed_d, qmed_c, qmed_b, Tmed_s_out

    def solve_3wv(self, update_attrs: bool = True) -> tuple[float, float, float, float]:
        
        if not self.use_models or not self.med_active:
            q3wv_src = 0.0
            R3wv = np.nan
            qts_dis = 0.0
            q3wv_dis = 0.0
            
            if update_attrs:
                self.q3wv_src = q3wv_src
                self.R3wv = R3wv
                self.qts_dis = qts_dis
                self.q3wv_dis = q3wv_dis
                
            return q3wv_src, R3wv, qts_dis, q3wv_dis 
        
        q3wv_src, R3wv = three_way_valve_model(
            Mdis=self.qmed_s, 
            Tsrc=self.Tts_h[0], 
            Tdis_in=self.Tmed_s_in, 
            Tdis_out=self.Tmed_s_out
        )
        qts_dis = self.q3wv_src
        q3wv_dis = self.qmed_s
            
        if update_attrs:
            self.q3wv_src = q3wv_src
            self.R3wv = R3wv
            self.qts_dis = qts_dis
            self.q3wv_dis = q3wv_dis
            
        return q3wv_src, R3wv, qts_dis, q3wv_dis
            
    def solve_thermal_storage(self, update_attrs: bool = True) \
            -> tuple[np.ndarray[conHotTemperatureType], np.ndarray[conHotTemperatureType], float]:
        """ Solve the thermal storage model """
        
        if not self.use_models:
            Tts_h = np.array([np.nan]*len(self.Tts_h))
            Tts_c = np.array([np.nan]*len(self.Tts_c))
            qts_src = self.qts_src_sp
            
            if update_attrs:
                self.Tts_h = Tts_h
                self.Tts_c = Tts_c
                self.qts_src = qts_src
            
            return Tts_h, Tts_c, qts_src
                    
        # if self.use_finite_state_machine:
        #     if self.sf_ts_state == SfTsState.RECIRCULATING_TS:
        #         qts_src = self.fsms_params.startup_conditions.qts_src
        #     elif not self.ts_state:
        #         qts_src = self.fsms_params.shutdown_conditions.qts_src
        #     else:
        #         qts_src = self.qts_src_sp
            
        # Else
        Tts_h, Tts_c = thermal_storage_two_tanks_model(
            Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c,  # [ºC], [ºC]
            Tt_in=self.Tts_h_in,  # ºC
            Tb_in=self.Tmed_s_out,  # ºC
            Tamb=self.Tamb,  # ºC

            qsrc=self.qts_src_sp,  # m³/h
            qdis=self.qts_dis,  # m³/h

            model_params=self.model_params.ts,
            fixed_model_params=self._fmp.ts,
            sample_time=self.sample_time, 
            # Tmin=self.fixed_model_params.sf.Tmin,  # seg, ºC
            water_props=self._water_props
        )

        if update_attrs:
            self.Tts_h = Tts_h
            self.Tts_c = Tts_c
            self.qts_src = self.qts_src_sp
    
        return Tts_h, Tts_c, 
        
    def solve_heat_exchanger(self, return_epsilon: bool = False, update_attrs: bool = True) -> tuple[float, float, float] | tuple[float, float]:
        """ Solve the heat exchanger model """
        
        if not self.use_models:
            if update_attrs:
                self.Thx_p_out = np.nan
                self.Thx_s_out = np.nan
                self.Tsf_out = self.Thx_p_out
                self.Tts_h_in = self.Thx_s_out
                self.epsilon_hx = np.nan
                
            return (np.nan) * 3 if return_epsilon else (np.nan) * 2
        
        Thx_p_out, Thx_s_out, epsilon = heat_exchanger_model(
            Tp_in=self.Tsf_out,  # Solar field outlet temperature (ºC)
            Ts_in=self.Tts_c[-1],  # Cold tank bottom temperature (ºC)

            qp=self.qsf_sp,  # Solar field flow rate (output, m³/h)
            qs=self.qts_src_sp,  # Thermal storage charge flow rate (output, m³/h)

            Tamb=self.Tamb,  # Ambient temperature (ºC)

            model_params=self.model_params.hex,
            Tmin=self.fixed_model_params.sf.Tmin,
            water_props=self._water_props,
            return_epsilon=True # Maravilloso
        )

        if update_attrs:
            self.Thx_p_out = Thx_p_out
            self.Thx_s_out = Thx_s_out
            self.Tsf_in = self.Thx_p_out
            self.Tts_h_in = Thx_s_out
            self.epsilon_hx = epsilon

        if return_epsilon:
            return Thx_p_out, Thx_s_out, epsilon
        else:
            return Thx_p_out, Thx_s_out

    def solve_solar_field(self, update_attrs: bool = True) -> tuple[float, float]:
        """
        Solve the solar field model
        Make sure to set `Tsf_out_ant` to the prior `Tsf_out` value before calling this method
        """

        if not self.use_models:
            Tsf_out = np.nan
            qsf = self.qsf_sp
            
            if update_attrs:
                self.Tsf_out = Tsf_out
                self.qsf = qsf
                
            return Tsf_out, qsf
        
        # if self.use_finite_state_machine:
            # if self.sf_ts_state == SfTsState.HEATING_UP_SF:
            #     qsf = self.fsms_params.startup_conditions.qsf
            # elif not self.sf_state:
            #     qsf = self.fsms_params.shutdown_conditions.qsf
            # else:
                # qsf = self.qsf_sp
            
        # Else
        qsf = np.append(self.qsf_ant, self.qsf_sp)
        Tsf_in = np.append(self.Tsf_in_ant, self.Tsf_in)

        Tsf_out = solar_field_model(
            Tin=Tsf_in, # From current value, up to array start
            q=qsf, # From current value, up to array start
            I=self.I, Tamb=self.Tamb, Tout_ant=self.Tsf_out_ant,

            model_params=self.model_params.sf,
            fixed_model_params=self.fixed_model_params.sf,
            sample_time=self.sample_time, consider_transport_delay=True,
            water_props=self._water_props[0]
        )
        
        if update_attrs:
            self.Tsf_out = Tsf_out
            self.qsf = qsf[-1]
        
        return Tsf_out, qsf[-1]

    def solve_coupled_subproblem(self, update_attrs: bool = True) -> tuple[float, float, float, np.ndarray, np.ndarray, float, float]:
        """
        Solve the coupled subproblem of the solar MED system
        """
        
        if not self.use_models:
            Tsf_in = np.nan
            Tsf_out = np.nan
            Tts_h_in = np.nan
            Tts_h = np.array([np.nan]*len(self.Tts_h))
            Tts_c = np.array([np.nan]*len(self.Tts_c))
            qsf = self.qsf_sp
            qts_src = self.qts_src_sp
            
            if update_attrs:
                self.Tsf_in = Tsf_in
                self.Tsf_out = Tsf_out
                self.Tts_h_in = Tts_h_in
                self.Tts_h = Tts_h
                self.Tts_c = Tts_c
                self.qsf = qsf
                self.qts_src = qts_src
            
            return Tsf_in, Tsf_out, Tts_h_in, Tts_h, Tts_c, qsf, qts_src
        
        qsf = np.append(self.qsf_ant, self.qsf_sp)

        Tsf_in, Tsf_out, Tts_h_in, Tts_h, Tts_c = heat_generation_and_storage_subproblem(
            # Solar field
            qsf=qsf,
            Tsf_in_ant = self.Tsf_in_ant,
            Tsf_out_ant= self.Tsf_out_ant,
            
            # Thermal storage
            qts_src=self.qts_src_sp, 
            qts_dis=self.qts_dis,
            Tts_b_in=self.Tmed_s_out, 
            Tts_h= self.Tts_h, 
            Tts_c= self.Tts_c, 
            
            # Environment
            Tamb=self.Tamb, I=self.I,  
            
            # Parameters
            model_params=self.model_params,
            fixed_model_params=self.fixed_model_params,
            sample_time = self.sample_time,
            water_props = self._water_props,
            problem_type = "1p2x",
            solver = "scipy",
            solver_method = "dogbox"
        )
        
        if update_attrs:
            self.Tsf_in = Tsf_in
            self.Tsf_out = Tsf_out
            self.Tts_h_in = Tts_h_in
            self.Tts_h = Tts_h
            self.Tts_c = Tts_c
            self.qsf = self.qsf_sp
            self.qts_src = self.qts_src_sp

        return Tsf_in, Tsf_out, Tts_h_in, Tts_h, Tts_c, self.qsf_sp, self.qts_src_sp

    def calculate_ts_aux_outputs(self) -> None:
        
        if not self.use_models:
            # Source
            self.Pth_ts_src = 0
            self.Jts = 0
            # Discharge
            self.Pth_ts_dis = 0
            self.Tts_h_out = 0
            self.Tts_c_in = 0
            
            return
        
        # Else
        if not self.ts_active:
            self.Pth_ts_src = 0
            self.Jts = 0
        else:
            # Pth_ts_out, Pth_ts_in, Jts_e
            # w_props_ts_in = w_props(P=0.16, T=(self.Tts_h_in + self.Tts_c[-1]) / 2 + 273.15)
            self.Pth_ts_src = self.qts_src * self._water_props[0].rho * (
                        self.Tts_h_in - self.Tts_c[-1]) * self._water_props[0].cp / 3600  # kWth
                        
            # Electrical consumption
            # Jts = 0
            # Jts += np.sum(
            #     [
            #         actuator.calculate_power_consumption(flow)
            #         for actuator, flow in zip(self.ts_actuators, [self.qts_src])
            #     ]
            # )
            self.Jts = 0
            self.Jts += np.sum(
                [actuator(getattr(self, var_id)) for var_id, actuator in self.actuators_consumptions.ts.items()]
            )
            
        if not self.med_active:
            self.Pth_ts_dis = 0
            self.Tts_h_out = 0
            self.Tts_c_in = 0
        else:
            # w_props_ts_out = w_props(P=0.16, T=(self.Tmed_s_out + self.Tts_h[1]) / 2 + 273.15)
            self.Pth_ts_dis = self.qts_dis * self._water_props[0].rho * (
                    self.Tts_h[1] - self.Tmed_s_out) * self._water_props[0].cp / 3600  # kWth

            # TODO: If there is an alternative thermal storage configuration, the index needs to be the one where the extraction is done
            self.Tts_h_out = self.Tts_h[0]
            self.Tts_c_in = self.Tmed_s_out

    def calculate_sf_aux_outputs(self) -> None:
        
        if not self.sf_active or not self.use_models:
            self.Pth_sf = 0
            self.Jsf = 0
            self.SEC_sf = np.nan
            
            return
        
        # Else
        # w_props_sf = w_props(P=0.16, T=(self.Tsf_in + self.Tsf_out) / 2 + 273.15)
        self.Pth_sf = self.qsf * self._water_props[0].rho * (self.Tsf_out - self.Tsf_in) * self._water_props[0].cp / 3600  # kWth
        
        # Electrical consumption
        self.Jsf = 0
        self.Jsf += np.sum(
            [actuator(getattr(self, var_id)) for var_id, actuator in self.actuators_consumptions.sf.items()]
        )
        self.SEC_sf = self.Jsf / self.Pth_sf if self.Pth_sf > 0 else np.nan  # kWhe/kWth
        
    def calculate_hx_aux_outputs(self) -> None:
        
        if not self.use_models:
            return
        
        # Else
        # Copied variables for the heat exchanger
        self.Thx_p_in = self.Tsf_out
        self.Thx_p_out = self.Tsf_in
        self.Thx_s_in = self.Tts_c[-1]
        self.Thx_s_out = self.Tts_h_in
        self.qhx_p = self.qsf
        self.qhx_s = self.qts_src
        self.Pth_hx_p = self.Pth_sf
        self.Pth_hx_s = self.Pth_ts_src

        self.epsilon_hx = calculate_heat_transfer_effectiveness(
            Tp_in=self.Thx_p_in,
            Tp_out=self.Thx_p_out,
            Ts_in=self.Thx_s_in,
            Ts_out=self.Thx_s_out,
            qp=self.qhx_p,
            qs=self.qhx_s
        )

    def calculate_med_aux_outputs(self) -> None:
        
        def auxiliary_consumption() -> float:
            """ Calculate the auxiliary consumptions of the MED system """
            return self.Jvacuum[self.med_vacuum_state.value]
        
        if not self.use_models:
            self.Jmed = 0
            self.Pth_med = np.nan
            self.STEC_med = np.nan
            self.SEEC_med = np.nan
            
            return
            
        # Electrical consumption
        self.Jmed = 0
        self.Jmed += np.sum(
            [actuator(getattr(self, var_id)) for var_id, actuator in self.actuators_consumptions.med.items()]
        )
        self.Jmed += auxiliary_consumption()

        if not self.med_active:
            self.SEEC_med = np.nan
            self.Pth_med = 0.0
            self.STEC_med = np.nan
        else:
            # Electrical performance
            self.SEEC_med = self.Jmed / self.qmed_d # kWhe/m³
            # Thermal performance
            # w_props_s = w_props(P=0.1, T=(self.Tmed_s_in + self.Tmed_s_out) / 2 + 273.15)
            cp_s = self._water_props[0].cp  # kJ/kg·K
            rho_s = self._water_props[0].rho  # kg/m³
            # rho_d = w_props(P=0.1, T=Tmed_c_out+273.15) # kg/m³
            qmed_s_kgs = self.qmed_s * rho_s / 3600  # kg/s

            self.Pth_med = qmed_s_kgs * (self.Tmed_s_in - self.Tmed_s_out) * cp_s  # kWth
            self.STEC_med = self.Pth_med / self.qmed_d  # kWhth/m³
            
    def evaluate_fitness_function(self,
                                  cost_w: float = None, cost_e: float = None,
                                  cost_type: Literal['economic', 'exergy'] = 'economic',
                                  objective_type: Literal['maximize', 'minimize'] = 'maximize',
                                  ) -> float:

        if cost_type == 'exergy':
            raise NotImplementedError("Exergy cost evaluation is not yet implemented")

        if cost_w is None:
            cost_w = self.env_params.cost_w
        if cost_e is None:
            cost_e = self.env_params.cost_e

        # Economic cost
        self.total_cost = self.Jtotal * cost_e # kWhe * u.m./kWhe -> u.m./h
        self.total_income = self.qmed_d * cost_w # m³/h * u.m./m³ -> u.m./h

        self.net_profit = (self.total_income - self.total_cost) * self.sample_time/3600 # u.m.
        self.net_loss = -self.net_profit # u.m.

        if objective_type not in ['maximize', 'minimize']:
            raise ValueError(f"Objective type {objective_type} not supported, choose one of 'maximize' or 'minimize'")
        elif objective_type == 'maximize':
            return self.net_profit
        
        return self.net_loss

    def to_dataframe(self, df=None, rename_flows: bool = False) -> pd.DataFrame:
        """
        Take all fields that are of type `timeseries` (property `var_type`) and export them in a pandas dataframe

        Notes:
        - Thermal storage temperatures are expanded following the nomenclature
        - If an existing dataframe is provided, it appends the new data to it
        - Flows can be exported using mass flow rate nomenclature (start with m) or as volumetric flow rates (start
        with q) depending on the `rename_flows` parameter. If set to False (defualt) both options are included.
        """
        # if self._export_fields_df is None:
        #     # The problem with only running it once, is that maybe some desired variable only gets a value (not None),
        #     # not in the first iteration?
        #     # UPDATE: Fixed, now variables that should be exported are explicitely defined
        #
        #     # Choose variables to export based on type
        #     # self._export_fields_df = [field for field in self.__fields__.keys() if
        #     #                          isinstance(getattr(self, field), (int, float, str, bool)) or type(getattr(self, field)) == EnumType]
        #
        #     # Choose variable to export based on explicit property
        #     self._export_fields_df = [field for field in self.__fields__.keys() if
        #                              self.__fields__[field].field_info.json_schema_extra.get('var_type', None) == ModelVarType.timeseries]


        # Create a dataframe from the dictionary
        data = pd.DataFrame(self.model_dump(include=self.export_fields_df, by_alias=True), index=[0])

        # Add the thermal storage temperatures
        data["Tts_h_t"], data["Tts_h_m"], data["Tts_h_b"] = self.Tts_h
        data["Tts_c_t"], data["Tts_c_m"], data["Tts_c_b"] = self.Tts_c

        # Duplicate flow rates to include both m and q options
        if not rename_flows:
            for col in data.columns:
                # logger.debug(f"Processing column {col}")
                # if (re.match(r'mmed*', col) or
                #     re.match(r'qsf*', col) or
                #     re.match(r'mhx*', col) or
                #     re.match(r'm3wv*', col) or
                #     re.match(r'qts*', col)):
                if re.match('m(?!ed)', col):
                    data[f'q{col[1:]}'] = data[col]

        # Rename flows from m* to q*, not med
        if rename_flows:
            data.rename(columns=lambda x: re.sub('^m(?!ed)', 'q', x), inplace=True) # Peligroso

        
        return pd.concat([df, data], ignore_index=True) if df is not None else data

    def model_dump_configuration(self) -> dict:
        """
        Export model instance parameters / configuration
        Returns:
            dict of model parameters
        """
        # if self._export_fields_config is None:
        #     # Choose variable to export based on explicit property
        #     self._export_fields_config = [field for field in self.__fields__.keys() if
        #                                  self.__fields__[field].field_info.json_schema_extra.get('var_type', None) == ModelVarType.parameter]
        # serialize_as_any: see [Serializing with duck-typing](https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing)
        return self.model_dump(include=self.export_fields_config, by_alias=True, serialize_as_any=True)

    def dump_instance(self, reset_samples: bool = False, to_file: Path | str = None) -> dict:
        """
        WIP

        Export an instance of the model on its current state as a dict, that can be directly used to recreate a new
        working (🤞) identical instance.

        Use example:
        ```
            model2 = SolarMEDModel(**model.model_dump_instance())
        ```

        :reset_samples (bool). Whether to reset samples to 0 or keep current values. For the FSMs, if for example
        a counter like `vacuum_started_sample` is not zero, it will be set to zero and the FSM `current_sample` to the
        corresponding shifted value. NOTE: The FSM current sample is different and not in sync with the model
        `current_sample`, which will be set to zero with this argument.

        Returns:
            dict of model instance

        """

        # TODO: How to handle the FSM instances correctly? Just take the current samples? (startup_started_sample, current_sample, etc)
        # if reset_samples:
        #     raise NotImplementedError("Resetting samples is not yet supported")

        # dump = self.model_dump()

        # # Filter out fields whose values are not set (None)
        # dump = {k: v for k, v in dump.items() if v is not None}

        # # Add FSMs
        # dump['_med_fsm'] = pickle.dumps(self._med_fsm)
        # dump['_sf_ts_fsm'] = pickle.dumps(self._sf_ts_fsm)
        

        # Add PID if set
        # if self._pid_sf is not None:
        #     dump['_pid_sf'] = pickle.dumps(self._pid_sf)
        
        
        dump = {
            **self.model_dump(include=self.export_fields_initial_states, by_alias=False, exclude_none=True, exclude_defaults=True),
            **self.model_dump(include=self.export_fields_df, by_alias=False, exclude_none=True, exclude_defaults=True),
            **self.model_dump(include=self.export_fields_config, by_alias=False, exclude_none=True, exclude_defaults=True),
        }
        
        # Computed fields are always included when dumping the model, filter them out
        for field in self.model_computed_fields:
            if field in dump:
                del dump[field]

        if to_file is not None:
            to_file = Path(to_file).with_suffix('.pkl')
            with open(to_file, 'wb') as f:
                pickle.dump(dump, f)

        return dump

    # def init_matlab_engine(self) -> None:
    #     """
    #     Manually initialize the MATLAB MED model, in case it was terminated.
    #     """
    #     # Conditionally import the module
    #     if self._MED_model is None:
    #         import MED_model
    #         import matlab

    #     self._MED_model = MED_model.initialize()
    #     self._matlab = matlab
    #     logger.info('MATLAB engine initialized')

    # def terminate(self) -> None:
    #     """
    #     Terminate the model and free resources. To be called when no more steps are needed.
    #     It just terminates the MATLAB engine, all the data and states are preserved.
    #     """

    #     self._MED_model.terminate()
