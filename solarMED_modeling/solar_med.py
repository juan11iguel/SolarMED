import json
import re
import numpy as np
import pandas as pd
import re
from iapws import IAPWS97 as w_props  # Librería propiedades del agua, cuidado, P Mpa no bar
from loguru import logger
from scipy.optimize import least_squares
from enum import Enum, EnumType
from typing import Any, Literal
import copy
import datetime
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    computed_field,
    ValidationError,
    ValidationInfo,
    PositiveFloat,
    PrivateAttr
)
from pathlib import Path
from simple_pid import PID

from .data_validation import (rangeType, within_range_or_zero_or_max, within_range_or_min_or_max,
                              conHotTemperatureType, check_value_single, conHotTemperatureType_upper_limit)
from .curve_fitting import evaluate_fit
from .solar_field import solar_field_model, solar_field_inverse_model
from .heat_exchanger import heat_exchanger_model, calculate_heat_transfer_effectiveness
from .thermal_storage import thermal_storage_two_tanks_model
from .power_consumption import Actuator, SupportedActuators
from .three_way_valve import three_way_valve_model
from . import MedState, SF_TS_State, ThermalStorageState, SolarFieldState, SolarMED_State, MedVacuumState
from .fsms import MedFSM, SolarFieldWithThermalStorage_FSM

np.set_printoptions(precision=2)
dot = np.multiply
Nsf_max = 100  # Maximum number of steps to keep track of the solar field flow rate, should be higher than the maximum expected delay

class ModelVarType(Enum):
    TIMESERIES = 0
    PARAMETER  = 1

states_sf_ts = [state for state in SF_TS_State]
states_med = [state for state in MedState]


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

    # Limits
    # Important to define first, so that they are available for validation
    ## Flows. Need to be defined separately to validate using `within_range_or_zero_or_max`
    lims_mts_src: rangeType = Field((0.95, 20), title="mts,src limits", json_schema_extra={"units": "m3/h", "var_type": ModelVarType.PARAMETER},
                                    description="Thermal storage heat source flow rate range (m³/h)", repr=False)
    ## Solar field, por comprobar!!
    lims_msf: rangeType = Field((4.7, 14), title="msf limits", json_schema_extra={"units": "m3/h", "var_type": ModelVarType.PARAMETER},
                                description="Solar field flow rate range (m³/h)", repr=False)
    lims_mmed_s: rangeType = Field((30, 48), title="mmed,s limits", json_schema_extra={"units": "m3/h", "var_type": ModelVarType.PARAMETER},
                                   description="MED hot water flow rate range (m³/h)", repr=False)
    lims_mmed_f: rangeType = Field((5, 9), title="mmed,f limits", json_schema_extra={"units": "m3/h", "var_type": ModelVarType.PARAMETER},
                                   description="MED feedwater flow rate range (m³/h)", repr=False)
    lims_mmed_c: rangeType = Field((8, 21), title="mmed,c limits", json_schema_extra={"units": "m3/h", "var_type": ModelVarType.PARAMETER},
                                   description="MED condenser flow rate range (m³/h)", repr=False)

    # Tmed_s_in, límite dinámico
    lims_Tmed_s_in: rangeType = Field((60, 75), title="Tmed,s,in limits", json_schema_extra={"units": "C", "var_type": ModelVarType.PARAMETER},
                                      description="MED hot water inlet temperature range (ºC)", repr=False)
    lims_Tsf_out: rangeType = Field((65, conHotTemperatureType_upper_limit), title="Tsf,out setpoint limits", json_schema_extra={"units": "C", "var_type": ModelVarType.PARAMETER},
                                    description="Solar field outlet temperature setpoint range (ºC)", repr=False)
    ## Common
    lims_T_hot: rangeType = Field((0, conHotTemperatureType_upper_limit), title="Thot* limits", json_schema_extra={"units": "C", "var_type": ModelVarType.PARAMETER},
                                  description="Solar field and thermal storage temperature range (ºC)", repr=False)

    # Parameters
    ## General parameters
    use_models: bool = Field(True, title="use_models", json_schema_extra={"units": "-", "var_type": ModelVarType.PARAMETER},
                             description="Whether to evaluate the models for the components of the SolarMED system, if set to False, `use_finite_state_machines` must be set to True")
    use_finite_state_machine: bool = Field(False, title="FSM enabled", json_schema_extra={"units": "-", "var_type": ModelVarType.PARAMETER},
                                           description="Whether to enable or not the Finite State Machine (FSM) for the SolarMED system")
    sample_time: float = Field(60, description="Sample rate (seg)", title="sample rate",
                               json_schema_extra={"units": "s", "var_type": ModelVarType.PARAMETER})
    resolution_mode: Literal['simple', 'precise'] = Field(..., repr=False, json_schema_extra={"var_type": ModelVarType.PARAMETER},
                                                          description="Mode of solving the model, can either be a simplified but faster version (`simple`)"
                                                                      "or a more precise but slower (`precise`)")
    ## FSM parameters
    vacuum_duration_time: int = Field(5 * 60, title="MED vacuum,duration", json_schema_extra={"units": "s", "var_type": ModelVarType.PARAMETER},
                                      description="Time to generate vacuum (seconds)")
    brine_emptying_time: int = Field(3 * 60, title="MED brine,emptying,duration", json_schema_extra={"units": "s", "var_type": ModelVarType.PARAMETER},
                                     description="Time to extract brine from MED plant (seconds)")
    startup_duration_time: int = Field(1 * 60, title="MED startup,duration", json_schema_extra={"units": "s", "var_type": ModelVarType.PARAMETER},
                                       description="Time to start up the MED plant (seconds)")
    # TODO: Add cooldown times / hystheresis? It would greatly delay/reduce the exponential growth/explosion of possible transitions
    # active_cooldown_time: int = Field(3*60, title="MED active,cooldown", json_schema_extra={"units": "s"},
    #                                   description="Minimum time than the MED system needs to stay active after transitioning to this state (seconds)")
    # off_cooldown_time: int = Field(5*60, title="MED off,cooldown", json_schema_extra={"units": "s"},
    #                                description="Minimum time than the MED system needs to stay off after transitioning to this state (seconds)")

    # Chapuza: Por favor, asegurarse de que aquí se definen en el mimso orden que se usan después al asociarle un caudal
    # mmed_b, mmed_f, mmed_d, mmed_c, mmed_s
    ## MED
    med_actuators: list[Actuator] | list[str] = Field(["med_brine_pump", "med_feed_pump",
                                                       "med_distillate_pump", "med_cooling_pump",
                                                       "med_heatsource_pump"],
                                                      description="Actuators to estimate electricity consumption for the MED",
                                                      title="MED actuators", repr=False, json_schema_extra={"var_type": ModelVarType.PARAMETER})
    ## Thermal storage
    ts_actuators: list[Actuator] | list[str] = Field(["ts_src_pump"], title="Thermal storage actuators", repr=False,
                                                     json_schema_extra={"var_type": ModelVarType.PARAMETER},
                                                     description="Actuators to estimate electricity consumption for the thermal storage")
    UAts_h: list[PositiveFloat] = Field([0.0069818, 0.00584034, 0.03041486], title="UAts,h",
                                        json_schema_extra={"units": "W/K", "var_type": ModelVarType.PARAMETER},
                                        description="Heat losses to the environment from the hot tank (W/K)",
                                        repr=False)
    UAts_c: list[PositiveFloat] = Field([0.01396848, 0.0001, 0.02286885], title="UAts,c",
                                        json_schema_extra={"units": "W/K", "var_type": ModelVarType.PARAMETER},
                                        description="Heat losses to the environment from the cold tank (W/K)",
                                        repr=False)
    Vts_h: list[PositiveFloat] = Field([5.94771006, 4.87661781, 2.19737023], title="Vts,h",
                                       json_schema_extra={"units": "m3", "var_type": ModelVarType.PARAMETER},
                                       description="Volume of each control volume of the hot tank (m³)", repr=False)
    Vts_c: list[PositiveFloat] = Field([5.33410037, 7.56470594, 0.90547187], title="Vts,c",
                                       json_schema_extra={"units": "m3", "var_type": ModelVarType.PARAMETER},
                                       description="Volume of each control volume of the cold tank (m³)", repr=False)

    ## Solar field
    sf_actuators: list[Actuator] | list[str] = Field(["sf_pump"], title="Solar field actuators", repr=False,
                                                     json_schema_extra={"var_type": ModelVarType.PARAMETER},
                                                     description="Actuators to estimate electricity consumption for the solar field")

    beta_sf: float = Field(4.36396e-02, title="βsf", json_schema_extra={"units": "m", "var_type": ModelVarType.PARAMETER},
                           repr=False, description="Solar field. Gain coefficient", gt=0, le=1)
    H_sf: float = Field(13.676448551722462, title="Hsf", json_schema_extra={"units": "W/m2", "var_type": ModelVarType.PARAMETER},
                        repr=False, description="Solar field. Losses to the environment", ge=0, le=20)
    gamma_sf: float = Field(0.1, title="γsf", json_schema_extra={"units": "-", "var_type": ModelVarType.PARAMETER},
                            repr=False, description="Solar field. Artificial parameters to account for flow variations within the "
                                        "whole solar field", ge=0, le=1)
    filter_sf: float = Field(0.1, title="filter_sf", json_schema_extra={"var_type": ModelVarType.PARAMETER, "units": "-"}, repr=False,
                             description="Solar field. Weighted average filter coefficient to smooth the flow rate", ge=0, le=1)

    nt_sf: int = Field(1, title="nt,sf", repr=False, json_schema_extra={"var_type": ModelVarType.PARAMETER},
                       description="Solar field. Number of tubes in parallel per collector. Defaults to 1.", ge=0)
    np_sf: int = Field(7 * 5, title="np,sf", repr=False, json_schema_extra={"var_type": ModelVarType.PARAMETER},
                       description="Solar field. Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments.",
                       ge=0)
    ns_sf: int = Field(2, title="ns,sf", repr=False, json_schema_extra={"var_type": ModelVarType.PARAMETER},
                       description="Solar field. Number of loops in series", ge=0)
    Lt_sf: float = Field(1.15 * 20, title="Ltsf", repr=False,
                         json_schema_extra={"var_type": ModelVarType.PARAMETER, "units": "m"}, description="Solar field. Collector tube length", gt=0)
    Acs_sf: float = Field(7.85e-5, title="Acs,sf", repr=False, json_schema_extra={"var_type": ModelVarType.PARAMETER, "units": "m2"},
                          description="Solar field. Flat plate collector tube cross-section area", gt=0)
    Kp_sf: float = Field(-0.1, title="Kp,sf", repr=False, description="Solar field. Proportional gain for the local PID controller", le=0,
                         json_schema_extra={"var_type": ModelVarType.PARAMETER}, )
    Ki_sf: float = Field(-0.01, title="Ki,sf", repr=False, description="Solar field. Integral gain for the local PID controller", le=0,
                         json_schema_extra={"var_type": ModelVarType.PARAMETER}, )

    ## Heat exchanger
    UA_hx: float = Field(13536.596, title="UA,hx", json_schema_extra={"units": "W/K", "var_type": ModelVarType.PARAMETER}, repr=False,
                         description="Heat exchanger. Heat transfer coefficient", gt=0)
    H_hx: float = Field(0, title="Hhx", json_schema_extra={"var_type": ModelVarType.PARAMETER, "units": "W/m2"}, repr=False,
                        description="Heat exchanger. Losses to the environment")

    # Variables (states, outputs, decision variables, inputs, etc.)
    ## General
    current_sample: int = Field(0, title="Current sample (s)", json_schema_extra={"units": "sec", "var_type": ModelVarType.TIMESERIES},
                                description="Current model sample (NOTE: not in sync with FSMs sample counters)")
    ## Environment
    wmed_f: float = Field(35, title="wmed,f", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "g/kg"},
                          description="Environment. Seawater / MED feedwater salinity (g/kg)", gt=0)
    Tamb: float = Field(None, title="Tamb", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                        description="Environment. Ambient temperature (ºC)", ge=-15, le=50)
    I: float = Field(None, title="I", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "W/m2"},
                     description="Environment. Solar irradiance (W/m2)", ge=0, le=2000)
    Tmed_c_in: float = Field(None, title="Tmed,c,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                             description="Environment. Seawater temperature (ºC)", ge=10, le=28)

    ## Thermal storage
    mts_src_sp: float = Field(None, title="mts,src*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                              description="Decision variable. Thermal storage recharge flow rate (m³/h)")

    mts_src: float = Field(None, title="mts,src", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                           description="Output. Thermal storage recharge flow rate (m³/h)")
    mts_dis: float = Field(None, title="mts,dis", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                           description="Output. Thermal storage discharge flow rate (m³/h)")
    Tts_h_in: conHotTemperatureType = Field(None, title="Tts,h,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                            description="Output. Thermal storage heat source inlet temperature, to top of hot tank == Thx_s_out (ºC)")
    Tts_c_in: conHotTemperatureType = Field(None, title="Tts,c,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                            description="Output. Thermal storage load discharge inlet temperature, to bottom of cold tank == Tmed_s_out (ºC)")
    Tts_h_out: conHotTemperatureType = Field(None, title="Tts,h,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                             description="Output. Thermal storage heat source outlet temperature, from top of hot tank == Tts_h_t (ºC)")
    Tts_h: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,h",
                                                                                   json_schema_extra={"var_type": None, "units": "C"},
                                                                                   description="Output. Temperature profile in the hot tank (ºC)")
    Tts_c: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,c",
                                                                                   json_schema_extra={"var_type": None, "units": "C"},
                                                                                   description="Output. Temperature profile in the cold tank (ºC)")
    Pts_src: float = Field(None, title="Pth,ts,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                           description="Output. Thermal storage inlet power (kWth)")
    Pts_dis: float = Field(None, title="Pth,ts,dis", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                           description="Output. Thermal storage outlet power (kWth)")
    Jts: float = Field(None, title="Jts,e", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWe"},
                       description="Output. Thermal storage electrical power consumption (kWe)")

    ## Solar field
    Tsf_out_sp: conHotTemperatureType = Field(None, title="Tsf,out*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                              description="Decision variable. Solar field outlet temperature (ºC)")

    Tsf_out: conHotTemperatureType = Field(None, title="Tsf,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                           description="Output. Solar field outlet temperature (ºC)")
    Tsf_in: conHotTemperatureType = Field(None, title="Tsf,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                          description="Output. Solar field inlet temperature (ºC)")
    Tsf_in_ant: np.ndarray[conHotTemperatureType] = Field(..., title="Tsf,in_ant", json_schema_extra={"var_type": None, "units": "C"},
                                                          description="Solar field inlet temperature in the previous Nsf_max steps (ºC)")
    msf_ant: np.ndarray[float] = Field(..., repr=False, exclude=False, json_schema_extra={"var_type": None},
                                       description='Solar field flow rate in the previous Nsf_max steps', )
    Tsf_out_ant: conHotTemperatureType = Field(None, title="Tsf,out,ant", json_schema_extra={"var_type": None, "units": "C"},
                                               description="Output. Solar field prior outlet temperature (ºC)")
    msf: float = Field(None, title="msf", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                       description="Output. Solar field flow rate (m³/h)", alias="qsf")
    SEC_sf: float = Field(None, title="SEC_sf", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWhe/kWth"},
                          description="Output. Solar field conversion efficiency (kWhe/kWth)")
    Jsf: float = Field(None, title="Jsf,e", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kW"},
                       description="Output. Solar field electrical power consumption (kWe)")
    Psf: float = Field(None, title="Pth_sf", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                       description="Output. Solar field thermal power generated (kWth)")

    ## MED
    mmed_s_sp: float = Field(None, title="mmed,s*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                             description="Decision variable. MED hot water flow rate (m³/h)")
    mmed_f_sp: float = Field(None, title="mmed,f*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                             description="Decision variable. MED feedwater flow rate (m³/h)")
    # Here absolute limits are defined, but upper limit depends on Tts_h_t
    Tmed_s_in_sp: float = Field(None, title="Tmed,s,in*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                description="Decision variable. MED hot water inlet temperature (ºC)")
    Tmed_c_out_sp: float = Field(None, title="Tmed,c,out*", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                                 description="Decision variable. MED condenser outlet temperature (ºC)")
    mmed_s: float = Field(None, title="mmed,s", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED hot water flow rate (m³/h)")
    mmed_f: float = Field(None, title="mmed,f", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED feedwater flow rate (m³/h)")
    Tmed_s_in: float = Field(None, title="Tmed,s,in", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                             description="Output. MED hot water inlet temperature (ºC)")
    Tmed_c_out: float = Field(None, title="Tmed,c,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                              description="Output. MED condenser outlet temperature (ºC)", ge=0)
    mmed_c: float = Field(None, title="mmed,c", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED condenser flow rate (m³/h)")
    Tmed_s_out: float = Field(None, title="Tmed,s,out", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "C"},
                              description="Output. MED heat source outlet temperature (ºC)")
    mmed_d: float = Field(None, title="mmed,d", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED distillate flow rate (m³/h)")
    mmed_b: float = Field(None, title="mmed,b", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                          description="Output. MED brine flow rate (m³/h)")
    Jmed: float = Field(None, title="Jmed", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWe"},
                        description="Output. MED electrical power consumption (kW)")
    Pmed: float = Field(None, title="Pmed", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                        description="Output. MED thermal power consumption ~= Pth_ts_out (kW)")
    STEC_med: float = Field(None, title="STEC_med", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWhe/m3"},
                            description="Output. MED specific thermal energy consumption (kWhe/m³)")
    SEEC_med: float = Field(None, title="SEEC_med", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWhth/m3"},
                            description="Output. MED specific electrical energy consumption (kWhth/m³)")

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
    mhx_p: float = Field(None, title="mhx,p", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                         description="Output. Heat exchanger primary circuit (hot side) flow rate == msf (m³/h)")
    mhx_s: float = Field(None, title="mhx,s", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                         description="Output. Heat exchanger secondary circuit (cold side) flow rate == mts_src (m³/h)")
    Phx_p: float = Field(None, title="Pth,hx,p", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                         description="Output. Heat exchanger primary circuit (hot side) power == Pth_sf (kWth)")
    Phx_s: float = Field(None, title="Pth,hx,s", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWth"},
                         description="Output. Heat exchanger secondary circuit (cold side) power == Pth_ts_in (kWth)")
    epsilon_hx: float = Field(None, title="εhx", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                              description="Output. Heat exchanger effectiveness (-)")

    ## Three-way valve
    # Same case as with heat exchanger
    R3wv: float = Field(None, title="R3wv", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                        description="Output. Three-way valve mix ratio (-)")
    m3wv_src: float = Field(None, title="m3wv,src", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
                            description="Output. Three-way valve source flow rate == mts,dis (m³/h)")
    m3wv_dis: float = Field(None, title="m3wv,dis", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "m3/h"},
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
    # penalty: float = Field(0, title="penalty", json_schema_extra={"var_type": ModelVarType.timeseries, "units": "u.m."}, ge=0,
    #                        description="Penalty for undesired states or conditions", repr=False)

    Jtotal: float = Field(None, title="Jtotal", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "kWe"},
                          description="Total electrical power consumption (kWe)")


    ## Finite State Machine (FSM) states
    med_vacuum_state: MedVacuumState = Field(MedVacuumState.OFF, title="MEDvacuum,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                             description="Input. MED vacuum system state")
    med_state: MedState = Field(MedState.OFF, title="MED,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                description="Input/Output. MED state. It can be used to define the MED initial state, after it's always an output")
    sf_state: SolarFieldState = Field(SolarFieldState.IDLE, title="SF,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                      description="Input/Output. Solar field state. It can be used to define the Solar Field initial state, after it's always an output")
    ts_state: ThermalStorageState = Field(ThermalStorageState.IDLE, title="TS,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                          description="Input/Output. Thermal storage state. It can be used to define the Thermal Storage initial state, after it's always an output")
    sf_ts_state: SF_TS_State = Field(SF_TS_State.IDLE, title="SF_TS,state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                     description="Output. Solar field with thermal storage state")
    current_state: SolarMED_State = Field(None, title="state", json_schema_extra={"var_type": ModelVarType.TIMESERIES, "units": "-"},
                                          description="Output. Current state of the SolarMED system")

    # Private attributes
    _MED_model: Any = PrivateAttr(None) #= Field(None, repr=False, description="MATLAB MED model instance")
    _pid_sf: PID | None = PrivateAttr(None)#Field(None, repr=False, description="PID controller for the solar field")
    # _export_fields_df: list[str] = PrivateAttr(None) # Fields to export into a dataframe
    # _export_fields_config: list[str] = PrivateAttr(None) # Fields to export into model parameters dict
    _med_fsm: MedFSM = PrivateAttr(None) # Finite State Machine object for the MED system. Should not be accessed/manipulated directly
    _sf_ts_fsm: SolarFieldWithThermalStorage_FSM = PrivateAttr(None) # Finite State Machine object for the Solar Field with Thermal Storage system. Should not be accessed/manipulated directly
    # _created_at: datetime = PrivateAttr(default_factory=datetime.datetime.now) # Should not be accessed/manipulated directly


    model_config = ConfigDict(
        validate_assignment=True,  # So that fields are validated, not only when created, but every time they are set
        arbitrary_types_allowed=True
        # numpy.ndarray[typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=110)])]]
    )

    @computed_field(title="Export fields df", description="Fields to export into a dataframe", json_schema_extra={"var_type": ModelVarType.PARAMETER})
    @property
    def export_fields_df(self) -> list[str]:
        # Fields to export into a dataframe
        # Choose variable to export based on explicit property, need to be single variables
        fields = []
        for field in self.__fields__.keys():
            if 'var_type' in self.__fields__[field].json_schema_extra:
                if self.__fields__[field].json_schema_extra.get('var_type', None) == ModelVarType.TIMESERIES:
                    field_value = getattr(self, field)
                    if check_value_single(field_value, field):
                        fields.append(field)
            else:
                logger.warning(f'Field {field} has no `json_schema_extra.var_type` set, it should be if it needs to be included in the exports')

        return fields

    @computed_field(title="Export fields model config", description="Fields to export into model parameters dict", json_schema_extra={"var_type": ModelVarType.PARAMETER})
    @property
    def export_fields_config(self) -> list[str]:
        # Fields to export into model parameters dict
        # return [field for field in self.__fields__.keys() if
        #         self.__fields__[field].json_schema_extra.get('var_type', None) == ModelVarType.PARAMETER]

        fields = []
        for field in self.__fields__.keys():
            if 'var_type' in self.__fields__[field].json_schema_extra:
                if self.__fields__[field].json_schema_extra.get('var_type', None) == ModelVarType.PARAMETER:
                        fields.append(field)
            else:
                logger.warning(
                    f'Field {field} has no `json_schema_extra.var_type` set, it should be if it needs to be included in the exports')

        return fields

    # @computed_field
    # def consumption_fits(self) -> dict:
    #     # Load electrical consumption fit curves
    #     try:
    #         with open(self.curve_fits_path, 'r') as file:
    #             return json.load(file)
    #
    #     except FileNotFoundError:
    #         raise ValidationError(f'Curve fits file not found in {self.curve_fits_path}')

    @field_validator("med_actuators", "sf_actuators", "ts_actuators", mode='before')
    @classmethod
    def generate_actuators(cls, actuator_ids: SupportedActuators | list[SupportedActuators]) -> list[Actuator]:
        if isinstance(actuator_ids, str):
            actuator_ids = [actuator_ids]

        return [Actuator(id=actuator_id) for actuator_id in actuator_ids]

    # @field_validator("Tts_h", "Tts_c")
    # @classmethod
    # def validate_Tts(cls, Tts: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType], info: ValidationInfo) -> np.ndarray[conHotTemperatureType]:
    #     # Just to make sure it's a numpy array
    #     return np.array(Tts, dtype=float)

    @field_validator("mmed_s_sp", "mmed_f_sp", "mts_src_sp", "msf", "mmed_c")
    @classmethod
    def validate_flow(cls, flow: float, info: ValidationInfo) -> PositiveFloat:
        lims_field = "lims_" + info.field_name.removesuffix("_sp")

        return within_range_or_zero_or_max(flow, range=info.data[lims_field], var_name=info.field_name)

    @field_validator("Tmed_s_in_sp")
    @classmethod
    def validate_Tmed_s_in(cls, Tmed_s_in: float, info: ValidationInfo) -> float:
        # Lower limit set by pre-defined operational limit, if lower -> 0
        # Upper bound, take the lower between the hot tank top temperature and the pre-defined operational limit
        return within_range_or_zero_or_max(
            Tmed_s_in, range=(info.data["lims_Tmed_s_in"][0],
                              np.min([info.data["lims_Tmed_s_in"][1], info.data["Tts_h"][0]]),),
            var_name=info.field_name
        )

    @field_validator("Tmed_c_out_sp")
    @classmethod
    def validate_Tmed_c_out(cls, Tmed_c_out: float, info: ValidationInfo) -> float:
        # The upper limit is not really needed, its value is an output restrained already by mmed_c upper lower bound
        return within_range_or_min_or_max(Tmed_c_out, range=(info.data["Tmed_c_in"],
                                                             info.data["lims_T_hot"][1]),
                                          var_name=info.field_name)

    @field_validator("Tsf_out_sp")
    @classmethod
    def validate_Tsf_out(cls, Tsf_out: float, info: ValidationInfo) -> float:
        # Lower limit set by greater value between pre-defined operational limit, and inlet temperature,
        # if input value is lower -> 0 (deactivated)
        lower_limit = np.max([info.data["lims_Tsf_out"][0], info.data["Tsf_in"]] if info.data["Tsf_in"] is not None else info.data["lims_Tsf_out"][0])
        # Upper limit set by pre-defined operational limit
        return within_range_or_zero_or_max(Tsf_out, range=(lower_limit, info.data["lims_Tsf_out"][1]),
                                           var_name=info.field_name)


    def model_post_init(self, ctx):

        if self.sample_time > 1200 and self.use_models:
            logger.warning("Sample time is too high for the time-dependent models (i.e. solar field). "
                                 "This is likely to cause unfeasibilities in the model evaluation")

        if not self.use_models and not self.use_finite_state_machine:
            raise ValueError("At least one of `use_models` or `use_finite_state_machine` must be set to True")

        if self.use_models:

            # Initialize the MATLAB engine
            self.init_matlab_engine()

        # Make sure thermal storage state is a numpy array
        self.Tts_h = np.array(self.Tts_h, dtype=float)
        self.Tts_c = np.array(self.Tts_c, dtype=float)

        # Make sure actuators are instances of the Actuator class
        for idx, actuator in enumerate(self.med_actuators):
            self.med_actuators[idx] = actuator if isinstance(actuator, Actuator) else Actuator(id=actuator)

        for idx, actuator in enumerate(self.sf_actuators):
            self.sf_actuators[idx] = actuator if isinstance(actuator, Actuator) else Actuator(id=actuator)

        for idx, actuator in enumerate(self.ts_actuators):
            self.ts_actuators[idx] = actuator if isinstance(actuator, Actuator) else Actuator(id=actuator)

        # Initialize outlet temperature to the provided inlet initial temperature
        self.Tsf_in = self.Tsf_in_ant[-1]
        self.Tsf_out_ant = self.Tsf_in

        # Make a list of field names that are of type numeric (int, float, etc)
        # self.export_fields = [field for field in self.__fields__.keys() if isinstance(getattr(self, field), (int, float))]

        if self.use_finite_state_machine:

            initial_sf_ts = SF_TS_State(str(self.sf_state.value) + str(self.ts_state.value))
            self.current_state = SolarMED_State(initial_sf_ts.value + str(self.med_state.value))

            self._sf_ts_fsm: SolarFieldWithThermalStorage_FSM = SolarFieldWithThermalStorage_FSM(
                name='SF-TS', initial_state=initial_sf_ts, sample_time=self.sample_time)
            self._med_fsm: MedFSM = MedFSM(
                name='MED', initial_state=self.med_state,
                sample_time=self.sample_time, vacuum_duration_time=self.vacuum_duration_time,
                brine_emptying_time=self.brine_emptying_time, startup_duration_time=self.startup_duration_time
            )


        logger.info(f'''
        SolarMED model initialized with: 
            - Evaluating models: {self.use_models}
            - Evaluating finite state machines: {self.use_finite_state_machine}
            - Resolution mode: {self.resolution_mode}
            - Sample time: {self.sample_time} s
            - MED actuators: {[actuator.id for actuator in self.med_actuators]}
            - Solar field actuators: {[actuator.id for actuator in self.sf_actuators]}
            - Thermal storage actuators: {[actuator.id for actuator in self.ts_actuators]}
        ''')

    # def set_operating_state(self) -> None:
    #
    #     if self.med_active and self.sf_active and self.ts_active:
    #         self.operating_state = SolarMED_states.SOLAR_FIELD_HEATING_THERMAL_STORAGE_MED
    #     elif self.med_active and self.sf_active:
    #         self.operating_state = SolarMED_states.SOLAR_FIELD_WARMUP_MED
    #     elif self.med_active:
    #         self.operating_state = SolarMED_states.THERMAL_STORAGE_DISCHARGE_MED
    #     elif self.sf_active:
    #         self.operating_state = SolarMED_states.SOLAR_FIELD_WARMUP
    #     elif self.sf_active and self.ts_active:
    #         self.operating_state = SolarMED_states.SOLAR_FIELD_HEATING_THERMAL_STORAGE
    #     elif self.ts_active:
    #         self.operating_state = SolarMED_states.THERMAL_STORAGE_RECIRCULATING
    #     else:
    #         self.operating_state = SolarMED_states.IDLE

        # logger.debug(f"Operating state: {self.operating_state.name}")

    def step(
            self,
            mts_src: float,  # Thermal storage decision variables
            Tsf_out: float,  # Solar field decision variables
            mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float,  # MED decision variables
            Tmed_c_in: float, Tamb: float, I: float, wmed_f: float = None,  # Environment variables
            msf: float = None, # Optional, to provide the solar field flow rate when starting up (Tsf_out takes priority)
            med_vacuum_state: int | MedVacuumState = None,  # Optional, to provide the MED vacuum state (OFF, LOW, HIGH)
    ) -> None:

        """
        Update model outputs given current environment variables and decision variables

            Inputs:
                - Decision variables
                    MED
                    ---------------------------------------------------
                    + mmed_s (m³/h): Heat source flow rate
                    + mmed_f (m³/h): Feed water flow rate
                    + Tmed_s_in (ºC): Heat source inlet temperature
                    + Tmed_c_out (ºC): Cooling water outlet temperature

                    THERMAL STORAGE
                    ---------------------------------------------------
                    + mts_src (m³/h): Thermal storage heat source flow rate

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
        self.mmed_s_sp = mmed_s #if mmed_s is not None else self.mmed_s # Use the previous value
        self.mmed_f_sp = mmed_f
        self.Tmed_s_in_sp = Tmed_s_in
        self.Tmed_c_out_sp = Tmed_c_out
        self.med_vacuum_state = med_vacuum_state

        # Thermal storage
        self.mts_src_sp = mts_src
        self.mts_src = self.mts_src_sp  # To make sure we use the validated value

        # Solar field
        self.Tsf_out_sp = Tsf_out # To make sure we use the validated value
        if self.Tsf_out_sp == 0:
            if msf is not None:
                self.msf = msf
                logger.debug(f"Solar field warm up mode, using provided qsf: {msf:.1f} (m³/h) instead of Tsf,out,sp")
            else:
                self.msf = 0

        # Initialize variables
        # self.penalty = 0
        self.med_active = False
        self.sf_active = False
        self.ts_active = False

        # Set operating mode
        # Check MED state
        if self.mmed_s_sp > 0 and self.mmed_f_sp > 0 and self.Tmed_s_in_sp > 0 and self.Tmed_c_out_sp > 0:
            self.med_active = True

        # Check solar field state
        if self.Tsf_out_sp > 0 or self.msf > 0:
            self.sf_active = True
            # Initialize solar field local controller
        if self.sf_active and self.use_models:
            self._pid_sf = PID(Kp=self.Kp_sf, Ki=self.Ki_sf, sample_time=self.sample_time, output_limits=(self.lims_msf[0], self.lims_msf[-1]), setpoint=self.Tsf_out_sp)

        # Check heat exchanger state / thermal storage state
        if self.mts_src_sp > 0:
            self.ts_active = True

        # # Update operating mode
        # self.set_operating_state()

        # After the validation, variables are either zero or within the limits (>0),
        # based on this, the step method in the individual state machines are called

        if self.use_finite_state_machine:
            # Save FSMs before updating them
            sf_ts_fsm0 = copy.deepcopy(self._sf_ts_fsm)
            med_fsm0 = copy.deepcopy(self._med_fsm)

            self._sf_ts_fsm.step(Tsf_out=self.Tsf_out_sp, qts_src=self.mts_src_sp, qsf=self.msf)
            self._med_fsm.step(mmed_s=self.mmed_s_sp, mmed_f=self.mmed_f_sp, Tmed_s_in=self.Tmed_s_in_sp,
                               Tmed_c_out=self.Tmed_c_out_sp, med_vacuum_state=self.med_vacuum_state)

            self.update_current_state()
            logger.debug(f"SolarMED state after inputs validation: {self.current_state}")

            # If the finite state machines are used, they need to set the values of: sf_active, ts_active and med_active
            # before evaluation the step
            if self.med_state != MedState.ACTIVE:
                self.med_active = False

            if self.sf_state != SolarFieldState.ACTIVE:
                self.sf_active = False

            if self.ts_state != ThermalStorageState.ACTIVE:
                self.ts_active = False

        # Solve model for current step
        self.solve_step()

        # Re-evaluate FSM once the models have been solved
        # Do we really want to be doing this?
        # Sporadic invalid inputs might change the operating mode which takes a long time to recover
        # if self.use_finite_state_machine and self.use_models:
        #     self._sf_ts_fsm = sf_ts_fsm0
        #     self._med_fsm = med_fsm0
        #
        #     self._sf_ts_fsm.step(Tsf_out=self.Tsf_out, qts_src=self.mts_src)
        #     self._med_fsm.step(mmed_s=self.mmed_s, mmed_f=self.mmed_f, Tmed_s_in=self.Tmed_s_in,
        #                        Tmed_c_out=self.Tmed_c_out, med_vacuum_state=self.med_vacuum_state)
        #
        #     self.update_current_state()
        #     logger.debug(f"SolarMED state after step evaluation: {self.current_state}")

    def init_matlab_engine(self):
        """
        Manually initialize the MATLAB MED model, in case it was terminated.
        """
        # Conditionally import the module
        if self._MED_model is None:
            import MED_model
            import matlab

        self._MED_model = MED_model.initialize()
        self._matlab = matlab
        logger.info('MATLAB engine initialized')

    def solve_step(self):
        # Every individual model `solve` method considers either the active state of the component
        # or the use_models logical variable to return its outputs. If any is false they are not evaluated
        # but return their default values
        # Maybe it would be preferable to make it more explicit?

        # 1st. MED

        (self.mmed_s, self.mmed_f, self.Tmed_s_in, self.Tmed_c_out, self.mmed_d, self.mmed_c, self.mmed_b,
         self.Tmed_s_out, self.Jmed, self.Pmed, self.SEEC_med, self.STEC_med) = \
            self.solve_MED(self.mmed_s_sp, self.mmed_f_sp, self.Tmed_s_in_sp, self.Tmed_c_out_sp, self.Tmed_c_in)


        # 2nd. Three-way valve
        self.m3wv_src, self.R3wv = three_way_valve_model(
            Mdis=self.mmed_s, Tsrc=self.Tts_h[0], Tdis_in=self.Tmed_s_in, Tdis_out=self.Tmed_s_out
        )

        self.mts_dis = self.m3wv_src
        self.m3wv_dis = self.mmed_s

        # 3rd. Solve solar field, heat exchanger and thermal storage

        # If both the solar field is active and the thermal storage is being recharged
        # Then the system is coupled, solve coupled subproblem
        if self.ts_active and self.sf_active and self.use_models:
            self.msf, self.Tsf_out, self.Tsf_in, self.Tts_h, self.Tts_c, self.Tts_h_in = self.solve_coupled_subproblem()

        # Otherwise, solar field and thermal storage are decoupled, or their models are not being used and defaults are returned
        # Solve each system independently
        else:
            # Solve thermal storage
            self.Tts_h_in = self.Tts_c[-1] # Hot tank inlet temperature is the bottom temperature of the cold tank
            self.Tts_h, self.Tts_c = self.solve_thermal_storage(Tts_h_in=self.Tts_h_in)

            # Solve solar field, calculate msf and validate it's within limits, then recalculate Tsf
            # if self.sf_active:
            if self.Tsf_out_sp > 0:
                if self.use_models:
                    # self.msf = self.solve_solar_field_inverse(Tsf_out=self.Tsf_out_sp)
                    self.msf = self._pid_sf(self.Tsf_out_ant, dt=self.sample_time)
                else:
                    self.msf = 0

            if not self.sf_active: # Ignore the provided flow rate if the solar field is not active
                self.msf = 0

            # Use either the provided flow rate or the calculated one from Tsf,out,sp or bypass the model
            self.Tsf_out = self.solve_solar_field(Tsf_in=self.Tsf_in, msf=self.msf)

            # Since they are decoupled, necessarily the outlet of the solar field becomes its inlet
            self.Tsf_in = self.Tsf_out + 3  # Tener en cuenta de que esto no tiene sentido físico, pero se observa en los datos experimentales

        # Update solar field prior values
        self.Tsf_out_ant = self.Tsf_out

        self.msf_ant = np.roll(self.msf_ant, -1)  # Shift all elements to the left
        self.msf_ant[-1] = self.msf  # Add the new value at the end

        self.Tsf_in_ant = np.roll(self.Tsf_in_ant, -1)  # Shift all elements to the left
        self.Tsf_in_ant[-1] = self.Tsf_in  # Add the new value at the end

        # Calculate additional outputs
        # TODO: Clean this part by moving it to a separate method / module
        # Pth_sf, Jsf_e, SEC_sf
        if self.sf_active and self.use_models:
            w_props_sf = w_props(P=0.16, T=(self.Tsf_in + self.Tsf_out) / 2 + 273.15)
            self.Psf = self.msf * w_props_sf.rho * (self.Tsf_out - self.Tsf_in) * w_props_sf.cp / 3600  # kWth
            # self.Jsf_e = self.electrical_consumption(self.msf, self.consumption_fits[""]) # kWhe # TODO: Add the right fit
            self.Jsf = self.sf_actuators[0].calculate_power_consumption(self.msf)
            self.SEC_sf = self.Jsf / self.Psf if self.Psf > 0 else np.nan  # kWhe/kWth
        else:
            self.Psf = 0
            self.Jsf = 0
            self.SEC_sf = np.nan

        if self.ts_active and self.use_models:
            # Pth_ts_out, Pth_ts_in, Jts_e
            w_props_ts_in = w_props(P=0.16, T=(self.Tts_h_in + self.Tts_c[-1]) / 2 + 273.15)
            self.Pts_src = self.mts_src * w_props_ts_in.rho * (
                        self.Tts_h_in - self.Tts_c[-1]) * w_props_ts_in.cp / 3600  # kWth
            self.Jts = self.ts_actuators[0].calculate_power_consumption(self.mts_src)
        else:
            self.Pts_src = 0
            self.Jts = 0

        if self.med_active and self.use_models:
            w_props_ts_out = w_props(P=0.16, T=(self.Tmed_s_out + self.Tts_h[1]) / 2 + 273.15)
            self.Pts_dis = self.mts_dis * w_props_ts_out.rho * (
                    self.Tts_h[1] - self.Tmed_s_out) * w_props_ts_out.cp / 3600  # kWth
            # self.Jts_e = self.electrical_consumption(self.mts_src, self.consumption_fits[""]) # kWhe # TODO: Add the right fit

            # TODO: If there is an alternative thermal storage configuration, the index needs to be the one where the extraction is done
            self.Tts_h_out = self.Tts_h[0]
            self.Tts_c_in = self.Tmed_s_out
        else:
            self.Pts_dis = 0
            self.Tts_h_out = 0
            self.Tts_c_in = 0


        if self.use_models:
            # Copied variables for the heat exchanger
            self.Thx_p_in = self.Tsf_out
            self.Thx_p_out = self.Tsf_in
            self.Thx_s_in = self.Tts_c[-1]
            self.Thx_s_out = self.Tts_h_in
            self.mhx_p = self.msf
            self.mhx_s = self.mts_src
            self.Phx_p = self.Psf
            self.Phx_s = self.Pts_src

            self.epsilon_hx = calculate_heat_transfer_effectiveness(
                Tp_in=self.Thx_p_in,
                Tp_out=self.Thx_p_out,
                Ts_in=self.Thx_s_in,
                Ts_out=self.Thx_s_out,
                qp=self.mhx_p,
                qs=self.mhx_s
            )

            # Total variables
            self.Jtotal = self.Jmed + self.Jts + self.Jsf

        # self.epsilon_hx = self.Pth_hx_s / self.Pth_hx_p if self.Pth_hx_p > 0 else np.nan

    def get_state(self, mode: Literal["default", "human_readable"] = 'default') -> SolarMED_State:
        # state_code = self.generate_state_code(self._sf_ts_fsm.state, self._med_fsm.state)

        state_code = str(self.sf_state.value) + str(self.ts_state.value) + str(self.med_state.value)

        if mode == 'human_readable':
            state_str = SolarMED_State(state_code).name
            # Replace _ by space and make everything minusculas
            state_str =  state_str.replace('_', ' ').lower()
            # Replace ts to TS, sf to SF and med to MED
            state_str = state_str.replace('ts', 'TS').replace('sf', 'SF').replace('med', 'MED')

            return state_str

        else:
            return SolarMED_State(state_code)

    def update_internal_states(self) -> None:
        self.med_state = self._med_fsm.get_state()
        self.sf_ts_state: SF_TS_State = self._sf_ts_fsm.get_state()
        self.sf_state = SolarFieldState(int(self.sf_ts_state.value[0]))
        self.ts_state = ThermalStorageState(int(self.sf_ts_state.value[1]))

    def update_current_state(self) -> None:
        self.update_internal_states()
        self.current_state = self.get_state()


    def solve_MED(self, mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float, Tmed_c_in: float):

        def auxiliary_consumption():

            # TODO: Replace these values by value from the class
            if self.med_vacuum_state == MedVacuumState.HIGH:
                return 5  # kW, high vacuum consumption

            if self.med_vacuum_state == MedVacuumState.LOW:
                return 1  # kW, reduced vacuum consumption


        def default_values():
            # Process outputs
            mmed_d = 0
            mmed_c = 0
            mmed_b = 0
            Tmed_s_out = 0

            # Consumptions / metrics
            Jmed = 0
            Pmed = 0
            SEEC_med = np.nan
            STEC_med = np.nan

            # Overiden decision variables
            mmed_s = 0
            mmed_f = 0
            Tmed_s_in = 0
            Tmed_c_out = Tmed_c_in # Or 0?

            if self.use_finite_state_machine:
                Jmed += auxiliary_consumption()

                if self.med_state == MedState.STARTING_UP:
                    mmed_s = 10 * 3.6
                    mmed_f = 5 * 3.6
                    mmed_b = mmed_f
                elif self.med_state == MedState.SHUTTING_DOWN:
                    mmed_b = 5

                # Additional consumptions when STARTING_UP for example
                Jmed += np.sum(
                    [actuator.calculate_power_consumption(flow)
                     for actuator, flow in zip(self.med_actuators, [mmed_b, mmed_f, mmed_d, mmed_c, mmed_s])
                     ]
                )

            return mmed_s, mmed_f, Tmed_s_in, Tmed_c_out, mmed_d, mmed_c, mmed_b, Tmed_s_out, Jmed, Pmed, SEEC_med, STEC_med


        Tmed_c_out0 = Tmed_c_out
        med_model_solved = False

        if self.med_active == False or not self.use_models:
            return default_values()


        MsIn = self._matlab.double([mmed_s / 3.6], size=(1, 1))  # m³/h -> L/s
        TsinIn = self._matlab.double([Tmed_s_in], size=(1, 1))
        MfIn = self._matlab.double([mmed_f], size=(1, 1))
        TcwinIn = self._matlab.double([Tmed_c_in], size=(1, 1))
        op_timeIn = self._matlab.double([0], size=(1, 1))
        # wf=wmed_f # El modelo sólo es válido para una salinidad así que ni siquiera
        # se considera como parámetro de entrada

        while not med_model_solved and (Tmed_c_in < Tmed_c_out < self.lims_T_hot[1]):

            TcwoutIn = self._matlab.double([Tmed_c_out], size=(1, 1))
            # try:
            mmed_d, Tmed_s_out, mmed_c, _, _ = self._MED_model.MED_model(
                MsIn,  # L/s
                TsinIn,  # ºC
                MfIn,  # m³/h
                TcwoutIn,  # ºC
                TcwinIn,  # ºC
                op_timeIn,  # hours
                nargout=5
            )

            if mmed_c > self.lims_mmed_c[1]:
                Tmed_c_out += 1
            elif mmed_c < self.lims_mmed_c[0]:
                Tmed_c_out -= 1
            else:
                med_model_solved = True

        if not med_model_solved:
            self.med_active = False
            logger.warning(
                f'MED is not active due to unfeasible operation in the condenser, setting all MED outputs to 0')

            return default_values()

        # Else
        if abs(Tmed_c_out0 - Tmed_c_out) > 0.1:
            logger.debug(
                f"MED condenser flow was out of range, changed outlet temperature from {Tmed_c_out0:.2f} to {Tmed_c_out:.2f}"
            )

        ## Brine flow rate
        mmed_b = mmed_f - mmed_d  # m³/h

        ## MED electrical consumption
        Jmed = 0
        Jmed += np.sum(
            [actuator.calculate_power_consumption(flow)
             for actuator, flow in zip(self.med_actuators, [mmed_b, mmed_f, mmed_d, mmed_c, mmed_s])
             ]
        )
        Jmed += auxiliary_consumption()

        SEEC_med = Jmed / mmed_d  # kWhe/m³

        ## MED STEC
        w_props_s = w_props(P=0.1, T=(Tmed_s_in + Tmed_s_out) / 2 + 273.15)
        cp_s = w_props_s.cp  # kJ/kg·K
        rho_s = w_props_s.rho  # kg/m³
        # rho_d = w_props(P=0.1, T=Tmed_c_out+273.15) # kg/m³
        mmed_s_kgs = mmed_s * rho_s / 3600  # kg/s

        Pmed = mmed_s_kgs * (Tmed_s_in - Tmed_s_out) * cp_s  # kWth
        STEC_med = Pmed / mmed_d  # kWhth/m³

        if not med_model_solved:
            self.med_active = False
            logger.warning(f'MED is not active due to unfeasible operation in the condenser, setting all MED outputs to 0')

        return mmed_s, mmed_f, Tmed_s_in, Tmed_c_out, mmed_d, mmed_c, mmed_b, Tmed_s_out, Jmed, Pmed, SEEC_med, STEC_med

    def solve_thermal_storage(self, Tts_h_in: float, calculate_consumption: bool = False) \
            -> (tuple[np.ndarray[conHotTemperatureType], np.ndarray[conHotTemperatureType], float] |
                tuple[np.ndarray[conHotTemperatureType], np.ndarray[conHotTemperatureType]]):

        if self.use_models:
            Tts_h, Tts_c = thermal_storage_two_tanks_model(
                Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c,  # [ºC], [ºC]
                Tt_in=Tts_h_in,  # ºC
                Tb_in=self.Tmed_s_out,  # ºC
                Tamb=self.Tamb,  # ºC

                qsrc=self.mts_src,  # m³/h
                qdis=self.mts_dis,  # m³/h

                UA_h=self.UAts_h,  # W/K
                UA_c=self.UAts_c,  # W/K
                Vi_h=self.Vts_h,  # m³
                Vi_c=self.Vts_c,  # m³
                ts=self.sample_time, Tmin=self.lims_Tmed_s_in[0]  # seg, ºC
            )

            # Electrical consumption
            Jts = 0
            Jts += np.sum(
                [actuator.calculate_power_consumption(flow)
                 for actuator, flow in zip(self.ts_actuators, [self.mts_src])
                 ]
            )

            if calculate_consumption:
                return Tts_h, Tts_c, Jts

        else:
            Tts_h = self.Tts_h
            Tts_c = self.Tts_c

        return Tts_h, Tts_c

    def solve_heat_exchanger(self, Tsf_out, Tts_c_b, msf, mts_src, return_epsilon=False) -> tuple[float, float, float] | tuple[float, float]:

        Thx_p_out, Thx_s_out, epsilon = heat_exchanger_model(
            Tp_in=Tsf_out,  # Solar field outlet temperature (decision variable, ºC)
            Ts_in=Tts_c_b,  # Cold tank bottom temperature (ºC)

            qp=msf,  # Solar field flow rate (m³/h)
            qs=mts_src,  # Thermal storage charge flow rate (decision variable, m³/h)

            Tamb=self.Tamb,  # Ambient temperature (ºC)

            UA=self.UA_hx,  # Heat transfer coefficient of the heat exchanger (W/K)
            H=self.H_hx,  # Losses to the environment

            return_epsilon=True
        )

        if return_epsilon:
            return Thx_p_out, Thx_s_out, epsilon
        else:
            return Thx_p_out, Thx_s_out

    def solve_solar_field(self, Tsf_in: float, msf: float, calculate_consumption: bool = False) -> float | tuple[float, float]:

        """
        Make sure to set `Tsf_out_ant` to the prior `Tsf_out` value before calling this method
        """

        if self.use_models:
            msf = np.append(self.msf_ant, msf)
            Tsf_in = np.append(self.Tsf_in_ant, Tsf_in)

            Tsf_out = solar_field_model(
                Tin=Tsf_in, # From current value, up to array start
                q=msf, # From current value, up to array start
                I=self.I, Tamb=self.Tamb, Tout_ant=self.Tsf_out_ant,

                # Model tuned parameters
                H=self.H_sf, beta=self.beta_sf, gamma=self.gamma_sf,
                # Model fixed parameters
                Acs=self.Acs_sf, nt=self.nt_sf, npar=self.np_sf, ns=self.ns_sf, Lt=self.Lt_sf,
                sample_time=self.sample_time, consider_transport_delay=True,
            )

            # Electrical consumption
            Jsf = 0
            Jsf += np.sum(
                [actuator.calculate_power_consumption(flow)
                 for actuator, flow in zip(self.ts_actuators, [self.mts_src])
                 ]
            )
            if calculate_consumption:
                return Tsf_out, Jsf
        else:
            Tsf_out = self.Tsf_out_sp # Just bypass the model

        return Tsf_out

    def energy_generation_and_storage_subproblem(self, inputs):
        # TODO: Allow to provide water properties as inputs so they are calculated only once

        if len(inputs) == 2:
            Tts_c_b = inputs[0]
            msf = inputs[1]
        else:
            # Bottom tank temperature is not considered to change
            Tts_c_b = self.Tts_c[-1]
            msf = inputs[0]

        # Heat exchanger of solar field - thermal storage
        Tsf_in, Tts_t_in = heat_exchanger_model(
            Tp_in=self.Tsf_out_sp,  # Solar field outlet temperature (decision variable, ºC)
            Ts_in=Tts_c_b,  # Cold tank bottom temperature (ºC)
            qp=msf,  # Solar field flow rate (m³/h)
            qs=self.mts_src_sp,  # Thermal storage charge flow rate (decision variable, m³/h)
            Tamb=self.Tamb,
            UA=self.UA_hx,
            H=self.H_hx,
            return_epsilon=False
        )

        # Solar field
        Tsf_in = np.append(self.Tsf_in_ant, Tsf_in)

        msf = solar_field_inverse_model(
            Tin=Tsf_in,
            q_ant=self.msf_ant,
            I=self.I, Tamb=self.Tamb, Tout_ant=self.Tsf_out_ant, Tout=self.Tsf_out_sp,

            # Model tuned parameters
            H=self.H_sf, beta=self.beta_sf, gamma=self.gamma_sf,
            # Model fixed parameters
            Acs=self.Acs_sf, nt=self.nt_sf, npar=self.np_sf, ns=self.ns_sf, Lt=self.Lt_sf,
            sample_time=self.sample_time, consider_transport_delay=True,
            filter_signal=True, f=self.filter_sf
        )

        # Si ahora msf realmente no depende de Tsf_in, dejamos de tener un problema acoplado no?
        # pid = copy.deepcopy(self.pid_sf)
        # msf = pid(self.Tsf_out_ant, dt=self.sample_time)

        # Thermal storage
        _, Tts_c = thermal_storage_two_tanks_model(
            Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c,  # [ºC], [ºC]
            Tt_in=Tts_t_in,  # ºC
            Tb_in=self.Tmed_s_out,  # ºC
            Tamb=self.Tamb,  # ºC
            qsrc=self.mts_src_sp,  # m³/h
            qdis=self.mts_dis,  # m³/h
            UA_h=self.UAts_h,  # W/K
            UA_c=self.UAts_c,  # W/K
            Vi_h=self.Vts_h,  # m³
            Vi_c=self.Vts_c,  # m³
            ts=self.sample_time, Tmin=self.lims_Tmed_s_in[0]  # seg, ºC
        )

        if len(inputs) == 2:
            return [abs(Tts_c[-1] - inputs[0]), abs(msf - inputs[1])]
        else:
            return [abs(msf - inputs[0])]

    def solve_coupled_subproblem(self, ) -> tuple[float, float, float, np.ndarray, np.ndarray, float]:
        """
        Solve the coupled subproblem of the solar MED system

        The given Tsf_out_sp is associated with a solar field flow, that depends on the solar field inlet
        temperature. This is determined by the heat exchanger. However, the heat exchanger on itself depends on the
        flow of the solar field and the thermal storage outlet temperature (does not change much).

        1. Find the flow of the solar field (msf) and the outlet temperature of the cold tank (Tts_c_b),
        that minimize the difference between the current Tts_c_b (it is assumed it does not change much between samples)
        and solar field flow for the given Tsf_out_sp.

        2. With the obtained msf and Tts_c_b, recalculate Tsf_out and Tts_c_b, and iterate until convergence
        """

        if self.resolution_mode == 'precise':

            initial_guess = [self.Tts_c[-1], self.msf if self.msf is not None else self.lims_msf[0]]
            bounds = ((self.lims_T_hot[0], self.lims_msf[0]), (self.lims_T_hot[1], self.lims_msf[1]))

            outputs = least_squares(self.energy_generation_and_storage_subproblem, initial_guess, bounds=bounds)
            Tts_c_b = outputs.x[0]
            msf = outputs.x[1]

            cnt_max = 10

        elif self.resolution_mode == 'simple':
            # """
            # In the simplified version, we assume Tts_c_b does not change and just solve for msf, also less iterations
            # for convergence
            # """
            # initial_guess = [self.msf if self.msf is not None else self.lims_msf[0]]
            # bounds = ((self.lims_msf[0], ), (self.lims_msf[1]), )
            #
            # if initial_guess[0] == 0:
            #     initial_guess[0] = self.lims_msf[0]
            #
            # outputs = least_squares(self.energy_generation_and_storage_subproblem, initial_guess, bounds=bounds)
            # Tts_c_b = self.Tts_c[-1]
            # msf = outputs.x[0]

            Tts_c_b = self.Tts_c[-1]
            if self.Tsf_out_sp > 0.1:
                pid = copy.deepcopy(self._pid_sf) # Setpoint already established when initializaing the PID
                msf = pid(self.Tsf_out_ant, dt=self.sample_time)
            else:
                msf = self.msf # If no valid setpoint is provided, a flow was provided, otherwise the solar field would've been inactive

            cnt_max = 10

        else:
            # Should never happen since the field is validated
            raise ValueError(f"`resolution_mode` needs to be one of the supported types, not {self.resolution_mode}")

        # With this solution, we can recalculate Tsf,out and Tts_c_b, and iterate until convergence or max n of
        # iterations is reached
        Tsf_out0 = self.Tsf_out_sp if self.Tsf_out_sp > 0.1 else self.Tsf_out_ant

        Tts_c_b0 = Tts_c_b
        deltaTsf_out = 999; cnt = 0
        while abs(deltaTsf_out) > 0.1 and cnt < cnt_max or Tsf_out0 > self.lims_T_hot[1]:
            if abs(deltaTsf_out) < 0.1 or cnt >= cnt_max:
                # It means the iteration finished but resulted in an unfeasible temperature, increase the flow until
                # it is feasible
                msf += 1 # m³/h
                logger.warning(f"Unfeasible temperature ({Tsf_out0:.0f} > {self.lims_T_hot[1]}), increasing flow to {msf:.1f} (m³/h) and recalculating")
                cnt = 0


            Tsf_in, Tts_h_in = self.solve_heat_exchanger(Tsf_out0, Tts_c_b0, msf, self.mts_src, return_epsilon=False)

            Tsf_out = self.solve_solar_field(Tsf_in=Tsf_in, msf=msf, )

            Tts_h, Tts_c = self.solve_thermal_storage(Tts_h_in)
            Tts_c_b = Tts_c[-1]

            deltaTsf_out = Tsf_out - Tsf_out0
            Tsf_out0 = Tsf_out
            Tts_c_b0 = Tts_c_b
            cnt += 1

        if self.resolution_mode == 'precise':
            if cnt == cnt_max:
                logger.debug(f"Not converged in {cnt} iterations, ΔTsf,out = {deltaTsf_out:.3f}")
            else:
                logger.debug(f"Converged in {cnt} iterations")
        else:
            # logger.debug(f"Coupled subproblem, after {cnt} iterations, ΔTsf,out = {deltaTsf_out:.3f}, Tsf,out = {Tsf_out:.2f}, Tsf,out,sp = {self.Tsf_out_sp:.2f}, qsf = {self.lims_msf[0]}<{msf:.1f}<{self.lims_msf[1]} (m³/h), Tsf,in = {self.Tsf_in:.2f}")
            deltaTts_h = Tts_h - self.Tts_h
            deltaTts_c = Tts_c - self.Tts_c
            logger.debug(f"Coupled subproblem, after {cnt} iterations, ΔTsf,out = {deltaTsf_out:.3f}, qts,src = {self.mts_src_sp:.1f} (m³/h), Tts,h,in = {Tts_h_in:.2f}, qts,dis = {self.mts_dis:.1f} (m³/h), Tts,c,in = {self.Tmed_s_out:.2f}")
            logger.debug(f"Temperature change in thermal storage, ΔTts,h = {deltaTts_h}, ΔTts,c = {deltaTts_c}")

        return msf, Tsf_out, Tsf_in, Tts_h, Tts_c, Tts_h_in


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
                #     re.match(r'msf*', col) or
                #     re.match(r'mhx*', col) or
                #     re.match(r'm3wv*', col) or
                #     re.match(r'mts*', col)):
                if re.match('m(?!ed)', col):
                    data[f'q{col[1:]}'] = data[col]

        # Rename flows from m* to q*, not med
        if rename_flows:
            data.rename(columns=lambda x: re.sub('^m(?!ed)', 'q', x), inplace=True) # Peligroso

        if df is not None:
            df = pd.concat([df, data], ignore_index=True)
        else:
            df = data

        return df

    def model_dump_configuration(self):
        """
        Export model instance parameters / configuration
        Returns:
            dict of model parameters
        """
        # if self._export_fields_config is None:
        #     # Choose variable to export based on explicit property
        #     self._export_fields_config = [field for field in self.__fields__.keys() if
        #                                  self.__fields__[field].field_info.json_schema_extra.get('var_type', None) == ModelVarType.parameter]

        return self.model_dump(include=self.export_fields_config, by_alias=True)

    def model_dump_state(self, reset_samples: bool = False):
        """
        WIP

        Export an instance of the model on its current state as a json, that can be directly used to recreate a new
        working (🤞) identical instance.

        :reset_samples (bool). Whether to reset samples to 0 or keep current values. For the FSMs, if for example
        a counter like `vacuum_started_sample` is not zero, it will be set to zero and the FSM `current_sample` to the
        corresponding shifted value. NOTE: The FSM current sample is different and not in sync with the model
        `current_sample`, which will be set to zero with this argument.

        Returns:

        """

        # TODO: How to handle the FSM instances correctly? Just take the current samples? (startup_started_sample, current_sample, etc)

        return self.model_dump()


    def terminate(self):
        """
        Terminate the model and free resources. To be called when no more steps are needed.
        It just terminates the MATLAB engine, all the data and states are preserved.
        """

        self._MED_model.terminate()
