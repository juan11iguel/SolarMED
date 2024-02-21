import json
import re
import numpy as np
from dataclasses import dataclass, field
from iapws import IAPWS97 as w_props # Librería propiedades del agua, cuidado, P Mpa no bar
from loguru import logger
from scipy.optimize import least_squares
from enum import Enum
from typing_extensions import Annotated
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    computed_field,
    ValidationError,
    ValidationInfo,
    PositiveFloat,
)
from pathlib import Path

# from utils.curve_fitting import polynomial_interpolation
# from utils.validation import validate_input_types
# import py_validate as pv # https://github.com/gfyoung/py-validate

# MATLAB MED model
import MED_model
import matlab

from .validation import rangeType, within_range_or_zero_or_max, within_range_or_min_or_max, conHotTemperatureType
from .curve_fitting import evaluate_fit
from .solar_field import solar_field_model, solar_field_model_inverse
from .heat_exchanger import heat_exchanger_model
from .thermal_storage import thermal_storage_model_two_tanks
from .power_consumption import Actuator, SupportedActuators
from .three_way_valve import three_way_valve_model

dot = np.multiply
Nsf_max = 100  # Maximum number of steps to keep track of the solar field flow rate, should be higher than the maximum expected delay

class ts_states(Enum):
    """
    Enumerates the possible states of the thermal storage system

    Examples:
        ts_state = Ts_state.JUST_RECHARGING or Ts_state(1) or Ts_state['JUST_RECHARGING'] # <Ts_state.JUST_RECHARGING: 1>
        ts_state.value # 1
        ts_state.name # 'JUST_RECHARGING'
    """
    # name  = value
    JUST_RECHARGING = 1
    JUST_DISCHARGING = 2
    RECHARGING = 3
    DISCHARGING = 4
    IDLE = 5
    
class SolarMED_states(Enum):
    """
    Enumerates the possible states of the Multi-effect distillation pilot plant

    Examples:
        med_state = MED_state.JUST_RECHARGING or MED_state(1) or MED_state['JUST_RECHARGING'] # <MED_state.JUST_RECHARGING: 1>
        med_state.value # 1
        med_state.name # 'JUST_RECHARGING'
    """
    # name  = value

    IDLE = 1  # No operation, just losses to the environment in thermal storage and solar field
    SOLAR_FIELD = 2  # Solar field heating up, no heat transfer to thermal storage, no MED operation
    SOLAR_FIELD_THERMAL_STORAGE = 3  # Solar field providing heat to thermal storage, no MED operation
    THERMAL_STORAGE = 4 # Solar field idle, thermal storage recirculating, no MED operation
    SOLAR_FIELD_MED = 5  # Solar field heating up, no heat transfer to thermal storage, thermal storage discharging, MED operation
    SOLAR_FIELD_THERMAL_STORAGE_MED = 6  # Solar field providing heat to thermal storage, MED operation
    THERMAL_STORAGE_MED = 7  # Solar field idle, thermal storage discharging, MED operation


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
        - To export (serialize) the model, use pydantic's built in method `solar_med.model_dump()` (being solar an
        instance of the class `solar_med = SolarMED(...)`)  to produce a dictionary representation of the model.
        By default, fields are configured to be exported or not, but this can be overided with arguments to the
        method, check its [docs](https://docs.pydantic.dev/latest/concepts/serialization/).
        - Alternatively, the model can be serialized to a JSON using the `solar_med.model_dump_json()` method.
    """

    # Limits
    # Important to define first, so that they are available for validation
    ## Flows. Need to be defined separately to validate using `within_range_or_zero_or_max`
    lims_mts_src: rangeType = Field([0, 8.48], title="mts,src limits", json_schema_extra={"units": "m3/h"},
                                    description="Thermal storage heat source flow rate range (m³/h)", repr=False)
    ## Solar field, por comprobar!!
    lims_msf: rangeType = Field([5, 14], title="msf limits", json_schema_extra={"units": "m3/h"},
                                    description="Solar field flow rate range (m³/h)", repr=False)
    lims_mmed_s: rangeType = Field([5.56*3.6, 14.8*3.6], title="mmed,s limits", json_schema_extra={"units": "m3/h"},
                                    description="MED hot water flow rate range (m³/h)", repr=False)
    lims_mmed_f: rangeType = Field([5, 9], title="mmed,f limits", json_schema_extra={"units": "m3/h"},
                                    description="MED feedwater flow rate range (m³/h)", repr=False)
    lims_mmed_c: rangeType = Field([8, 21], title="mmed,c limits", json_schema_extra={"units": "m3/h"},
                                    description="MED condenser flow rate range (m³/h)", repr=False)


    # Tmed_s_in, límite dinámico
    lims_Tmed_s_in: rangeType = Field([60, 89], title="Tmed,s,in limits", json_schema_extra={"units": "C"},
                                    description="MED hot water inlet temperature range (ºC)", repr=False)
    lims_Tsf_out_sp: rangeType = Field([65, 110], title="Tsf,out setpoint limits", json_schema_extra={"units": "C"},
                                    description="Solar field outlet temperature setpoint range (ºC)", repr=False)
    ## Common
    lims_T_hot = rangeType = Field([0, 110], title="Thot* limits", json_schema_extra={"units": "C"},
                                    description="Solar field and thermal storage temperature range (ºC)", repr=False)

    # Parameters
    ## General parameters
    sample_time: float = Field(60, description="Sample rate (seg)", title="sample rate", json_schema_extra={"units": "s"})
    curve_fits_path: Path = Field(Path('data/curve_fits.json'), description="Path to the file with the curve fits", repr=False)

    ## MED
    med_actuators: list[Actuator] = Field(["med_brine_pump", "med_feed_pump",
                                    "med_distillate_pump", "med_cooling_pump", "med_heatsource_pump"],
                                     description="Actuators to estimate electricity consumption for the MED",
                                     title="MED actuators", repr=False)

    ## Thermal storage
    ts_actuators: list[Actuator] = Field(["ts_src_pump"], title="Thermal storage actuators", repr=False,
                                        description="Actuators to estimate electricity consumption for the thermal storage")

    UAts_h: list[PositiveFloat] = Field([0.0069818 , 0.00584034, 0.03041486], title="UAts,h", json_schema_extra={"units": "W/K"},
                                description="Heat losses to the environment from the hot tank (W/K)", repr=False)
    UAts_c: list[PositiveFloat] = Field([0.01396848, 0.0001    , 0.02286885], title="UAts,c", json_schema_extra={"units": "W/K"},
                                description="Heat losses to the environment from the cold tank (W/K)", repr=False)
    Vts_h: list[PositiveFloat]  = Field([5.94771006, 4.87661781, 2.19737023], title="Vts,h", json_schema_extra={"units": "m3"},
                                description="Volume of each control volume of the hot tank (m³)", repr=False)
    Vts_c: list[PositiveFloat]  = Field([5.33410037, 7.56470594, 0.90547187], title="Vts,c", json_schema_extra={"units": "m3"},
                                description="Volume of each control volume of the cold tank (m³)", repr=False)

    ## Solar field
    sf_actuators: list[Actuator] = Field(["sf_pump"], title="Solar field actuators", repr=False,
                                        description="Actuators to estimate electricity consumption for the solar field")

    beta_sf: float = Field(0.001037318, title="βsf", json_schema_extra={"units": "1/K"}, repr=False,
                           description="Solar field. Gain coefficient", gt=0)
    H_sf: float = Field(0, title="Hsf", json_schema_extra={"units": "W/m2"}, repr=False,
                        description="Solar field. Losses to the environment", ge=0)
    nt_sf: int = Field(1, title="nt,sf", repr=False,
                       description="Solar field. Number of tubes in parallel per collector. Defaults to 1.", ge=0)
    np_sf: int = Field(7 * 5, title="np,sf", repr=False,
                       description="Solar field. Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments.", ge=0)
    ns_sf: int = Field(2, title="ns,sf", repr=False,
                       description="Solar field. Number of loops in series", ge=0)
    Lt_sf: float = Field(1.15 * 20, title="Ltsf", repr=False,
                         json_schema_extra={"units": "m"}, description="Solar field. Collector tube length", gt=0)

    ## Heat exchanger
    UA_hx: float = Field(2.16e3, title="UA,hx", json_schema_extra={"units": "W/K"}, repr=False,
                         description="Heat exchanger. Heat transfer coefficient", gt=0)
    H_hx: float = Field(0.05, title="Hhx", json_schema_extra={"units": "W/m2"}, repr=False,
                        description="Heat exchanger. Losses to the environment")

    # Variables (states, outputs, decision variables, inputs, etc.)
    # Environment
    wmed_f: float = Field(35, title="wmed,f", json_schema_extra={"units": "g/kg"}, description="Environment. Seawater / MED feedwater salinity (g/kg)", gt=0)
    Tamb: float = Field(None, title="Tamb", json_schema_extra={"units": "C"}, description="Environment. Ambient temperature (ºC)", ge=-15, le=50)
    I: float = Field(None, title="I", json_schema_extra={"units": "W/m2"}, description="Environment. Solar irradiance (W/m2)", ge=0, le=2000)
    Tmed_c_in: float = Field(None, title="Tmed,c,in", json_schema_extra={"units": "C"}, description="Environment. Seawater temperature (ºC)", ge=10, le=28)

    # Thermal storage
    mts_src_sp: PositiveFloat = Field(None, title="mts,src*", json_schema_extra={"units": "m3/h"}, description="Decision variable. Thermal storage recharge flow rate (m³/h)")

    mts_src: PositiveFloat = Field(None, title="mts,src", json_schema_extra={"units": "m3/h"}, description="Output. Thermal storage recharge flow rate (m³/h)")
    mts_dis: PositiveFloat = Field(None, title="mts,dis", json_schema_extra={"units": "m3/h"}, description="Output. Thermal storage discharge flow rate (m³/h)")
    Tts_h_in: conHotTemperatureType = Field(None, title="Tts,h,in", json_schema_extra={"units": "C"}, description="Output. Thermal storage heat source inlet temperature, to top of hot tank == Thx_s_out (ºC)")
    Tts_c_in: conHotTemperatureType = Field(None, title="Tts,c,in", json_schema_extra={"units": "C"}, description="Output. Thermal storage load discharge inlet temperature, to bottom of cold tank == Tmed_s_out (ºC)")
    Tts_h: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,h", json_schema_extra={"units": "C"}, description="Output. Temperature profile in the hot tank (ºC)")
    Tts_c: list[conHotTemperatureType] | np.ndarray[conHotTemperatureType] = Field(..., title="Tts,c", json_schema_extra={"units": "C"}, description="Output. Temperature profile in the cold tank (ºC)")
    Pth_ts_in: PositiveFloat = Field(None, title="Pth,ts,in", json_schema_extra={"units": "kWth"}, description="Output. Thermal storage inlet power (kWth)")
    Pth_ts_out: PositiveFloat = Field(None, title="Pth,ts,out", json_schema_extra={"units": "kWth"}, description="Output. Thermal storage outlet power (kWth)")
    Jts_e: PositiveFloat = Field(None, title="Jts,e", json_schema_extra={"units": "kWe"}, description="Output. Thermal storage electrical power consumption (kWe)")

    # Solar field
    Tsf_out_sp: conHotTemperatureType = Field(None, title="Tsf,out*", json_schema_extra={"units": "C"}, description="Decision variable. Solar field outlet temperature (ºC)")

    Tsf_out: conHotTemperatureType = Field(None, title="Tsf,out", json_schema_extra={"units": "C"}, description="Output. Solar field outlet temperature (ºC)")
    Tsf_in: conHotTemperatureType = Field(..., title="Tsf,in", json_schema_extra={"units": "C"}, description="Output. Solar field inlet temperature (ºC)")
    msf: PositiveFloat = Field(None, title="msf", json_schema_extra={"units": "m3/h"}, description="Output. Solar field flow rate (m³/h)")
    SEC_sf: PositiveFloat = Field(None, title="SEC_sf", json_schema_extra={"units": "kWhe/kWth"}, description="Output. Solar field conversion efficiency (kWhe/kWth)")
    Jsf_e: PositiveFloat = Field(None, title="Jsf,e", json_schema_extra={"units": "kW"}, description="Output. Solar field electrical power consumption (kWe)")
    Pth_sf: PositiveFloat = Field(None, title="Pth_sf", json_schema_extra={"units": "kWth"}, description="Output. Solar field thermal power generated (kWth)")

    # MED
    mmed_s_sp: PositiveFloat = Field(None, title="mmed,s*", json_schema_extra={"units": "m3/h"}, description="Decision variable. MED hot water flow rate (m³/h)")
    mmed_f_sp: PositiveFloat = Field(None, title="mmed,f*", json_schema_extra={"units": "m3/h"}, description="Decision variable. MED feedwater flow rate (m³/h)")
    # Here absolute limits are defined, but upper limit depends on Tts_h_t
    Tmed_s_in_sp: float = Field(None, title="Tmed,s,in*", json_schema_extra={"units": "C"}, description="Decision variable. MED hot water inlet temperature (ºC)", ge=0, le=89)
    Tmed_c_out_sp: float = Field(None, title="Tmed,c,out*", json_schema_extra={"units": "C"}, description="Decision variable. MED condenser outlet temperature (ºC)", ge=0)

    mmed_s: PositiveFloat = Field(None, title="mmed,s", json_schema_extra={"units": "m3/h"}, description="Output. MED hot water flow rate (m³/h)")
    mmed_f: PositiveFloat = Field(None, title="mmed,f", json_schema_extra={"units": "m3/h"}, description="Output. MED feedwater flow rate (m³/h)")
    Tmed_s_in: float = Field(None, title="Tmed,s,in", json_schema_extra={"units": "C"}, description="Output. MED hot water inlet temperature (ºC)", ge=0, le=89)
    Tmed_c_out: float = Field(None, title="Tmed,c,out", json_schema_extra={"units": "C"}, description="Output. MED condenser outlet temperature (ºC)", ge=0)
    mmed_c: PositiveFloat = Field(None, title="mmed,c", json_schema_extra={"units": "m3/h"}, description="Output. MED condenser flow rate (m³/h)")
    Tmed_s_out: float = Field(None, title="Tmed,s,out", json_schema_extra={"units": "C"}, description="Output. MED heat source outlet temperature (ºC)")
    mmed_d: PositiveFloat = Field(None, title="mmed,d", json_schema_extra={"units": "m3/h"}, description="Output. MED distillate flow rate (m³/h)")
    mmed_b: PositiveFloat = Field(None, title="mmed,b", json_schema_extra={"units": "m3/h"}, description="Output. MED brine flow rate (m³/h)")
    Jmed_e: PositiveFloat = Field(None, title="Jmed,e", json_schema_extra={"units": "kWe"}, description="Output. MED electrical power consumption (kW)")
    Jmed_th: PositiveFloat = Field(None, title="Jmed,th", json_schema_extra={"units": "kWth"}, description="Output. MED thermal power consumption ~= Pth_ts_out (kW)")
    STEC_med: PositiveFloat = Field(None, title="STEC_med", json_schema_extra={"units": "kWhe/m3"}, description="Output. MED specific thermal energy consumption (kWhe/m³)")
    SEEC_med: PositiveFloat = Field(None, title="SEEC_med", json_schema_extra={"units": "kWhth/m3"}, description="Output. MED specific electrical energy consumption (kWhth/m³)")

    # Heat exchanger
    # Basically copies of existing variables, but with different names, no bounds checking
    Thx_p_in: conHotTemperatureType = Field(None, title="Thx,p,in", json_schema_extra={"units": "C"}, description="Output. Heat exchanger primary circuit (hot side) inlet temperature == Tsf_out (ºC)")
    Thx_p_out: conHotTemperatureType = Field(None, title="Thx,p,out", json_schema_extra={"units": "C"}, description="Output. Heat exchanger primary circuit (hot side) outlet temperature == Tsf_in (ºC)")
    Thx_s_in: conHotTemperatureType = Field(None, title="Thx,s,in", json_schema_extra={"units": "C"}, description="Output. Heat exchanger secondary circuit (cold side) inlet temperature == Tts_c_out(ºC)")
    Thx_s_out: conHotTemperatureType = Field(None, title="Thx,s,out", json_schema_extra={"units": "C"}, description="Output. Heat exchanger secondary circuit (cold side) outlet temperature == Tts_t_in (ºC)")
    mhx_p: PositiveFloat = Field(None, title="mhx,p", json_schema_extra={"units": "m3/h"}, description="Output. Heat exchanger primary circuit (hot side) flow rate == msf (m³/h)")
    mhx_s: PositiveFloat = Field(None, title="mhx,s", json_schema_extra={"units": "m3/h"}, description="Output. Heat exchanger secondary circuit (cold side) flow rate == mts_src (m³/h)")
    Pth_hx_p: PositiveFloat = Field(None, title="Pth,hx,p", json_schema_extra={"units": "kWth"}, description="Output. Heat exchanger primary circuit (hot side) power == Pth_sf (kWth)")
    Pth_hx_s: PositiveFloat = Field(None, title="Pth,hx,s", json_schema_extra={"units": "kWth"}, description="Output. Heat exchanger secondary circuit (cold side) power == Pth_ts_in (kWth)")
    eta_hx: PositiveFloat = Field(None, title="𝜂hx", json_schema_extra={"units": "-"}, description="Output. Heat exchanger efficiency (-)")

    # Three-way valve
    # Same case as with heat exchanger
    R3wv: float = Field(None, title="R3wv", json_schema_extra={"units": "-"}, description="Output. Three-way valve mix ratio (-)")
    m3wv_src: PositiveFloat = Field(None, title="m3wv,src", json_schema_extra={"units": "m3/h"}, description="Output. Three-way valve source flow rate == mts,dis (m³/h)")
    m3wv_dis: PositiveFloat = Field(None, title="m3wv,dis", json_schema_extra={"units": "m3/h"}, description="Output. Three-way valve discharge flow rate == mmed,s (m³/h)")
    T3wv_src: conHotTemperatureType = Field(None, title="T3wv,src", json_schema_extra={"units": "C"}, description="Output. Three-way valve source temperature == Tts,h,t (ºC)")
    T3wv_dis_in: conHotTemperatureType = Field(None, title="T3wv,dis,in", json_schema_extra={"units": "C"}, description="Output. Three-way valve discharge inlet temperature == Tmed,s,in (ºC)")
    T3wv_dis_out: conHotTemperatureType = Field(None, title="T3wv,dis,out", json_schema_extra={"units": "C"}, description="Output. Three-way valve discharge outlet temperature == Tmed,s,out (ºC)")

    # Others
    MED_model = Field(None, repr=False, exclude=True, description="MATLAB MED model instance")

    # Probably a deque would be a better variable type
    msf_prior: np.ndarray[PositiveFloat] = Field(None, repr=False, exclude=False, description='Solar field flow rate in the previous Nsf_max steps', max_items=Nsf_max)
    operating_state: SolarMED_states = Field(SolarMED_states.IDLE, title="operating_state",
                                             json_schema_extra={"units": "-"},
                                             description="Current operating state of the solar MED system")
    ts_state: ts_states = Field(ts_states.IDLE, title="ts_state", json_schema_extra={"units": "-"},
                                description="Current operating state of the thermal storage system")
    default_penalty: float = Field(1e6, title="penalty", json_schema_extra={"units": "u.m."}, ge=0,
                           description="Default penalty for undesired states or conditions", repr=False, exclude=True)
    penalty: float = Field(0, title="penalty", json_schema_extra={"units": "u.m."}, ge=0,
                            description="Penalty for undesired states or conditions", repr=False)
    med_active: bool = Field(False, title="med_active", json_schema_extra={"units": "-"},
                                description="Flag indicating if the MED is active", repr=True)
    sf_active: bool = Field(False, title="sf_active", json_schema_extra={"units": "-"},
                                description="Flag indicating if the solar field is active", repr=True)
    ts_recharging: bool = Field(False, title="hx_active", json_schema_extra={"units": "-"},
                                description="Flag indicating if the heat exchanger is transfering heat from solar field to thermal storage", repr=True)

    # So that fields are validated, not only when created, but every time they are set
    model_config = ConfigDict(validate_assignment=True)

    @computed_field
    def consumption_fits(self) -> dict:
        # Load electrical consumption fit curves
        try:
            with open(self.curve_fits_path, 'r') as file:
                return json.load(file)

        except FileNotFoundError:
            raise ValidationError(f'Curve fits file not found in {self.curve_fits_path}')

    @field_validator("med_actuators", "sf_actuators", "ts_actuators", mode='before')
    @classmethod
    def generate_actuators(cls, actuator_ids: SupportedActuators | list[SupportedActuators]) -> list[Actuator]:
        if isinstance(actuator_ids, str):
            actuator_ids = [actuator_ids]

        return [Actuator(id=actuator_id) for actuator_id in actuator_ids]

    # Limits validation for flows
    @field_validator("mmed_s", "mmed_f", "mmed_c", "mts_src", "msf")
    @classmethod
    def validate_flow(cls, flow: PositiveFloat, info: ValidationInfo) -> PositiveFloat:
        lims_field = "m_" + info.field_name

        return within_range_or_zero_or_max(flow, range=info.data[lims_field])

    @field_validator("Tmed_s_in_sp")
    @classmethod
    def validate_Tmed_s_in(cls, Tmed_s_in: float, info: ValidationInfo) -> float:
        # Lower limit set by pre-defined operational limit, if lower -> 0
        # Upper bound, take the lower between the hot tank top temperature and the pre-defined operational limit
        return within_range_or_zero_or_max(
            Tmed_s_in, range=( info.data["lims_Tmed_s_in"][0],
                               np.min(info.data["lims_Tmed_s_in"], info.data["Tts_h"][0]) )
        )

    @field_validator("Tmed_c_out_sp")
    @classmethod
    def validate_Tmed_c_out(cls, Tmed_c_out: float, info: ValidationInfo) -> float:
        # The upper limit is not really needed, its value is an output restrained already by mmed_c upper lower bound
        return within_range_or_min_or_max(Tmed_c_out, range=(info.data["Tmed_c_in"],
                                                             info.data["lims_T_hot"][1]))

    @field_validator("Tsf_out_sp")
    @classmethod
    def validate_Tsf_out(cls, Tsf_out: float, info: ValidationInfo) -> float:
        # Lower limit set by pre-defined operational limit, if lower -> 0
        # Upper limit set by pre-defined operational limit
        return within_range_or_zero_or_max(Tsf_out, range=(info.data["lims_Tsf_out"][0],
                                                           info.data["lims_Tsf_out"][1]))

    def __post_init__(self):

        # Initialize the MATLAB engine
        self.init_matlab_engine()

        # Make sure thermal storage state is a numpy array
        self.Tts_h = np.array(self.Tts_h, dtype=float)
        self.Tts_c = np.array(self.Tts_c, dtype=float)

    def set_operating_state(self) -> None:
        if self.med_active and self.sf_active and self.ts_recharging:
            self.operating_state = SolarMED_states.SOLAR_FIELD_THERMAL_STORAGE_MED
        elif self.med_active and self.sf_active:
            self.operating_state = SolarMED_states.SOLAR_FIELD_MED
        elif self.med_active:
            self.operating_state = SolarMED_states.MED
        elif self.sf_active:
            self.operating_state = SolarMED_states.SOLAR_FIELD
        elif self.sf_active and self.ts_recharging:
            self.operating_state = SolarMED_states.SOLAR_FIELD_THERMAL_STORAGE
        elif self.ts_recharging:
            self.operating_state = SolarMED_states.THERMAL_STORAGE
        else:
            self.operating_state = SolarMED_states.IDLE

        logger.debug(f"Operating state: {self.operating_state.name}")

    def solve_MED(self, mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float, Tmed_c_in: float):

        Tmed_c_out0 = Tmed_c_out

        if self.med_active:

            MsIn = matlab.double([mmed_s / 3.6], size=(1, 1))  # m³/h -> L/s
            TsinIn = matlab.double([Tmed_s_in], size=(1, 1))
            MfIn = matlab.double([mmed_f], size=(1, 1))
            TcwoutIn = matlab.double([Tmed_c_out], size=(1, 1))
            TcwinIn = matlab.double([Tmed_c_in], size=(1, 1))
            op_timeIn = matlab.double([0], size=(1, 1))
            # wf=wmed_f # El modelo sólo es válido para una salinidad así que ni siquiera
            # se considera como parámetro de entrada

            med_model_solved = False
            while not med_model_solved and (Tmed_c_in < Tmed_c_out < self.lims_T_hot[1]):

                try:
                    mmed_d, Tmed_s_out, mmed_c, _, _ = self.MED_model.MED_model(
                        MsIn,  # L/s
                        TsinIn,  # ºC
                        MfIn,  # m³/h
                        TcwoutIn,  # ºC
                        TcwinIn,  # ºC
                        op_timeIn,  # hours
                        nargout=5
                    )

                    med_model_solved = True
                except Exception as e:
                    # TODO: Put the right variable in the search and set the delta to the right value
                    if re.search('mmed_c', str(e)):
                        # self.penalty = self.default_penalty
                        deltaTmed_cout = 0.5  # or -0.5
                        self.logger.warning(f"Unfeasible operation in MED")
                    else:
                        raise e

                    # La dirección de cambio debería ser en función si el caudal de refrigeración es poco o demasiado
                    Tmed_c_out = np.min(Tmed_c_out + deltaTmed_cout, self.lims_T_hot[1])
                    Tmed_c_out = np.max(Tmed_c_out, Tmed_c_in)

                    TcwoutIn = matlab.double([Tmed_c_out], size=(1, 1))
            else:
                self.logger.warning('Deactivating MED due to unfeasible operation on condenser')
                self.med_active = False

            if self.med_active:

                if abs(Tmed_c_out0 - Tmed_c_out) > 0.1:
                    self.logger.debug(
                        f"MED condenser flow was out of range, changed outlet temperature from {Tmed_c_out0} to {Tmed_c_out}"
                    )

                ## Brine flow rate
                mmed_b = mmed_f - mmed_d  # m³/h

                ## MED electrical consumption
                Jmed_e = 0
                for flow, pump in zip([mmed_b, mmed_f, mmed_d, mmed_c, mmed_s], self.med_pumps):
                    Jmed_e += self.electrical_consumption(flow, self.fit_config[pump])  # kWhe

                SEEC_med = Jmed_e / mmed_d  # kWhe/m³

                ## MED STEC
                w_props_s = w_props(P=0.1, T=(Tmed_s_in + Tmed_s_out) / 2 + 273.15)
                cp_s = w_props_s.cp  # kJ/kg·K
                rho_s = w_props_s.rho  # kg/m³
                # rho_d = w_props(P=0.1, T=Tmed_c_out+273.15) # kg/m³
                mmed_s_kgs = mmed_s * rho_s / 3600  # kg/s

                Jmed_th = mmed_s_kgs * (Tmed_s_in - Tmed_s_out) * cp_s  # kWth
                STEC_med = Jmed_th / mmed_d  # kWhth/m³

        if not self.med_active:
            mmed_s = 0
            mmed_f = 0
            Tmed_s_in = 0
            Tmed_c_out = Tmed_c_in

            mmed_d = 0
            mmed_c = 0
            mmed_b = 0
            Tmed_s_out = 0

            Jmed_e = 0
            Jmed_th = 0
            SEEC_med = 0
            STEC_med = 0

        return mmed_s, mmed_f, Tmed_s_in, Tmed_c_out, mmed_d, mmed_c, mmed_b, Tmed_s_out, Jmed_e, Jmed_th, SEEC_med, STEC_med

    def solve_thermal_storage(self, Tts_h_in: float, ) -> tuple[np.ndarray[conHotTemperatureType], np.ndarray[conHotTemperatureType]]:

        Tts_h, Tts_c = thermal_storage_model_two_tanks(
            Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c, # [ºC], [ºC]
            Tt_in=Tts_h_in, # ºC
            Tb_in=self.Tmed_s_out,  # ºC
            Tamb=self.Tamb,  # ºC

            msrc=self.mts_src,  # m³/h
            mdis=self.mts_dis,  # m³/h

            UA_h=self.UAts_h,  # W/K
            UA_c=self.UAts_c,  # W/K
            Vi_h=self.Vts_h,  # m³
            Vi_c=self.Vts_c,  # m³
            ts=self.sample_time, Tmin=self.lims_Tmed_s_in[0]  # seg, ºC
        )

        return Tts_h, Tts_c

    def solve_heat_exchanger(self, Tsf_out, Tts_c_b, msf, mts_src) -> tuple[float, float]:

        Thx_p_out, Thx_s_out = heat_exchanger_model(
            Tp_in=Tsf_out,  # Solar field outlet temperature (decision variable, ºC)
            Ts_in=Tts_c_b,  # Cold tank bottom temperature (ºC)

            Qp=msf,  # Solar field flow rate (m³/h)
            Qs=mts_src,  # Thermal storage charge flow rate (decision variable, m³/h)

            Tamb=self.Tamb,  # Ambient temperature (ºC)

            UA=self.UA_hx,  # Heat transfer coefficient of the heat exchanger (W/K)
            H=self.H_hx  # Losses to the environment
        )

        return Thx_p_out, Thx_s_out

    def solve_solar_field_inverse(self):
        msf = solar_field_model_inverse()

        return msf

    def solve_solar_field(self, Tsf_in: float, msf: float):

        Tsf_out = solar_field_model(
            Tin=Tsf_in, qsf=msf, qsf_ant=self.msf_prior, I=self.I, Tamb=self.Tamb, H=self.H_sf, beta=self.beta_sf,
            nt=self.nt_sf, np=self.np_sf, ns=self.ns_sf, Lt=self.Lt_sf, sample_time=self.sample_time
        )

        return Tsf_out

    def energy_generation_and_storage_subproblem(self, inputs):
        # TODO: Allow to provide water properties as inputs so they are calculated only once

        Tts_c_b = inputs[0]
        msf = inputs[1]

        # Heat exchanger of solar field - thermal storage
        Tsf_in, Tts_t_in = heat_exchanger_model(
            Tp_in=self.Tsf_out_sp, # Solar field outlet temperature (decision variable, ºC)
            Ts_in=Tts_c_b,  # Cold tank bottom temperature (ºC)
            Qp=msf,  # Solar field flow rate (m³/h)
            Qs=self.mts_src_sp, # Thermal storage charge flow rate (decision variable, m³/h)
            Tamb=self.Tamb,
            UA=self.UA_hx,
            H=self.H_hx
        )

        # Solar field
        msf = solar_field_model_inverse(
            Tsf_in, self.Tsf_out, self.I, self.Tamb,
            beta=self.beta_sf, H=self.H_sf, nt=self.nt_sf, np=self.np_sf, ns=self.ns_sf, Lt=self.Lt_sf
        )

        # Thermal storage
        _, Tts_c = thermal_storage_model_two_tanks(
            Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c,  # [ºC], [ºC]
            Tt_in=Tts_t_in,  # ºC
            Tb_in=self.Tmed_s_out,  # ºC
            Tamb=self.Tamb,  # ºC
            msrc=self.mts_src_sp,  # m³/h
            mdis=self.mts_dis,  # m³/h
            UA_h=self.UAts_h,  # W/K
            UA_c=self.UAts_c,  # W/K
            Vi_h=self.Vts_h,  # m³
            Vi_c=self.Vts_c,  # m³
            ts=self.ts, Tmin=self.Tmed_s_in_min  # seg, ºC
        )

        return [abs(Tts_c[-1] - inputs[0]), abs(msf - inputs[1])]


    def solve_coupled_subproblem(self) -> tuple[float, float, float, np.ndarray, np.ndarray, float]:
        """
        Solve the coupled subproblem of the solar MED system

        The given Tsf_out_sp is associated with a flow, that depends on the heat exchanger.
        The heat exchanger on itself depends on the flow of the solar field and the thermal storage temperature.

        1. Find the flow of the solar field (msf) and the outlet temperature of the cold tank (Tts_c_b),
        that minimize the difference between the current Tts_c_b (it is assumed it does not change much between samples)
         and the given Tsf_out_sp.

        2. With the obtained msf and Tts_c_b, recalculate Tsf_out and Tts_c_b, and iterate until convergence
        """
        initial_guess = [self.Tts_c[-1], self.msf if self.msf is not None else self.lims_msf[0]]
        bounds = ((self.lims_T_hot[0], self.lims_msf[0]), (self.lims_T_hot[1], self.lims_msf[1]))

        outputs = least_squares(self.energy_generation_and_storage_subproblem, initial_guess, bounds=bounds)
        Tts_c_b = outputs.x[0]
        msf = outputs.x[1]

        # With this solution, we can recalculate Tsf,out and Tts_c_b, and iterate until convergence
        Tsf_out0 = self.Tsf_out_sp; Tts_c_b0 = Tts_c_b
        deltaTsf_out = 999; cnt = 0
        while abs(deltaTsf_out) > 0.1 and cnt < 10:
            Tsf_in, Tts_h_in = self.solve_heat_exchanger(Tsf_out0, Tts_c_b0, msf, self.mts_src)
            Tsf_out = self.solve_solar_field(Tsf_in, msf)
            Tts_h, Tts_c = self.solve_thermal_storage(Tts_h_in)
            Tts_c_b = Tts_c[-1]

            deltaTsf_out = abs(Tsf_out - Tsf_out0)
            Tsf_out0 = Tsf_out; Tts_c_b0 = Tts_c_b
            cnt += 1

        if cnt == 10:
            self.logger.debug(f"Not converged in {cnt} iterations, deltaTsf_out = {deltaTsf_out:.3f}")
        else:
            logger.debug(f"Converged in {cnt} iterations")

        return msf, Tsf_out, Tsf_in, Tts_h, Tts_c, Tts_h_in

    def step(self,
        mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float,  # MED decision variables
        mts_src: float,  # Thermal storage decision variables
        Tsf_out: float,  # Solar field decision variables
        Tmed_c_in: float, Tamb: float, I: float, wmed_f: float = None  # Environment variables
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
                    + wmed_f (g/kg): Seawater salinity
                    + Tamb (ºC): Ambient temperature
                    + I (W/m²): Solar irradiance

        """


        # Process inputs
        # Most of the validation is now done in the class definition
        if wmed_f is not None:
            self.wmed_f = wmed_f

        # Environment
        self.Tamb = Tamb
        self.I = I
        self.Tmed_c_in = Tmed_c_in
        # MED
        self.mmed_s_sp = mmed_s
        self.mmed_f_sp = mmed_f
        self.Tmed_s_in_sp = Tmed_s_in
        self.Tmed_c_out_sp = Tmed_c_out
        # Thermal storage
        self.mts_src_sp = mts_src
        # Solar field
        self.Tsf_out_sp = Tsf_out

        # Initialize variables
        self.penalty = 0
        self.med_active = False
        self.sf_active = False
        self.ts_recharging = False

        # Set operating mode
        # Check MED state
        if self.mmed_s_sp > 0 and self.mmed_f_sp > 0 and self.Tmed_s_in_sp > 0:
            self.med_active = True

        # Check solar field state
        if self.Tsf_out_sp > 0:
            self.sf_active = True

        # Check heat exchanger state / thermal storage state
        if self.mts_src_sp > 0:
            self.ts_recharging = True

        # Update operating mode
        self.set_operating_state()

        # Solve model for current step

        # 1st. MED
        (self.mmed_s, self.mmed_f, self.Tmed_s_in, self.Tmed_c_out, self.mmed_d, self.mmed_c, self.mmed_b,
         self.Tmed_s_out, self.Jmed_e, self.Jmed_th, self.SEEC_med, self.STEC_med) = \
            self.solve_MED(self.mmed_s_sp, self.mmed_f_sp, self.Tmed_s_in_sp, self.Tmed_c_out_sp, self.Tmed_c_in_sp)

        # Update operating mode if necessary
        self.set_operating_state()

        # 2nd. Three-way valve
        self.m3wv_src, self.R3wv = three_way_valve_model(
                Mdis=self.mmed_s, Tsrc=self.Tts_h[0], Tdis_in=self.Tmed_s_in, Tdis_out=self.Tmed_s_out
        )

        self.mts_dis = self.m3wv_src
        self.m3wv_dis = self.mmed_s

        # 3rd. Solve solar field, heat exchanger and thermal storage

        # If both the solar field is active and the thermal storage is being recharged
        # Then the system is coupled, solve coupled subproblem
        if self.ts_recharging and self.sf_active:
            self.msf, self.Tsf_out, self.Tsf_in, self.Tts_h, self.Tts_c, self.Tts_h_in = self.solve_coupled_subproblem()

        # Otherwise, solar field and thermal storage are decoupled
        # Solve each system independently
        else:
            # Solve thermal storage
            self.Tts_h, self.Tts_c = self.solve_thermal_storage()

            # Solve solar field, calculate msf, and then recalculate Tsf
            self.Tsf_out, self.Tsf_in = self.solve_solar_field()

        # Calculate additional outputs
        # Pth_sf, Jsf_e, SEC_sf
        w_props_sf = w_props(P=0.16, T=(self.Tsf_in + self.Tsf_out) / 2 + 273.15)
        self.Pth_sf = self.msf*w_props_sf.rho * (self.Tsf_out - self.Tsf_in) * w_props_sf.cp / 3600 # kWth
        # self.Jsf_e = self.electrical_consumption(self.msf, self.consumption_fits[""]) # kWhe # TODO: Add the right fit
        self.Jsf_e = 0
        self.SEC_sf = self.Jsf_e / self.Pth_sf # kWhe/kWth

        # Pth_ts_out, Pth_ts_in, Jts_e
        w_props_ts_in = w_props(P=0.16, T=(self.Tts_h_in + self.Tts_c[-1]) / 2 + 273.15)
        self.Pth_ts_in = self.mts_src*w_props_ts_in.rho * (self.Tts_c[-1] - self.Tts_h_in) * w_props_ts_in.cp / 3600 # kWth
        w_props_ts_out = w_props(P=0.16, T=(self.Tmed_s_out + self.Tts_h[1]) / 2 + 273.15)
        self.Pth_ts_out = self.mts_dis*w_props_ts_out.rho * (self.Tts_h[1] - self.Tmed_s_out) * w_props_ts_out.cp / 3600 # kWth
        # self.Jts_e = self.electrical_consumption(self.mts_src, self.consumption_fits[""]) # kWhe # TODO: Add the right fit
        self.Jts_e = 0

        # Copied variables for the heat exchanger
        self.Thx_p_in = self.Tsf_out
        self.Thx_p_out = self.Tsf_in
        self.Thx_s_in = self.Tts_c[-1]
        self.Thx_s_out = self.Tts_h_in
        self.mhx_p = self.msf
        self.mhx_s = self.mts_src
        self.Pth_hx_p = self.Pth_sf
        self.Pth_hx_s = self.Pth_ts_in

        self.eta_hx = self.Pth_hx_s / self.Pth_hx_p


    def init_matlab_engine(self):
        """
        Manually initialize the MATLAB MED model, in case it was terminated.
        """

        self.MED_model = MED_model.initialize()
        self.logger.info('MATLAB MED model initialized')

    def terminate(self):
        """
        Terminate the model and free resources. To be called when no more steps are needed.
        It just terminates the MATLAB engine, all the data and states are preserved.
        """

        self.MED_model.terminate()



@dataclass
class solar_MED:
    """
    Model of the Multi-effect distillation plant and the thermal storage system.
    
    It includes several functions:
        -  
    """
    
    # States (need to be initialized)
    ## Thermal storage
    Tts_h: list[float] # Hot tank temperature profile (ºC)
    Tts_c: list[float] # Cold tank temperature profile (ºC)
    
    
    # Parameters
    ts: float = 60 # Sample rate (seg)
    # cost_e:float = None # Cost of electricty (€/kWhe)
    # cost_w = None # Sale price of water (€/m³)
    curve_fits_path: str = 'datos/curve_fits.json' # Path to the file with the curve fits
    default_penalty: float = 1e6 # Default penalty for infeasible solutions
    
    ## MED
    # Pumps to calculate SEEC_med, must be in the same order as in step method
    med_pumps = ["brine_electrical_consumption", "feedwater_electrical_consumption", 
                 "prod_electrical_consumption", "cooling_electrical_consumption", 
                 "hotwater_electrical_consumption"]
    
    ## Thermal storage
    UAts_h: list[float] = field(default_factory=lambda: [0.0069818 , 0.00584034, 0.03041486]) # Heat losses to the environment from the hot tank (W/K)
    UAts_c: list[float] = field(default_factory=lambda: [0.01396848, 0.0001    , 0.02286885]) # Heat losses to the environment from the cold tank (W/K)
    Vts_h: list[float]  = field(default_factory=lambda: [5.94771006, 4.87661781, 2.19737023]) # Volume of each control volume of the hot tank (m³)
    Vts_c: list[float]  = field(default_factory=lambda: [5.33410037, 7.56470594, 0.90547187]) # Volume of each control volume of the cold tank (m³)
    
    ## Solar field
    beta_sf = 0.001037318
    H_sf = 0
    nt_sf=1 # Number of tubes in parallel per collector. Defaults to 1.
    np_sf=7*5 # Number of collectors in parallel per loop. Defaults to 7 packages * 5 compartments.
    ns_sf=2 # Number of loops in series
    Lt_sf=1.15*20 # Collector tube length [m].
    
    ## Heat exchanger
    UA_hx = 2.16e3 # Heat transfer coefficient of the heat exchanger [W/K]
    H_hx = 0.05 # Losses to the environment
    
    # Decision variables
    ## MED
    # Tmed_s_in: float  = None # MED hot water inlet temperature (ºC)
    # Tmed_c_out: float = None # MED condenser outlet temperature (ºC)
    # mmed_s: float = None # MED hot water flow rate (m³/h)
    # mmed_f: float = None # MED feedwater flow rate (m³/h)
    
    ## Thermal storage
    # mts_src: float # Thermal storage heat source flow rate (m³/h)
    
    # Environment
    # Tamb: float = None # Ambient temperature (ºC)
    # # I: float # Solar irradiance (W/m2)
    # Tmed_c_in: float = None # Default 20 # Seawater temperature (ºC)
    # wmed_f: float = None # Default 35 # Seawater / MED feedwater salinity (g/kg)
    
    # # Outputs
    # ## Thermal storage
    # Tts_h_t: float  = None # Temperature of the top of the hot tank (ºC)
    # Tts_t_in: float = None # Temperature of the heating fluid to top of hot tank (ºC)
    # mts_dis: float  = None # Thermal storage discharge flow rate (m³/h)
    
    # ## MED
    # mmed_c: float = None # MED condenser flow rate (m³/h)
    # Tmed_s_out: float = None # MED hot water outlet temperature (ºC)
    # STEC_med: float = None # MED specific thermal energy consumption (kWh/m³)
    # SEEC_med: float = None # MED specific electrical energy consumption (kWh/m³)
    # mmed_d: float = None # MED distillate flow rate (m³/h)
    # mmed_b: float = None # MED brine flow rate (m³/h)
    # Emed_e: float = None # MED electrical power consumption (kW)
    
    # ## Three-way valve
    # R_3wv: float = None # Three-way valve mix ratio (-)
    
    # Limits
    ## Decision variables
    ### MED
    Tmed_s_in_max: float = 90 # ºC, maximum temperature of the hot water inlet, changes dynamically with Tts_h_t
    Tmed_s_in_min: float = 60 # ºC, minimum temperature of the hot water inlet, operational limit
    
    # Tmed_c_out_max: float = 50 # ºC, maximum temperature of the condenser outlet, depends on Tmed_c_in, Mmed_c_min, Mmed_d and Pmed_c
    # Tmed_c_out_min: float = 12 # ºC, maximum temperature of the condenser outlet, depends on Tmed_c_in, Mmed_c_min, Mmed_d and Pmed_c
    
    mmed_s_max: float = 14.8*3.6 # m³/h, maximum hot water flow rate
    mmed_s_min: float = 5.56*3.6 # m³/h, minimum hot water flow rate
    
    mmed_f_max: float = 9 # m³/h, maximum feedwater flow rate
    mmed_f_min: float = 5 # m³/h, minimum feedwater flow rate
    
    mmed_d_max: float = 3.2 # m³/h, maximum distillate flow rate
    mmed_d_min: float = 1.2 # m³/h, minimum distillate flow rate
    
    mmed_b_max: float = 6   # m³/h, maximum brine flow rate
    mmed_b_min: float = 1.2 # m³/h, minimum brine flow rate
    
    mmed_c_max: float = 21 # m³/h, maximum condenser flow rate
    mmed_c_min: float = 8  # m³/h, minimum condenser flow rate
    
    ### Thermal storage
    mts_src_max: float = 8.48 # m³/h, maximum thermal st. heat source / heat ex. secondary flow rate
    mts_src_min: float = 0    # m³/h, (ZONA MUERTA: 1.81) minimum thermal st. heat source / heat ex. secondary flow rate    
    
    ### Solar field
    msf_max: float = 14 # m³/h, maximum solar field flow rate
    msf_min: float = 5   # m³/h, minimum solar field flow rate
    Tsf_max: float = 110 # ºC, maximum solar field temperature
    Tsf_min: float = 0  # ºC, minimum solar field temperature
    
    ## Environment
    Tamb_max: float = 50 # ºC, maximum ambient temperature
    Tamb_min: float = -15 # ºC, minimum ambient temperature
    Tmed_c_in_max: float = 28 # ºC, maximum temperature of the condenser inlet cooling water / seawater
    Tmed_c_in_min: float = 10 # ºC, minimum temperature of the condenser inlet cooling water / seawater
    wmed_f_min: float = 30 # g/kg, minimum salinity of the seawater / MED feedwater
    wmed_f_max: float = 90 # g/kg, maximum salinity of the seawater / MED feedwater
    Imin: float = 0 # W/m2, minimum solar irradiance
    Imax: float = 2000 # W/m2, maximum solar irradiance
    
    ## Outputs / others
    ### MED
    mmed_c_min: float = 10 # m³/h, minimum condenser flow rate
    mmed_c_max: float = 21 # m³/h, maximum condenser flow rate
    
    # Costs
    cost_w: float = 3 # €/m³, cost of water
    cost_e: float = 0.05 # €/kWh, cost of electricity

    def __setattr__(self, name, value):
        """Input validation. Check inputs are within the allowed range.
        """
        
        # Keep dataclass default input validation
        super().__setattr__(name, value)
        
        # MED
        if name == "Tmed_s_in":
            if (value < self.Tmed_s_in_min or 
                value > self.Tmed_s_in_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Tmed_s_in_min}, {self.Tmed_s_in_max}] ({value})")
        
        # elif name == "Tmed_s_out":
        #     if (value < self.Tmed_s_out_min or 
        #         value > self.Tmed_s_out_max or
        #         not isinstance(value, (int,float))):
        #         raise ValueError(f"Value of {name} must be a number within: [{self.Tmed_s_out_min}, {self.Tmed_s_out_max}] ({value})")
        
        elif name == "mmed_s":
            if value < self.mmed_s_min or not isinstance(value, (int,float) ):
                raise ValueError(f"Value of {name} must be a number within: [{self.mmed_s_min}, {self.mmed_s_max}] ({value})")
            
            elif value < self.mmed_s_min:
                self.logger.debug(f"Value of {name} ({value}) is below the minimum ({self.mts_src_min}). Deactivated MED")
        
        elif name == "mmed_f":
            if (value < self.mmed_f_min or 
                value > self.mmed_f_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.mmed_f_min}, {self.mmed_f_max}] ({value})")
        
        elif name == "mmed_c":
            if (value < self.mmed_c_min or 
                value > self.mmed_c_max or
                value < self.mmed_f or
                not isinstance(value, (int,float))):
                raise ValueError(f"""Value of {name} must be a number within: [{self.mmed_c_min}, {self.mmed_c_max}] ({value}) 
                                 and greater than mmed_f ({self.mmed_f})""")
        
        # Thermal storage
        elif name == "mts_src":
            if value > self.mts_src_max or \
                not isinstance(value, (int,float)):
                raise ValueError(f"Value of {name} must be a number within: [{self.mts_src_min}, {self.mts_src_max}] ({value})")
        
            elif value < self.mts_src_min:
                self.logger.debug(f"Value of {name} ({value}) is below the minimum ({self.mts_src_min}). Deactivated")
        
        elif name == "Tts_h":
            if np.any( np.array(value) < 0):
                raise ValueError(f"Elements of {name} must be greater than zero. It's freezing out here! ({value})")
            if hasattr(self, 'Tmed_s_in'):
                if value[0] < self.Tmed_s_in:
                    raise ValueError(f"Value of {name}_t ({value[0]}) must be greater than Tmed,s,in ({self.Tmed_s_in})")
           
            # Make sure it's a numpy array
            if not isinstance(self.Tts_h, np.ndarray):
                self.Tts_h = np.array(self.Tts_h)
        
        elif name == "Tts_c":
            if np.any( np.array(value) < 0):
                raise ValueError(f"Elements of {name} must be greater than zero. It's freezing out here! ({value})")
            
            # Make sure it's a numpy array
            if not isinstance(self.Tts_c, np.ndarray):
                self.Tts_c = np.array(self.Tts_c)
                
        elif name in ["UAts_h", "UAts_c"]:
            # To check whether all elements in list are floats
            if ( isinstance(value, list) or isinstance(value, np.ndarray) ):
                
                if not set(map(type, value)) == {float}:
                    raise TypeError(f'All elements of {name} must be floats')
                
                if np.any( np.array(value) < 0) or np.any( np.array(value) > 1):
                    raise ValueError(f"Elements of {name} must be a number within: [{0}, {1}] ({value})")
            else:
                 raise TypeError(f'{name} must be either a list of floats or a numpy array')
        
        # Environment
        elif name == "Tmed_c_in":
            if (value < self.Tmed_c_in_min or 
                value > self.Tmed_c_in_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Tmed_c_in_min}, {self.Tmed_c_in_max}] ({value})")
        
        elif name == "Tamb":
            if (value < self.Tamb_min or 
                value > self.Tamb_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Tamb_min}, {self.Tamb_max}] ({value})")
        
        elif name == "HR":
            if (value < 0 or 
                value > 100 or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{0}, {100}] ({value})")
        
        elif name == "wmed_f":
            if (value < self.wmed_f_min or 
                value > self.wmed_f_max or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.wmed_f_min}, {self.wmed_f_max}] ({value})")
        
        elif name == "I":
            if (value < self.Imin or 
                value > self.Imax or
                not isinstance(value, (int,float))):
                raise ValueError(f"Value of {name} must be a number within: [{self.Imin}, {self.Imax}] ({value})")
        
        
        # Solar field
        elif name == "msf":
            if (value > self.msf_max or
                    not isinstance(value, (int,float))):
                raise ValueError(f"The given decision variables produce unsafe operation for the solar field, {name} is above the maximum ({self.mmed_f_max})")
                # raise ValueError(f"Value of {name} must be a number within: [{self.mmed_f_min}, {self.mmed_f_max}] ({value})")
        
        
        
        # If the value is within the allowed range, set the attribute
        super().__setattr__(name, value)
    
    def __post_init__(self):
        # Initialize logger
        self.logger = logger
        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.DEBUG)
        #
        # # Filter matplotlib logging
        # logging.getLogger("matplotlib").setLevel(logging.WARNING)
        
        # Initalize MATLAB MED model
        self.MED_model = MED_model.initialize()
        self.logger.info('MATLAB MED model initialized')
        
        # Load electrical consumption fit curves
        try:
            with open(self.curve_fits_path, 'r') as file:
                    self.fit_config = json.load(file)
            self.logger.debug(f'Curve fits file loaded from {self.curve_fits_path}')
        except FileNotFoundError:
            self.logger.error(f'Curve fits file not found in {self.curve_fits_path}')
            raise
            
        self.logger.debug('Initialization completed')
         
        
        
    # def __post_init__(self):
        
    #     # Update limits that depend on other variables
    #     self.update_limits(step_done=False)
    # def check_limits(self):
    #     """
    #     Check if the current values are within the allowed range for each variable
    #     """
    #     # Iterate over all the attributes of the class and set them to their 
    #     # own value to trigger __setattr__ validation
    #     for attr in self.__dict__.keys():
    #         setattr(self, attr, getattr(self, attr))
                
    # def update_limits(self):
    #     """ Method to update the limits that depend on other variables 
    #     (dynamic restrictions)
        
    #     """
        
    #     if self.step_done:
    #         # Update limits that depend on other variables and can only be 
    #         # calculated after the step
            
    #         pass
    #     else:
    #         # Update limits that depend on other variables and can be 
    #         # calculated before the step
    #         pass
        
    #     # Check limits
    #     self.check_limits()
    
    
    def energy_generation_and_storage_subproblem(self, inputs):
        
        Tts_c_b = inputs[0]
        msf = inputs[1]
        
        # Heat exchanger of solar field - thermal storage
        Tsf_in, Tts_t_in= heat_exchanger_model(Tp_in=self.Tsf_out, # Solar field outlet temperature (decision variable, ºC)
                                               Ts_in=Tts_c_b, # Cold tank bottom temperature (ºC) 
                                               Qp=msf, # Solar field flow rate (m³/h)
                                               Qs=self.mts_src, # Thermal storage charge flow rate (decision variable, m³/h) 
                                               Tamb=self.Tamb,
                                               UA=self.UA_hx,
                                               H=self.H_hx)
        
        
        # Solar field
        msf = solar_field_model(Tsf_in, self.Tsf_out, self.I, self.Tamb, beta=self.beta_sf, H=self.H_sf, nt=self.nt_sf, np=self.np_sf, ns=self.ns_sf, Lt=self.Lt_sf)
        
        # Thermal storage
        _, Tts_c = thermal_storage_model_two_tanks(
            Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c, # [ºC], [ºC]
            Tt_in = Tts_t_in, # ºC
            Tb_in = self.Tmed_s_out, # ºC
            Tamb = self.Tamb, # ºC
            msrc = self.mts_src, # m³/h
            mdis = self.mts_dis, # m³/h
            UA_h = self.UAts_h, # W/K
            UA_c = self.UAts_c, # W/K
            Vi_h = self.Vts_h, # m³
            Vi_c = self.Vts_c, # m³
            ts=self.ts, Tmin=self.Tmed_s_in_min # seg, ºC
        )
        
        return [ abs(Tts_c[-1]-inputs[0]), abs(msf-inputs[1]) ]
        
    # def energy_generation_and_storage_subproblem_temp(self, inputs):
    #
    #     Tts_c_b = inputs[0]
    #     Tsf_out = inputs[1]
    #
    #     # Heat exchanger of solar field - thermal storage
    #     Tsf_in, Tts_t_in = heat_exchanger_model(Tp_in=Tsf_out, # Solar field outlet temperature (decision variable, ºC)
    #                                             Ts_in=Tts_c_b, # Cold tank bottom temperature (ºC)
    #                                             Qp=self.msf, # Solar field flow rate (m³/h)
    #                                             Qs=self.mts_src, # Thermal storage charge flow rate (decision variable, m³/h)
    #                                             Tamb=self.Tamb,
    #                                             UA=self.UA_hx,
    #                                             H=self.H_hx)
    #
    #     # Solar field
    #     Tsf_out = solar_field_model_temp(Tsf_in, self.msf, self.I, self.Tamb, beta=self.beta_sf, H=self.H_sf, nt=self.nt_sf, np=self.np_sf, ns=self.ns_sf, Lt=self.Lt_sf)
    #
    #     # Thermal storage
    #     _, Tts_c = thermal_storage_model_two_tanks(Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c, # [ºC], [ºC]
    #                                                    Tt_in = Tts_t_in,                    # ºC
    #                                                    Tb_in = self.Tmed_s_out,                  # ºC
    #                                                    Tamb = self.Tamb,                              # ºC
    #                                                    msrc = self.mts_src,                           # m³/h
    #                                                    mdis = self.mts_dis,                      # m³/h
    #                                                    UA_h = self.UAts_h,                       # W/K
    #                                                    UA_c = self.UAts_c,                       # W/K
    #                                                    Vi_h = self.Vts_h,                        # m³
    #                                                    Vi_c = self.Vts_c,                        # m³
    #                                                    ts=self.ts, Tmin=self.Tmed_s_in_min)      # seg, ºC
    #
    #     return [ abs(Tts_c[-1]-inputs[0]), abs(Tsf_out-inputs[1]) ]
        
    # @pv.validate_inputs(a="number", mmed_s="number", mmed_f="number", Tmed_s_in="number", Tmed_c_out="number", 
    #                     Tmed_c_in="number", wmed_f="number", mts_src="number", Tamb="number")
    def step(self, 
             mmed_s:float, mmed_f:float, Tmed_s_in:float, Tmed_c_out:float, # MED decision variables
             mts_src:float,  # Thermal storage decision variables
             Tsf_out:float,  # Solar field decision variables
             Tmed_c_in:float, wmed_f:float, Tamb:float, I:float): # Environment variables
        
        """
        Calculate model outputs given current environment variables and decision variables

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
                    + mts_src (m³/h): .... units?
                    
                    SOLAR FIELD
                    ---------------------------------------------------
                    + Tsf_out (ºC): Solar field outlet temperature
                    
                - Environment variables
                    + Tmed_c_in (ºC): Seawater temperature
                    + wmed_f (g/kg): Seawater salinity
                    + Tamb (ºC): Ambient temperature
                    
        Returns:
            _type_: _description_
        """
        
        self.penalty = 0
        
        # Update limits that depend on other variables and can be calculated before the step
        # self.step_done = False; self.update_limits()
        
        # Update class decision and environment variables values
        ## MED
        self.mmed_s = mmed_s
        self.mmed_f = mmed_f
        self.Tmed_s_in = Tmed_s_in
        self.Tmed_c_out = Tmed_c_out
        self.Tmed_c_in = Tmed_c_in
        self.wmed_f = wmed_f
        ## Thermal storage
        self.mts_src = mts_src
        ## Solar field
        self.Tsf_out = Tsf_out
        ## Environment
        self.Tamb = Tamb
        self.I = I
        
        # Make sure thermal storage state is a numpy array
        self.Tts_h = np.array(self.Tts_h)
        self.Tts_c = np.array(self.Tts_c)
        
        if self.Tmed_s_in > self.Tts_h[0]:
            self.Tmed_s_in = self.Tts_h[0]
            logger.warning(f'Hot water inlet temperature ({self.Tmed_s_in:.2f}) is higher than the top of the hot tank ({self.Tts_h[0]:.2f}). Lowering Tmed,s,in')
        
        # MED
        if (self.mmed_s > self.mmed_s_min or
            self.mmed_f > self.mmed_f_min or
            self.Tmed_s_in > self.Tmed_s_in_min):
            
            self.med_active = True
        
            MsIn = matlab.double([mmed_s/3.6], size=(1, 1)) # m³/h -> L/s
            TsinIn = matlab.double([Tmed_s_in], size=(1, 1))
            MfIn = matlab.double([mmed_f], size=(1, 1))
            TcwoutIn = matlab.double([Tmed_c_out], size=(1, 1))
            TcwinIn = matlab.double([Tmed_c_in], size=(1, 1))
            op_timeIn = matlab.double([0], size=(1, 1))
            # wf=wmed_f # El modelo sólo es válido para una salinidad así que ni siquiera 
            # se considera como parámetro de entrada
            
            med_model_solved = False
            while not med_model_solved:
            
                try:
                    self.mmed_d, self.Tmed_s_out, self.mmed_c, _, _ = self.MED_model.MED_model(MsIn,     # L/s
                                                                                               TsinIn,   # ºC
                                                                                               MfIn,     # m³/h
                                                                                               TcwoutIn, # ºC
                                                                                               TcwinIn,  # ºC
                                                                                               op_timeIn,# hours
                                                                                               nargout=5 )
                    med_model_solved = True
                except Exception as e:
                    
                    # Introducir penalización
                    """ (OLD)
                    # If the error is raied bu mmed_c being out of range
                    #if e.contains('mmed_c'):
                    #     if Tmed_c_out + 0.2 < self.Tmed_c_in_max:
                    #         Tmed_c_out += 0.2
                    #     else: 
                    #         med_model_solved = True
                    #         self.med_active = False
                         
                    # elif e.contains('mmed_c too low'):
                    #     if Tmed_c_out - 0.2 > self.Tmed_c_in_min:
                    #         Tmed_c_out -= 0.2
                    #     else: 
                    #         self.med_active = False
                    #         med_model_solved = True
                    """
                    if re.search('mmed_c', str(e)):
                        self.penalty = self.default_penalty
                        self.logger.warning(f"Unfeasible operation in MED")
                        return None, None, None, None
                    else:
                        raise e
                    TcwoutIn = matlab.double([Tmed_c_out], size=(1, 1))

            if self.med_active:

                if abs(self.Tmed_c_out - Tmed_c_out) > 0.1:
                    self.logger.debug(f"MED condenser flow out of range, changed outlet temperature from {self.Tmed_c_out} to {Tmed_c_out}") 

                ## Brine flow rate
                self.mmed_b = self.mmed_f - self.mmed_d # m³/h

                ## MED electrical consumption
                Emed_e = 0
                for flow, pump in zip([self.mmed_b, self.mmed_f, self.mmed_d, 
                                    self.mmed_c, self.mmed_s], self.med_pumps):
                    Emed_e = Emed_e + self.electrical_consumption(flow, self.fit_config[pump]) # kWhe

                self.Emed_e = Emed_e
                self.SEEC_med = Emed_e / self.mmed_d # kWhe/m³

                ## MED STEC
                w_props_s = w_props(P=0.1, T=(Tmed_s_in + self.Tmed_s_out)/2+273.15)
                cp_s = w_props_s.cp # kJ/kg·K
                rho_s = w_props_s.rho # kg/m³
                # rho_d = w_props(P=0.1, T=Tmed_c_out+273.15) # kg/m³
                mmed_s_kgs = mmed_s * rho_s / 3600 # kg/s

                self.STEC_med = mmed_s_kgs * (Tmed_s_in - self.Tmed_s_out) * cp_s / self.mmed_d # kWhth/m³

            else: self.logger.warning('Deactivating MED due to unfeasible operation on condenser')
        
        if not self.med_active:
        
            self.mmed_s = 0
            self.mmed_f = 0
            self.Tmed_c_out = Tmed_c_in
            
            self.Tmed_s_out = Tmed_s_in
            
            self.mmed_d = 0
            self.mmed_c = 0
            self.mmed_b = 0
            
            self.Emed_e = 0
            self.SEEC_med = 0
            self.STEC_med = 0
            
        # Three-way valve
        self.mts_dis, self.R_3wv = three_way_valve_model(Mdis=mmed_s, Tsrc=self.Tts_h[0], 
                                                         Tdis_in=self.Tmed_s_in, Tdis_out=self.Tmed_s_out)
        
        # Solve solar field + heat exchanger + thermal storage subproblem
        
        initial_guess = [self.Tts_c[-1], self.msf if hasattr(self, 'msf') else self.msf_min]
        bounds = ( (0, self.msf_min), (self.Tts_c[-2], self.msf_max) )
        
        outputs = least_squares(self.energy_generation_and_storage_subproblem, initial_guess,  bounds=bounds)
        Tts_c_b = outputs.x[0]
        msf = outputs.x[1]
        
        if msf < self.msf_min:
            
            # Alternativa 1
            # Introducir penalización
            self.penalty = self.default_penalty
            
            self.logger.warning(f"Unfeasible operation in solar field, msf ({msf}) is below the minimum ({self.msf_min})")
            
            return None, None, None, None
            
            # Alternativa 2
            # If the solar field flow rate is below the minimum, fix the flow rate and calculate the new lower outlet temperature
            # self.msf = self.msf_min
                        
            # initial_guess = [self.Tts_c[-1], self.Tsf_in+5]
            # bounds = ( (0, self.Tsf_in+0.5), (self.Tts_c[-2], self.Tsf_out-0.5) )
            # outputs = least_squares(self.energy_generation_and_storage_subproblem_temp, initial_guess,  bounds=bounds)
            # Tts_c_b = outputs.x[0]
            # self.Tsf_out = outputs.x[1]
            
            # self.logger.debug(f"msf ({msf}) is below the minimum ({self.msf_min}). Lowered Tsf_out to maximum achievable value ({self.Tsf_out}) with minimum flow rate ({self.msf})")

        elif msf > self.msf_max:
            
            # Alternativa 1
            # Introducir penalización
            self.penalty = self.default_penalty
            
            self.logger.warning(f"Unfeasible operation in solar field, msf ({msf}) is above the maximum ({self.msf_max})")
            
            return None, None, None, None
            
            # Alternativa 2
            # If the solar field flow rate is above the maximum, fix the flow and calculate the new higher outlet temperature
            # self.msf = self.msf_max
                        
            # initial_guess = [self.Tts_c[-1], self.Tsf_in+5]
            # bounds = ( (0, self.Tsf_in+0.5), (self.Tts_c[-2], self.Tsf_max) )
            # outputs = least_squares(self.energy_generation_and_storage_subproblem_temp, initial_guess,  bounds=bounds)
            # Tts_c_b = outputs.x[0]
            # self.Tsf_out = outputs.x[1]
            
            # self.logger.debug(f"msf ({msf}) is below the minimum ({self.msf_min}). Increased Tsf_out to maximum achievable value ({self.Tsf_out}) with maximum flow rate ({self.msf})")
        else: self.msf = msf
        
        self.Tts_c_b = Tts_c_b

        # Heat exchanger of solar field - thermal storage
        self.Tsf_in, self.Tts_t_in= heat_exchanger_model(Tp_in=self.Tsf_out, # Solar field outlet temperature (decision variable, ºC)
                                                         Ts_in=self.Tts_c_b, # Cold tank bottom temperature (ºC) 
                                                         Qp=self.msf, # Solar field flow rate (m³/h)
                                                         Qs=self.mts_src, # Thermal storage charge flow rate (decision variable, m³/h) 
                                                         Tamb=self.Tamb,
                                                         UA=self.UA_hx,
                                                         H=self.H_hx)
        
        
        # Thermal storage
        self.Tts_t, self.Tts_c = thermal_storage_model_two_tanks(Ti_ant_h=self.Tts_h, Ti_ant_c=self.Tts_c, # [ºC], [ºC]
                                                                 Tt_in = self.Tts_t_in,                    # ºC
                                                                 Tb_in = self.Tmed_s_out,                  # ºC 
                                                                 Tamb = self.Tamb,                              # ºC
                                                                 msrc = self.mts_src,                           # m³/h
                                                                 mdis = self.mts_dis,                      # m³/h
                                                                 UA_h = self.UAts_h,                       # W/K
                                                                 UA_c = self.UAts_c,                       # W/K
                                                                 Vi_h = self.Vts_h,                        # m³
                                                                 Vi_c = self.Vts_c,                        # m³
                                                                 ts=self.ts, Tmin=self.Tmed_s_in_min)      # seg, ºC
        
        ## Solar field and thermal storage electricity consumption
        Esf_ts_e = 0
        Esf_ts_e += 0.1 * self.msf
        Esf_ts_e += 0.3 * self.mts_src
        # for flow, pump in zip([self.msf, self.mts_src], self.med_pumps):
        #     Esf_ts_e += 0.05* # kWhe Temporal!!!
            # Esf_ts_e = Esf_ts_e + self.electrical_consumption(flow, self.fit_config[pump]) # kWe

        self.Esf_ts_e = Esf_ts_e
        wprops_sf = w_props(P=0.16, T=(self.Tsf_in+self.Tsf_out)/2+273.15)
        msf_kgs = self.msf * wprops_sf.rho / 3600 # [kg/m³], [m³/h] -> [kg/s]
        
        # Solar field performance metric
        self.SEC_sf = Esf_ts_e / ( msf_kgs*( self.Tsf_out - self.Tsf_in ) * wprops_sf.cp ) # [kJ/kg·K], # kWe/kWth
        
        # Update limits that depend on other variables and can only be calculated after the step
        # self.step_done=True; self.update_limits()
        
        # Store inputs as properties to have them available in get_properties
        # Done last to make sure that any model call is not using a class property
        # instead of method input
        # mmed_s, mmed_f, Tmed_s_in, Tmed_c_out, Tmed_c_in, wmed_f, mts_src, Tamb
                    
        return self.mmed_d, self.STEC_med, self.SEEC_med, self.SEC_sf
        
    def electrical_consumption(self, Q, fit_config):
        """Returns the electrical consumption (kWe) of a pump given its 
           flow rate in m^3/h.

        Args:
            Q (float): Volumentric flow rate [m³/h]
            
        Retunrs:
            power (float): Electrical power consumption [kWe]
        """
        
        power = evaluate_fit(
            x=Q, fit_type=fit_config['best_fit'], params=fit_config['params'][ fit_config['best_fit'] ]
        )
        
        return power
    
    def calculate_fixed_costs(self, ):
        # TODO
        # costs_fixed = cost_med + cost_ts + cost_sf
        return 0
    
    
    def calculate_cost(self, cost_w=None, cost_e=None):
        
        if self.penalty:
            return self.penalty
        
        else:
            self.cost_w = cost_w if cost_w else self.cost_w # €/m³
            self.cost_e = cost_e if cost_e else self.cost_e # €/kWhe
            
            # Operational costs (€/m³h)
            self.cost_op = self.cost_e * ( self.SEEC_med + self.STEC_med*self.SEC_sf )
            
            # Fixed costs (€/h)
            if not hasattr(self, 'cost_fixed'):
                self.cost_fixed = self.calculate_fixed_costs() 
            
            self.cost = ( (self.cost_w-self.cost_op)*self.mmed_d + self.cost_fixed ) * self.ts/3600
        
            return self.cost

    def get_properties(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        output = vars(self).copy()
        
        # Filter some properties
        output.pop('fit_config') if 'fit_config' in output else None
        output.pop('curve_fits_path') if 'curve_fits_path' in output else None
        output.pop('MED_model') if 'MED_model' in output else None
        output.pop('logger') if 'logger' in output else None
        
        return output

    def terminate(self):
        """
        

        Returns
        -------
        None.

        """
        
        self.MED_model.terminate()
