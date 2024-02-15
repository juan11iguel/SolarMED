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

from .validation import rangeType, within_range_or_zero_or_max, within_range_or_min_or_max
from .curve_fitting import evaluate_fit
from .solar_field import solar_field_model
from .heat_exchanger import heat_exchanger_model
from .thermal_storage import thermal_storage_model_two_tanks
from .power_consumption import Actuator, SupportedActuators

dot = np.multiply
conHotTemperatureType = Annotated[float, Field(..., ge=0, le=110)]
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
    SOLAR_FIELD_HEATING = 2  # Solar field heating up, no heat transfer to thermal storage, no MED operation
    SOLAR_FIELD_THERMAL_STORAGE = 3  # Solar field heating up, heat transfer to thermal storage, no MED operation
    SOLAR_FIELD_THERMAL_STORAGE_MED = 4  # Solar field heating up, heat transfer to thermal storage, MED operation
    THERMAL_STORAGE_MED = 5  # Solar field idle, thermal storage discharging, MED operation


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

    ## Common
    lims_T_hot = rangeType = Field([0, 110], title="T* limits", json_schema_extra={"units": "C"},
                                    description="Solar field and thermal storage temperature range (ºC)", repr=False)

    # Parameters
    ## General parameters
    ts: float = Field(60, description="Sample rate (seg)", title="sample rate", json_schema_extra={"units": "s"})
    curve_fits_path: Path = Field(Path('data/curve_fits.json'), description="Path to the file with the curve fits", repr=False)
    default_penalty: float = Field(1e6, description="Default penalty for infeasible solutions",
                                   title="default penalty", json_schema_extra={"units": "u.m."}, repr=False)

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
    Tts_h: list[conHotTemperatureType]  = Field(..., title="Tts,h", json_schema_extra={"units": "C"}, description="Output. Temperature profile in the hot tank (ºC)")
    Tts_c: list[conHotTemperatureType] = Field(..., title="Tts,c", json_schema_extra={"units": "C"}, description="Output. Temperature profile in the cold tank (ºC)")
    mts_src: PositiveFloat = Field(None, title="mts,src*", json_schema_extra={"units": "m3/h"}, description="Decision variable. Thermal storage recharge flow rate (m³/h)")

    # Solar field
    Tsf_out: conHotTemperatureType = Field(None, title="Tsf,out*", json_schema_extra={"units": "C"}, description="Decision variable. Solar field outlet temperature (ºC)")

    Tsf_in: conHotTemperatureType = Field(..., title="Tsf,in", json_schema_extra={"units": "C"}, description="Output. Solar field inlet temperature (ºC)")
    msf: PositiveFloat = Field(None, title="msf", json_schema_extra={"units": "m3/h"}, description="Output. Solar field flow rate (m³/h)")
    SEC_sf: PositiveFloat = Field(None, title="SEC_sf", json_schema_extra={"units": "kWhe/kWth"}, description="Output. Solar field conversion efficiency (kWhe/kWth)")
    Jsf_e: PositiveFloat = Field(None, title="Jsf,e", json_schema_extra={"units": "kW"}, description="Output. Solar field electrical power consumption (kWe)")

    # MED
    mmed_s: PositiveFloat = Field(None, title="mmed,s*", json_schema_extra={"units": "m3/h"}, description="Decision variable. MED hot water flow rate (m³/h)")
    mmed_f: PositiveFloat = Field(None, title="mmed,f*", json_schema_extra={"units": "m3/h"}, description="Decision variable. MED feedwater flow rate (m³/h)")
    # Here absolute limits are defined, but upper limit depends on Tts_h_t
    Tmed_s_in: float = Field(None, title="Tmed,s,in*", json_schema_extra={"units": "C"}, description="Decision variable. MED hot water inlet temperature (ºC)", ge=0, le=89)
    Tmed_c_out: float = Field(None, title="Tmed,c,out*", json_schema_extra={"units": "C"}, description="Decision variable. MED condenser outlet temperature (ºC)", ge=0)

    mmed_c: PositiveFloat = Field(None, title="mmed,c", json_schema_extra={"units": "m3/h"}, description="Output. MED condenser flow rate (m³/h)")
    Tmed_s_out: float = Field(None, title="Tmed,s,out", json_schema_extra={"units": "C"}, description="Output. MED heat source outlet temperature (ºC)")
    mmed_d: PositiveFloat = Field(None, title="mmed,d", json_schema_extra={"units": "m3/h"}, description="Output. MED distillate flow rate (m³/h)")
    mmed_b: PositiveFloat = Field(None, title="mmed,b", json_schema_extra={"units": "m3/h"}, description="Output. MED brine flow rate (m³/h)")
    Jmed_e: PositiveFloat = Field(None, title="Jmed,e", json_schema_extra={"units": "kWe"}, description="Output. MED electrical power consumption (kW)")
    Jmed_th: PositiveFloat = Field(None, title="Jmed,th", json_schema_extra={"units": "kWth"}, description="Output. MED thermal power consumption (kW)")
    STEC_med: PositiveFloat = Field(None, title="STEC_med", json_schema_extra={"units": "kWhe/m3"}, description="Output. MED specific thermal energy consumption (kWhe/m³)")
    SEEC_med: PositiveFloat = Field(None, title="SEEC_med", json_schema_extra={"units": "kWhth/m3"}, description="Output. MED specific electrical energy consumption (kWhth/m³)")

    # Heat exchanger
    # Basically copies of existing variables, but with different names, no bounds checking
    Thx_p_in: conHotTemperatureType = Field(None, title="Thx,p,in", json_schema_extra={"units": "C"}, description="Output. Heat exchanger primary circuit (hot side) inlet temperature (ºC)")
    Thx_p_out: conHotTemperatureType = Field(None, title="Thx,p,out", json_schema_extra={"units": "C"}, description="Output. Heat exchanger primary circuit (hot side) outlet temperature (ºC)")
    Thx_s_in: conHotTemperatureType = Field(None, title="Thx,s,in", json_schema_extra={"units": "C"}, description="Output. Heat exchanger secondary circuit (cold side) inlet temperature (ºC)")
    Thx_s_out: conHotTemperatureType = Field(None, title="Thx,s,out", json_schema_extra={"units": "C"}, description="Output. Heat exchanger secondary circuit (cold side) outlet temperature (ºC)")
    mhx_p: PositiveFloat = Field(None, title="mhx,p", json_schema_extra={"units": "m3/h"}, description="Output. Heat exchanger primary circuit (hot side) flow rate (m³/h)")
    mhx_s: PositiveFloat = Field(None, title="mhx,s", json_schema_extra={"units": "m3/h"}, description="Output. Heat exchanger secondary circuit (cold side) flow rate (m³/h)")

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
    msf_prior: float = Field(None, repr=False, exclude=False, description='Solar field flow rate in the previous Nsf_max steps', max_items=Nsf_max)


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

    @field_validator("Tmed_s_in")
    @classmethod
    def validate_Tmed_s_in(cls, Tmed_s_in: float, info: ValidationInfo) -> float:
        # Lower limit set by pre-defined operational limit, if lower -> 0
        # Upper bound, take the lower between the hot tank top temperature and the pre-defined operational limit
        return within_range_or_zero_or_max(
            # The upper limit is not really needed, its value is an output restrained already by mmed_c upper lower bound
            Tmed_s_in, range=( info.data["lims_Tmed_s_in"][0], info.data["Tmed_s_in"][1]-10 )
        )

    @field_validator("Tmed_c_out")
    @classmethod
    def validate_Tmed_c_out(cls, Tmed_c_out: float, info: ValidationInfo) -> float:
        return within_range_or_min_or_max(Tmed_c_out, range=(info.data["Tmed_c_in"],
                                                             info.data["lims_T_hot"][1]))

    @field_validator("Tsf_out")
    @classmethod
    def validate_Tsf_out(cls, Tsf_out: float, info: ValidationInfo) -> float:
        return within_range_or_min_or_max(Tsf_out, range=(info.data["Tsf_in"],
                                                          info.data["lims_T_hot"][1]))

    def __post_init__(self):

        # Initialize the MATLAB engine
        self.init_matlab_engine()

    def step(self, wmed_f: float = None):

        # mmed_s: float, mmed_f: float, Tmed_s_in: float, Tmed_c_out: float,  # MED decision variables
        # mts_src: float,  # Thermal storage decision variables
        # Tsf_out: float,  # Solar field decision variables
        # Tmed_c_in: float, wmed_f: float, Tamb: float, I: float):  # Environment variables

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
                    + mts_src (m³/h): Thermal storage heat source flow rate

                    SOLAR FIELD
                    ---------------------------------------------------
                    + Tsf_out (ºC): Solar field outlet temperature

                - Environment variables
                    + Tmed_c_in (ºC): Seawater temperature
                    + wmed_f (g/kg): Seawater salinity
                    + Tamb (ºC): Ambient temperature
                    + I (W/m²): Solar irradiance

        Returns:
            _type_: _description_
        """


        # Process inputs
        if wmed_f is not None:
            self.wmed_f = wmed_f

        # Set operating mode

        # Solve model for current step

        # Update states

    def terminate(self):
        """
        Terminate the model and free resources. To be called when no more steps are needed.
        It just terminates the MATLAB engine, all the data and states are preserved.
        """

        self.MED_model.terminate()

    def init_matlab_engine(self):
        """
        Manually initialize the MATLAB MED model, in case it was terminated.
        """

        self.MED_model = MED_model.initialize()
        self.logger.info('MATLAB MED model initialized')


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
    
# @dataclass
# class solarMED:
#     """
#     Model of the complete system for case study 3, this includes:
#         - Solar flat-plate collector field
#         - Heat exchanger
#         - Thermal storage
#         - Three-way valve
#         - Multi-effect distillation plant
        
        
#     """
    
#     # Inputs
    
#     # MED
#     # Tmed_s_in_sp: float # MED hot water inlet temperature (ºC)
#     # Tmed_cw_out_sp: float # MED condenser outlet temperature (ºC)
#     # Mmed_s_sp: float # MED hot water flow rate (m^3/h)
#     # Mmed_f_sp: float # MED feed water flow rate (m^3/h)
    
#     # # Solar field
#     # Tsf_out_sp: float # Solar field outlet temperature (ºC)
    
#     # # Thermal storage
#     # Mtk_src_sp: float # Thermal storage heating flow rate (m^3/h)
    
#     # Environment variables
#     # Xf: float # Water salinity (g/kg)
#     # Tcwin: float # Seawater inlet temperature (ºC)
#     # Tamb: float # Ambient temperature (ºC)
#     # I: float # Solar radiation (W/m^2)
    
#     # Model outputs
#     # - Solar field
#     Tsf_in: float # Solar field inlet temperature (ºC)
    
#     # - Thermal storage
#     Tts_t_in: float # Inlet temperature to top of the tank after heat source (ºC)
#     Tts_b_out: float # Outlet temperature from bottom of the tank to heat source (ºC)
    
    
#     # Parameters
#     ts: int = 600 # Sample rate (s)
#     Ts_min: float = 50 # Minimum temperature of the hot water inlet temperature (ºC)
    
#     # - Costs
#     Ce: float # Electricity cost (€/kWh_e)
#     thermal_self_production: bool = True # Whether thermal energy is by a self-owned solar field or provided externally
#     Cw: float # Water sale price (€/m^3)
    
#     # - Fixed costs
#     cost_investment_MED: float = 0 # Investment cost of the MED (€)
#     cost_investment_solar_field: float = 0 # Investment cost of the MED (€)
#     cost_investment_storage: float = 0 # Investment cost of the MED (€)
#     # amortizacion, etc
    
#     # - Solar field
#     SF_beta: float = 0.0975 # Irradiance model parameter (m) # Pendiente de cambiar
#     SF_H: float = 2.2 # Thermal losses coefficient for the loop (W/ºC)
#     SF_nt: int = 1 # Number of parallel tubes per collector
#     SF_np: int = 7 # Number of parallel collectors in each loop
#     SF_ns: int = 2 # Number of serial connections of collector rows
#     SF_Lt: float = 1.94 # Length of the collector inner tube (m)
    
#     # - Thermal storage
#     TS_UA: float = 28000 # Heat transfer coefficient of the thermal storage (W/ºC)
#     TS_V: float = 30 # Total volume of the tank(s) [m^3] 
    
#     # - MED
#     med_pumps = ['brine_electrical_consumption', 'feedwater_electrical_consumption', 'prod_electrical_consumption', 'cooling_electrical_consumption', 'heatsource_electrical_consumption']
    
#     # curve_fits: dict # Dictionary with the curve fits for the electrical consumption of the pumps
#     curve_fits_path: str = 'curve_fits.json' # Path to the file with the curve fits for the electrical consumption of the pumps
    
#     # Outputs
#     # Mmed_d: float # MED distillate flow rate (m^3/h)
#     # SEEC_MED: float # Specific electric energy consumption of the MED (kWh_e/m^3)
#     # STEC_MED: float # Specific thermal energy consumption of the MED (kWh_th/m^3)
#     # SEC_SF: float # Specific electric energy consumption of the solar field (kWh_e/kWh_th)
    
#     # def __post_init__(self):
#     #     # Check initial values for attributes are within allowed limits
#     #     pass
    
#     # def __setattr__(self, name, value):
#     #     # Check values for attributes every time they are updated are within allowed limits
#     #     if name == 'value' and value < 5:
#     #         raise ValueError('Value must be greater than 5')
#     #     super().__setattr__(name, value)
    
#     def __init__(self):
        
#         # Load curve fits for the electrical consumption of the pumps
#         try:
#             with open(self.curve_fits_path, 'r') as file:
#                     self.curve_fits = json.load(file)
#         except FileNotFoundError:
#             logger.error(f'Curve fits file not found in {self.curve_fits_path}')
#             raise
        
                
#     def electrical_consumption(Q, fit_config=None):
#         """Returns the electrical consumption (kWe) of a pump given the flow rate in m^3/h.

#         Args:
#             Q (float): Volumentric flow rate [m^3/h]
#         """
        
#         power = evaluate_fit( x=Q, fit_type=fit_config['best_fit'], params=fit_config['params'][fit_config['best_fit']] )
        
#         return power
        
    
#     def calculate_fixed_costs(self):
#         pass
    
#     def get_thermal_energy_cost(self):
#         pass
        
#         # return self.cost_investment_MED
    
#     def calculate_cost(self, Mmed_d, SEEC_MED, STEC_MED, SEC_SF):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
        
#         if self.thermal_self_production:
#             Cop = self.Ce*( SEEC_MED + SEC_SF*STEC_MED )
#         else:
#             Cop = self.Ce*SEEC_MED + self.get_thermal_energy_cost()*STEC_MED
            
#         Cfixed = self.calculate_fixed_costs()
        
#         return (self.Cw - Cop)*Mmed_d - Cfixed
    
    
#     def update(self, I, Tcwin, Tamb, Xf, Tmed_s_in, Tmed_cw_out, Mmed_s, Mmed_f, Tsf_out, Mts_src):
#         """Calculates the outputs of the system based on the current state of the environment
#         variables, the new setpoints and the last state of the system.

#         Args:
#             I (_type_): _description_
#             Tcwin (_type_): _description_
#             Tamb (_type_): _description_
#             Xf (_type_): _description_
#             Tmed_s_in (_type_): _description_
#             Tmed_cw_out (_type_): _description_
#             Mmed_s (_type_): _description_
#             Mmed_f (_type_): _description_
#             Tsf_out (_type_): _description_
#             Mtk_src (_type_): _description_
            
#         Returns:
#             _type_: _description_
#         """
#         # Check inputs are within allowed limits
#         # Solar irradiance
        
#         # Cooling water inlet temperature
        
#         # Ambient temperature
        
#         # Water salinity
        
#         # MED hot water inlet temperature
        
#         # MED condenser outlet temperature
        
#         # MED hot water flow rate
        
#         # MED feed water flow rate
        
#         # Solar field outlet temperature
        
#         # Thermal storage heating flow rate
        
        
#         # Solar field
#         Msf = solar_field(I, self.Tsf_in, Tsf_out, Tamb, beta=self.SF_beta, H=self.SF_H, 
#                           nt=self.SF_nt, np=self.SF_np, ns=self.SF_ns, Lt=self.SF_Lt)
        
#         # Heat exchanger
#         self.Tsf_in, self.Tts_t_in, Phx_in, Phx_out = heat_exchanger(Tp_in=Tsf_out, Ts_in=self.Tts_b_out, 
#                                                                      Qp=Msf, Qs=Mts_src, UA=self.HX_UA)
        
#         SEC_SF = Phx_in / self.electrical_consumption(Msf, fit_config=self.curve_fits['solarfield_electrical_consumption'])
                
#         # Thermal storage
#         # Por actualizar
#         self.Tts_t, self.Tts_b, self.Ets_net = thermal_storage(self.Tts_t_in, self.Tmed_s_out, Mts_src, self.M3wv, 
#                                                                self.Ts_min, Tamb, UA=self.TS_UA, V=self.TS_V)
        
#         # Three-way valve
#         self.M3wv = three_way_valve(Mmed_s, Tmed_s_in, self.Tmed_s_out, self.Tts_t)
        
#         # MED
#         Mmed_d, self.Tmed_s_out, Mmed_c, STEC_MED = MED(Ms=Tmed_s_in, Ts_in=Tmed_s_in, Mf=Mmed_f, Tcw_out=Tmed_cw_out)
#         Mmed_b = Mmed_f - Mmed_d
        
#         med_power_e = 0
#         for flow, pump in zip([Mmed_b, Mmed_f, Mmed_d, Mmed_c, Mmed_s], self.med_pumps):
#             med_power_e = med_power_e + self.electrical_consumption(flow, fit_config=self.curve_fits[pump])
        
#         SEEC_MED = med_power_e / Mmed_d

#         return Mmed_d, SEEC_MED, STEC_MED, SEC_SF
