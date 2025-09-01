from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, model_validator
from loguru import logger


SupportedActuators = Literal[
    # MED
    "med_brine_pump", "med_feed_pump", "med_distillate_pump", "med_cooling_pump", "med_heatsource_pump",
    # Thermal storage
    "ts_src_pump",
    # Solar field
    "sf_pump",
]


actuator_coefficients = {
    # Important: coefficients must be in order of increasing power
    # p(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3 + ... + cn * x^n

    # Note 2: When adding new actuator coefficients, make sure to add the actuator to the SupportedActuators type,
    # otherwise the type checker will raise an error.

    # Note 3: Importante que el ajuste se haga para que la unidad de salida sean kW, entrada en m³/h

    # cuidado, los coeficientes tal como los define electrical_consumption en calibration están al revés de cómo lo toma numpy.polyval

    # MED
    'med_brine_pump':      [0.010371467694486103, -0.025160600483389525, 0.03393870518526908], # m³/h -> kW [modeling / notebooks / model_calibrations / electrical_consumption.ipynb]
    'med_feed_pump':       [0.7035299527191431, -0.09466303549610014, 0.019077706335712326],   # m³/h -> kW [modeling / notebooks / model_calibrations / electrical_consumption.ipynb]
    'med_distillate_pump': [4.149635559273511, -3.6572156762250954, 0.9484207971761789],       # m³/h -> kW [modeling / notebooks / model_calibrations / electrical_consumption.ipynb]
    'med_cooling_pump':    [5.2178993694785625, -0.9238542100009888, 0.056680794931454774],    # m³/h -> kW [modeling / notebooks / model_calibrations / electrical_consumption.ipynb]
    'med_heatsource_pump': [0.031175213554380448, -0.01857544733009508, 0.0013320144040346285],# m³/h -> kW [modeling / notebooks / model_calibrations / electrical_consumption.ipynb]

    # Thermal storage
    'ts_src_pump': [0.04883341, -0.00695794, 0.01049481], # m³/h -> kW [20240925 gitlab.psa.es/jmserrano/aquasol-librescada](http://gitlab.psa.es/jmserrano/aquasol-librescada/-/tree/main/python)
    'FTSF001_calibration': [-0.0003028263249430021, 0.1237842280794974, -0.9609378680243738], # % -> m³/h

    # Solar field
    'sf_pump': [-8.47935428e-02, 2.29091363e-02, -8.72483421e-04, 1.29582523e-05], # m³/h -> kW [20240925-20240927 gitlab.psa.es/jmserrano/aquasol-librescada](http://gitlab.psa.es/jmserrano/aquasol-librescada/-/tree/main/python)
}


def get_coeffs(actuator_id: SupportedActuators) -> list[float]:
    return actuator_coefficients[actuator_id]

class Actuator(BaseModel):

    """

    """

    id: SupportedActuators = Field(..., description='Actuator identifier', examples=['fan_wct'], title='Actuator ID')
    coefficients: list[float] = Field(None, description='Polynomial coefficients for power consumption')

    @model_validator(mode='after')
    def get_actuator_coefficients(self) -> 'Actuator':

        if self.coefficients is None:
            self.coefficients = get_coeffs(self.id)
        else:
            # If coefficients are provided, skip the default coefficients
            logger.debug(f'Custom coefficients provided for actuator {self.id}. Skipping default coefficients.')
        return self

    def __call__(self, input: float | int | np.ndarray) -> float | np.ndarray:
        """Calculate power using a generalized polynomial expression

        Args:
            input (float | int | np.ndarray): actuator input value(s)

        Returns:
            float | np.ndarray: power consumption in kilowatts (kW)
        """
        
        # Evaluate polynomial expression using numpy
        return np.polynomial.polynomial.polyval(input, self.coefficients) # kW
