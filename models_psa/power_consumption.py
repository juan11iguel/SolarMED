from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, model_validator
from loguru import logger

# TODO: Update this with all curve fits, merge with 

SupportedActuators = Literal['fan_wct', 'fan_dc', 'pump_c']

# This would be used for the equivalent type hinting for uncertainty determination
# supported_instruments = Literal['pt100', 'pt1000', 'humidity_capacitive', 'vortex_flow_meter', 'paddle_wheel_flow_meter']

actuator_coefficients = {
    # Important: coefficients must be in order of increasing power
    # p(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3 + ... + cn * x^n

    # Note 2: When adding new actuator coefficients, make sure to add the actuator to the SupportedActuators type,
    # otherwise the type checker will raise an error.

    # Note 3: Importante que el ajuste se haga para que la unidad de salida sean kW

    # Cuidado! En MATLAB se definieron al revés!
    'fan_wct': [189.4, -11.54, 0.4118],
    'fan_dc': [-295.6, 48.63, -2.2, 0.04761, -0.0002431],
    'pump_c': [227.8, -38.32, 5.763, 0.1461]
}


def get_coeffs(actuator_id: SupportedActuators) -> list[float]:
    return actuator_coefficients[actuator_id]

class Actuator(BaseModel):

    """

    TODO: El problema con la implementación actual, es que si se quiere usar otro actuador dando directamente los coeficientes,
    daría error por no ser uno de los soportados, habría que añadir una lógica para que si se proveen los coeficientes,
    no se haga la comprobación de si es uno de los soportados.

    """

    id: SupportedActuators = Field(..., description='Actuator identifier', example='fan_wct', title='Actuator ID')
    coefficients: list[float] = Field(None, description='Polynomial coefficients for power consumption')

    @model_validator(mode='after')
    def get_actuator_coefficients(self) -> 'Actuator':

        if self.coefficients is None:
            self.coefficients = get_coeffs(self.id)

        # If coefficients are provided, skip the default coefficients
        logger.debug(f'Custom coefficients provided for actuator {self.id}. Skipping default coefficients.')
        return self

    def calculate_power_consumption(self, input: float | int | np.ndarray) -> float | np.ndarray:

        # Calculate power using a generalized polynomial expression
        # power = sum(coeff * input ** order for order, coeff in enumerate(coefficients)) / 1000  # kW

        # Evaluate polynomial expression using numpy
        return np.polynomial.polynomial.polyval(input, self.coefficients) # kW


# def power_consumption(input: float | int | np.ndarray, actuator: SupportedActuators) -> float | np.ndarray:
#     """
#     Calculate power consumption based on the input and actuator type.
#
#     Parameters:
#     - input (float, int, or np.ndarray): Input value(s) representing a unit.
#     - actuator (Literal[tuple(actuator_coefficients.keys())]): Type of actuator.
#
#     Returns:
#     - float or np.ndarray: Power consumption in kilowatts (kW).
#
#     Raises:
#     - ValueError: If the actuator is not one of the supported types.
#     """
#     # input (unit) -> power (kW)
#
#     coefficients = actuator_coefficients.get(actuator)
#     if coefficients is None:
#         raise ValueError(f'Actuator must be one of supported types: {SupportedActuators}')
#
#     # Calculate power using a generalized polynomial expression
#     # power = sum(coeff * input ** order for order, coeff in enumerate(coefficients)) / 1000  # kW
#
#     # Evaluate polynomial expression using numpy
#     power = np.polynomial.polynomial.polyval(input, coefficients) / 1000
#
#     return power
