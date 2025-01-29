from typing_extensions import Annotated
from pydantic import Field, ValidationError
from loguru import logger
import numpy as np


rangeType = Annotated[tuple[float, float], Field(..., min_items=2, max_items=2)]

# Sometimes when using larger sample rates, the controller will produce great changes in the solar field outlet
# temperature, which makes this validation to fail, that's why it's increased to 125 to give a bit more margin
# A better alternative would be to give feedback to the controller when this happens, to re-evaluate the control action,
# or maybe just modify the control action until the output is within limits
conHotTemperatureType_upper_limit: float = 110.0
conHotTemperatureType_lower_limit: float = 0.0
conHotTemperatureType = Annotated[float, Field(..., 
                                               ge=conHotTemperatureType_lower_limit, 
                                               le=conHotTemperatureType_upper_limit)]

def check_value_single(field_value, field_name):

    output = True
    if field_value is not None:
        try:
            if len(field_value) > 1:
                logger.warning(
                    f'To a dataframe only single variables are supported (not arrays), skipping field {field_name}: {field_value}')
                output = False
        except TypeError:
            pass

    return output


def within_range_or_make_zero(value: float, range: rangeType, var_name:str= None) -> float:

    if value < range[0]:
        logger.debug(f"({var_name}) Value {value:.2f} is less than {range[0]:.2f} -> 0.0")
        return 0.0
    elif value > range[1]:
        raise ValidationError([{"loc": ["value"], "msg": f"Value {value:.2f} is greater than {range[1]}", "type": "value_error"}])
    else:
        return value

def within_range_or_zero_or_max(value: float, range: rangeType, var_name:str= None) -> float:

    if range[0] >= range[1]:
        logger.debug(f"({var_name}) Range {range} is invalid, lower bound is greater than upper bound, returning 0")
        return 0

    if value < range[0]:
        logger.debug(f"({var_name}) Value {value:.2f} is less than {range[0]:.2f} -> 0.0")
        return 0.0
    elif value > range[1]:
        logger.debug(f"({var_name}) Value {value:.2f} is greater than {range[1]:.2f} -> {range[1]:.2f}")
        return range[1]
    else:
        return value

def within_range_or_nan_or_max(value: float, range: rangeType, var_name:str= None) -> float:

    if value < range[0]:
        logger.debug(f"({var_name}) Value {value:.2f} is less than {range[0]:.2f} -> 0.0")
        return np.nan
    elif value > range[1]:
        logger.debug(f"({var_name}) Value {value:.2f} is greater than {range[1]:.2f} -> {range[1]:.2f}")
        return range[1]
    else:
        return value

def within_range_or_min_or_max(value: float, range: rangeType, var_name:str= None) -> float:

    if value < range[0]:
        logger.debug(f"({var_name}) Value {value:.2f} is less than {range[0]:.2f} -> {range[0]:.2f}")
        return range[0]
    elif value > range[1]:
        logger.debug(f"({var_name}) Value {value:.2f} is greater than {range[1]:.2f} -> {range[1]:.2f}")
        return range[1]
    else:
        return value