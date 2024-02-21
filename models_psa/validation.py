from typing_extensions import Annotated
from pydantic import Field, ValidationError
from loguru import logger

rangeType = Annotated[tuple[float, float], Field(..., min_items=2, max_items=2)]
conHotTemperatureType = Annotated[float, Field(..., ge=0, le=110)]

def within_range_or_make_zero(value: float, range: rangeType) -> float:

    if value < range[0]:
        logger.debug(f"Value {value} is less than {range[0]}")
        return 0.0
    elif value > range[1]:
        raise ValidationError([{"loc": ["value"], "msg": f"Value {value} is greater than {range[1]}", "type": "value_error"}])
    else:
        return value

def within_range_or_zero_or_max(value: float, range: rangeType) -> float:

    if value < range[0]:
        logger.debug(f"Value {value} is less than {range[0]}")
        return 0.0
    elif value > range[1]:
        logger.debug(f"Value {value} is greater than {range[1]}")
        return range[1]
    else:
        return value

def within_range_or_min_or_max(value: float, range: rangeType) -> float:

    if value < range[0]:
        logger.debug(f"Value {value} is less than {range[0]}")
        return range[0]
    elif value > range[1]:
        logger.debug(f"Value {value} is greater than {range[1]}")
        return range[1]
    else:
        return value