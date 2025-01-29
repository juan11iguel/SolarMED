#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:13:02 2023

@author: patomareao
"""

from functools import wraps
from typing import Any, Callable, TypeVar, get_type_hints

T = TypeVar('T')

def validate_input_types(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Get the type hints of the function
        hints = get_type_hints(func)

        # Validate positional arguments
        for i, arg in enumerate(args):
            arg_name = list(hints.keys())[i]
            if arg_name in hints and not isinstance(arg, hints[arg_name]):
                raise TypeError(f"Argument '{arg_name}' must be of type {hints[arg_name]}")

        # Validate keyword arguments
        for arg_name, arg in kwargs.items():
            if arg_name in hints and not isinstance(arg, hints[arg_name]):
                raise TypeError(f"Argument '{arg_name}' must be of type {hints[arg_name]}")

        return func(*args, **kwargs)

    return wrapper
