import time
from loguru import logger

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        initial_states = kwargs.get('initial_states', None)
        max_step_idx = kwargs.get('max_step_idx', None)

        logger.info(f"Function {func.__name__} took {(end_time - start_time):.2f} seconds to run. Evaluated initial states {initial_states} for {max_step_idx} steps")
        return result
    return wrapper