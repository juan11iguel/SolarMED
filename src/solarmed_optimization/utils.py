import time
from typing import Any
from loguru import logger
import numpy as np

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

def get_nested_attr(d: dict, attr: str) -> Any:
    """ Get nested attributes separated by a dot in a dictionary """
    keys = attr.split('.')
    for key in keys:
        d = d.get(key, None)
    return d
    
    
def forward_fill_resample(source_array: np.ndarray, target_size: int) -> np.ndarray:
    """Resample a smaller array to a larger array by forward filling

        Args:
            source_array (np.ndarray): Original array
            target_size (int): Size of the target array

        Returns:
            np.ndarray: Resampled array
            
        Example usage
            source_array = np.array([1, 2, 3])
            target_size = 10
            resampled_array = resample_array(source_array, target_size)
            assert len(resampled_array) == target_size, "Resampled array size mismatch"
    """
    if len(source_array) >= target_size:
        logger.warning(f"Source array size {len(source_array)} is greater than or equal to target size {target_size}. Returning trimmed source array")
        return source_array[:target_size]
    
    span = target_size // len(source_array)  # Integer division
    remainder = target_size % len(source_array)  # Remaining slots to fill
    
    # Repeat each element by span times
    resampled_array = np.repeat(source_array, span)
    
    # Forward fill the remaining slots if target_size isn't a multiple of source_array length
    if remainder > 0:
        resampled_array = np.concatenate((resampled_array, np.repeat(source_array[-1], remainder)))
    
    return resampled_array


def downsample_by_segments(source_array: np.ndarray, target_size: int) -> np.ndarray:
    """
    Downsamples the source array to the target size by selecting the mean value 
    in each segment of the array.

    Parameters:
        source_array (np.ndarray): The input array to be downsampled.
        target_size (int): The desired size of the output array.

    Returns:
        np.ndarray: The downsampled array, where each element is the mean value 
        of a segment from the input array.
    
    Raises:
        ValueError: If target_size is less than 1 or greater than the size of source_array.
        
    Example usage
        source_array = np.array([1, 1000, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        target_size = 5
        downsampled_array = downsample_by_segments(source_array, target_size)

        print("Original array:", source_array)
        print("Downsampled array:", downsampled_array)
        assert len(downsampled_array) == target_size, "Downsampled array size mismatch"
    """
    if target_size < 1 or target_size > len(source_array):
        raise ValueError("target_size must be between 1 and the size of the source array.")

    segment_size = len(source_array) / target_size
    return np.array([
        source_array[int(i * segment_size):int((i + 1) * segment_size)].mean()
        for i in range(target_size)
    ])
