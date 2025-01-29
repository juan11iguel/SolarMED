# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "h5py",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///

"""Simple script that saves and reads the same date in three different formats:
- HD5
- Feather
- Parquet

It outputs their write time, read time and file size.

This script should be run with uv: `uv run compare_export_formats.py`
If not installed, check how to do it [here](https://docs.astral.sh/uv/getting-started/installation/) 
"""
import numpy as np
import pandas as pd
import h5py
import pyarrow.feather as feather
import pyarrow.parquet as pq
import time
import os

data_shape: tuple[int, int, int] = (100, 10, 10)
data = np.random.rand(*data_shape)
alternative_ids: tuple[str, str, str] = ("HD5", "parquet", "feather") 
write_times: dict[str, float] = {}
read_times: dict[str, float] = {}
file_sizes: dict[str, float] = {}
write_comp: dict[str, float] = {}
read_comp: dict[str, float] = {}
file_size_comp: dict[str, float] = {}

# Function to measure time and file size
def measure_time_and_size(write_func, read_func, file_path):
    start_time = time.time()
    write_func(file_path)
    write_time = time.time() - start_time

    file_size = os.path.getsize(file_path)

    start_time = time.time()
    read_func(file_path)
    read_time = time.time() - start_time

    return write_time, read_time, file_size

# HDF5
def write_hdf5(file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dataset_name', data=data)

def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        _ = f['dataset_name'][:]

alt_id: str = "HD5"
write_times[alt_id], read_times[alt_id], file_sizes[alt_id] = measure_time_and_size(write_hdf5, read_hdf5, 'data.h5')

# Parquet
def write_parquet(file_path):
    reshaped_data = data.reshape(-1, data.shape[-1])
    df = pd.DataFrame(reshaped_data)
    df.to_parquet(file_path)

def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    _ = df.values.reshape(*data_shape)

alt_id: str = "parquet"
write_times[alt_id], read_times[alt_id], file_sizes[alt_id] = measure_time_and_size(write_parquet, read_parquet, 'data.parquet')

# Feather
def write_feather(file_path):
    reshaped_data = data.reshape(-1, data.shape[-1])
    df = pd.DataFrame(reshaped_data)
    df.to_feather(file_path)

def read_feather(file_path):
    df = pd.read_feather(file_path)
    _ = df.values.reshape(*data_shape)

alt_id: str = "feather"
write_times[alt_id], read_times[alt_id], file_sizes[alt_id] = measure_time_and_size(write_feather, read_feather, 'data.feather')

# Determine the best formats
best_write_format: str = alternative_ids[int( np.argmin([*write_times.values()]) )]
best_read_format: str = alternative_ids[int( np.argmin([*read_times.values()]) )]
best_size_format: str = alternative_ids[int( np.argmin([*file_sizes.values()]) )]

# Calculate percentage improvements
def percentage_improvement(best, other):
    return ((other - best) / other) * 100

write_comp: dict[str, float] = {alt_id: percentage_improvement(write_times[best_write_format], write_times[alt_id]) for alt_id in alternative_ids if alt_id != best_write_format}
read_comp: dict[str, float] = {alt_id: percentage_improvement(read_times[best_read_format], read_times[alt_id]) for alt_id in alternative_ids if alt_id != best_read_format}
file_size_comp: dict[str, float] = {alt_id: percentage_improvement(file_sizes[best_size_format], file_sizes[alt_id]) for alt_id in alternative_ids if alt_id != best_size_format}

# Print results
for alt_id in alternative_ids:
    print(f"{alt_id} - Write time: {write_times[alt_id]:.4f}, Read time {read_times[alt_id]:.4f}s, File size: {file_sizes[alt_id]} bytes")

# Print conclusions
for topic, best_format, results_dict, comp_dict in zip(["write time (s)", "read time (s)", "file size (bytes)"], 
                                         [best_write_format, best_read_format, best_size_format], 
                                         [write_times, read_times, file_sizes],
                                         [write_comp, read_comp, file_size_comp]):
    comp_values_str: str = ''.join([f'{alt_value:.2f}%, ' for alt_value in comp_dict.values()])
    comp_ids_str: str = ''.join([f'{alt_id:}, ' for alt_id in comp_dict.keys()])

    print(f"Best {topic}: {results_dict[best_format]:.4f} for {best_format}, {comp_values_str} better than {comp_ids_str}respectively")
