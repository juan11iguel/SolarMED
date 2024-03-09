import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

sns.set_theme()
myFmt = mdates.DateFormatter('%H:%M')
plot_colors = sns.color_palette()

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats = ['%y',  # ticks are mostly years
                     '%b',  # ticks are mostly months
                     '%d',  # ticks are mostly days
                     '%H:%M',  # hrs
                     '%H:%M',  # min
                     '%S.%f', ]  # secs
# these are mostly just the level above...
formatter.zero_formats = [''] + formatter.formats[:-1]
# ...except for ticks that are mostly hours, then it is nice to have
# month-day:
formatter.zero_formats[3] = '%d-%b'

formatter.offset_formats = ['',
                            '%Y',
                            '%b %Y',
                            '%d %b %Y',
                            '%d %b %Y',
                            '%d %b %Y %H:%M', ]



# def filter_nan(data):
#     data_temp = data.dropna(how='any')
#     if len(data_temp) < len(data):
#         print(f"Some rows contain NaN values and were removed ({len(data) - len(data_temp)}).")
#
#     return data_temp


def generate_env_variables(base_data_path, sample_period=pd.Timedelta(days=9), sample_rate: int = 30,
                           visualize_result=True, variables: list = None) -> pd.DataFrame:
    # Define the given sample length
    # given_sample_length = pd.Timedelta(days=9)  # Adjust this value as needed
    # base_timeseries = pd.read_csv(base_data_path, parse_dates=['time'], date_parser=date_parser)
    base_timeseries = pd.read_csv(base_data_path, parse_dates=['time'], date_format='ISO8601')

    # Convert the time column to datetime format and set it as the index
    base_timeseries.set_index('time', inplace=True)

    # Select variables from "variables"
    if variables is not None:
        base_timeseries = base_timeseries[variables]

    # Get the first and last timestamp in the selected data
    start_time = base_timeseries.index[0];
    end_time = base_timeseries.index[-1]

    # Calculate the time difference between start and end timestamps
    time_diff = end_time - start_time

    # Create a regular time series with the desired frequency using resample and linear interpolation
    base_timeseries = base_timeseries.resample(f'{sample_rate}S').interpolate(method='linear')

    # Calculate the number of periods that fit in the given sample length
    num_periods = sample_period // time_diff

    # Calculate the remaining duration after the last complete period
    remaining_duration = sample_period - (num_periods * time_diff)

    # Create a new time index spanning the desired duration
    new_time_index = pd.date_range(start=start_time, end=start_time + sample_period, freq=f'{sample_rate}S')

    # Repeat the selected data as many times as needed to cover the desired duration
    if num_periods > 0:
        output_timeseries = pd.concat(
            [base_timeseries.set_index(base_timeseries.index + i * time_diff) for i in range(num_periods)])
        # Adjust the time component of the end portion to accumulate over the previous value
        end_portion = base_timeseries.iloc[:int(remaining_duration.total_seconds() // sample_rate)].copy()
        end_portion.index = new_time_index[-len(end_portion):]

        # Concatenate the repeated selected data with the end portion
        output_timeseries = pd.concat([output_timeseries, end_portion])
    else:
        # Just take a slice from the base data
        output_timeseries = base_timeseries.iloc[0:len(new_time_index)]

    if visualize_result:
        num_cols = 2
        num_rows = base_timeseries.shape[1] // num_cols
        num_rows = num_rows + 1 if base_timeseries.shape[1] % num_cols > 0 else num_rows

        # Create the subplots with shared x-axis
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, figsize=(10, 6))

        # Loop through each column and plot in the corresponding subplot
        for i, column in enumerate(base_timeseries.columns):
            axs[i // num_cols, i % num_cols].plot(output_timeseries.index, output_timeseries[column])
            axs[i // num_cols, i % num_cols].plot(base_timeseries.index, base_timeseries[column])

            axs[i // num_cols, i % num_cols].set_ylabel(column)

            # Format xaxis
            axs[i // num_cols, i % num_cols].xaxis.set_major_locator(locator)
            axs[i // num_cols, i % num_cols].xaxis.set_major_formatter(formatter)

        # Adjust layout and show the plot
        # plt.tight_layout()
        plt.show()

    return output_timeseries