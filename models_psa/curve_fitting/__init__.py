from models_psa.curve_fitting.curves import *


def evaluate_fit(x, fit_type, params):

    if fit_type == 'linear_fit':
        return linear_fit(x, *params)
    elif fit_type == 'spline_fit':
        return spline_fit(x, *params)
    elif fit_type == 'exponential_curve':
        return exponential_curve(x, *params)
    elif fit_type == 'quadratic_curve':
        return quadratic_curve(x, *params)
    elif fit_type == 'logarithmic_curve':
        return logarithmic_curve(x, *params)
    else:
        raise ValueError(f'Unknown fit type: {fit_type}')


def ensure_monotony(data: np.array, gap=0.01) -> np.array:
    """
    This function ensures that the given data is monotonically increasing. If a value is found to be less than its
    predecessor, it is replaced with the value of its predecessor plus a small gap.

    Parameters:
    data (np.array): The input data array that needs to be checked for monotonicity.
    gap (float, optional): The minimum difference between consecutive values in the output array. Default is 0.01.

    Returns:
    np.array: The input data array adjusted to ensure monotonicity.
    """

    # Iterate over the data array starting from the second element
    for i in range(1, len(data)):
        # If the current element is less than its predecessor
        if data[i] < data[i - 1]:
            # Replace it with the value of its predecessor plus the gap
            data[i] = data[i - 1] + gap

    return data
