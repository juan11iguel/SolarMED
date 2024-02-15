

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

    # Define an objective function to evaluate the goodness of fit
    def objective_function(params, curve_function):
        residuals = y_data - curve_function(x_data, *params)
        return np.sum(residuals ** 2)  # Sum of squared residuals

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

    def fit_curve(x_data: np.array, y_data: np.array, fit_name: str, unit='kW', visualize_result=True,
                  save_result=False, result_path=None, include_spline=False):

        """
        This function fits a curve to the given data using various curve models and returns the best fit. It also has the
        option to visualize the result and save it to a specified path.

        Parameters:
        x_data (np.array): The independent variable data.
        y_data (np.array): The dependent variable data.
        fit_name (str): The name of the fit.
        unit (str, optional): The unit of the y_data. Default is 'kW'.
        visualize_result (bool, optional): Whether to visualize the result. Default is True.
        save_result (bool, optional): Whether to save the result. Default is False.
        result_path (str, optional): The path to save the result if save_result is True. Default is None.
        include_spline (bool, optional): Whether to include spline fit in the curve models. Default is False.

        Returns:
        dict: A dictionary containing the best fit, its parameters, cost, and the original data.
        """

        # Define a list of candidate curve functions
        curve_functions = [
            exponential_curve,
            quadratic_curve,
            logarithmic_curve,
            linear_fit,
        ]

        curve_functions.append(spline_fit) if include_spline else None

        # Perform curve fitting with different curve models
        best_params = None
        best_cost = float('inf')
        params = {}
        fit = {
            'best_fit': None,
            'params': {},
            'cost': None,
            'x_data': None,
            'y_data': None
        }

        if unit == 'W':
            y_data = y_data * 1e-3
            print('Converted ydata to kW from W (·1e⁻³)')

        for curve_function in curve_functions:
            # Fit the curve to the data using curve_fit
            try:
                if curve_function.__name__ == 'linear_fit':
                    popt, _ = curve_fit(curve_function, x_data, y_data, method='lm')
                elif curve_function.__name__ == 'spline_fit':
                    popt = splrep(x_data, y_data, k=3, s=0.1)
                else:
                    popt, _ = curve_fit(curve_function, x_data, y_data)
            except RuntimeError:
                print(f'Failed attemp to fit using {curve_function.__name__}, skipping')
                params[curve_function.__name__] = None
            else:
                # Evaluate the goodness of fit using the objective function
                cost = objective_function(popt, curve_function)

                # Save params for each alternative
                if isinstance(popt, numpy.ndarray):
                    params[curve_function.__name__] = popt.tolist()
                else:
                    params[curve_function.__name__] = []
                    for p in popt:
                        if isinstance(p, numpy.ndarray):
                            params[curve_function.__name__].append(p.tolist())
                        else:
                            params[curve_function.__name__].append(p)

                # Check if this model provides a better fit
                if cost < best_cost:
                    best_params = popt
                    best_cost = cost
                    fit['best_fit'] = curve_function.__name__
                    fit['cost'] = best_cost

        fit['params'] = params
        fit['x_data'] = x_data.tolist()
        fit['y_data'] = y_data.tolist()

        # Print the best-fit parameters
        print('Best-fit parameters:')
        for i, param in enumerate(best_params):
            print(f'Parameter {i + 1}: {param}')

        if visualize_result:
            close('all')

            # Plot the data points
            plt.scatter(x_data, y_data, label='Data')

            # Plot the best-fit curve for each alternative
            x_range = np.linspace(x_data.min(), x_data.max(), 100)

            for curve_function in curve_functions:
                # if curve_function.__name__ == 'linear_fit':
                if params[curve_function.__name__] is not None:
                    y_fit = curve_function(x_range, *params[curve_function.__name__])
                    # else:
                    #     y_fit = curve_function(x_range, *best_params)
                    label = curve_function.__name__
                    plt.plot(x_range, y_fit, label=label)

            # Add labels and legend to the plot
            plt.xlabel('Independent Variable')
            plt.ylabel('Dependent Variable')
            plt.legend()

            # Display the plot
            plt.show()

        # Save results
        if save_result:
            # Read existing JSON file, if it exists
            existing_data = {}
            try:
                with open(result_path, 'r') as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                pass

            # Append new data to the existing data
            existing_data[fit_name] = fit

            # Write the serialized JSON to the file
            with open(result_path, 'w') as f:
                json.dump(existing_data, f, indent=4)

        return fit