#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:09:28 2023

@author: patomareao
"""

# import scipy
import numpy as np

def objective_function(parameters, model_function, *args):
    # Extract the inputs and outputs from the arguments
    # params is the fixed parameters
    # parameters are the parameters that need to be adjusted
    # params_objective_function is a dict containing optional keyword arguments 
    # used by this function
    #
    # The outputs from the function to optimize needs to be a single output that
    # can be a vector, so if the model or function has multiple outputs for 
    # different variables, modify the function so that it returns all outputs
    # packed in a vector: unified_output=True
    
    inputs, reference_outputs, params, params_objective_function = args
    
    L = len(reference_outputs)
    
    # Extract the metric from the keyword arguments
    metric = params_objective_function.get('metric', 'ITAE')
    
    # Check if recursive model
    # Recursive input should be the last input
    recursive = params_objective_function.get('recursive', False)
    
    # Check number of different outputs to evaluate, note that each output
    # can be a number or a vector of multiple values
    n_outputs = params_objective_function.get('n_outputs', 1)
    
    # Check the size of each output to evaluate
    len_outputs = params_objective_function.get('len_outputs', 1)
    
    # Check the number of different parameters to optimize, note that each 
    # parameter can be a number or a vector of multiple values
    n_parameters = params_objective_function.get('n_parameters', 1)
    
    # Define parameters depending on whether it's just one or several
    if n_parameters > 1:
        segment_size = len(parameters) // n_parameters
        remainder = len(parameters) % n_parameters
        
        parameters_ = []
        start = 0
        
        for i in range(n_parameters):
            segment_length = segment_size + (1 if i < remainder else 0)
            end = start + segment_length
            parameter = parameters[start:end]
            # if len(parameter) == 1:
            #     parameter = parameter[0] # To not input an array to a function that is probably expecting a number
            parameters_.append(parameter)
            start = end
    else:
        parameters_ = [parameters] # Wrap it in a list so when using the star 
                                   # operator it will just unpack parameters 
                                   # from [parameters], not the vales themselves
    
    # Initial checks
    # if len(len_outputs) != n_outputs:
    #     raise ValueError(f'The size of len_outputs ({len_outputs}) needs to match n_outputs ({n_outputs})')
    
    # if n_outputs > 1:
    #     predicted_outputs = np.zeros((L, n_outputs), dtype=float)        
    # else:
    #     predicted_outputs = np.zeros(L, dtype=float)
        
    if recursive: 
        if n_outputs > 1:
            previous_output = [inputs[i] for i in range(n_outputs)]    
        else:
            previous_output = [inputs[0]] 
    
    predicted_outputs = []
    for idx in range(L):
        # Thermal storage model
        # Inputs: Tt_in[idx], Tb_in[idx], Tamb[idx], msrc[idx], mdis[idx], Ti_ant, 
        # Parameters: UA0, 
        # Params: N=N
        
        # Extract inputs for the current time step
        # If recursive, the last input is not a list for each time step but 
        # just the provious output(s)
        i_start = 0 if not recursive else n_outputs
        current_inputs = [inputs[i][idx] if len(inputs[i])==L else inputs[i] for i in range(i_start, len(inputs))]
        
        if recursive: 
            current_inputs = previous_output + current_inputs
        
        predicted_outputs.append( model_function(*current_inputs, *parameters_, *params) )
        
        # Update previous output
        if recursive: 
            if n_outputs>1:
                previous_output = [predicted_outputs[idx][i] for i in range(n_outputs)] 
            else:
                previous_output = predicted_outputs[idx]
    
    # Flatten the predicted_outputs so there will be only a vector for each iteration
    # predicted_outputs = np.concatenate(np.array(predicted_outputs, dtype=object), axis=1)
    if type(predicted_outputs[0]) in [list, tuple]:
        predicted_outputs = np.array( [np.concatenate(row) for row in predicted_outputs] )
    else:
        predicted_outputs = np.array( predicted_outputs )
    # print(f'Dim predicted: {predicted_outputs.shape}, dim reference: {reference_outputs.shape}')
    
    if metric.upper() == 'IAE':
        objective = calculate_iae(predicted_outputs, reference_outputs)
    elif metric.upper() == 'ITAE':
        objective = calculate_itae(predicted_outputs, reference_outputs)
    elif metric.upper() == 'ISE':
        objective = calculate_ise(predicted_outputs, reference_outputs)
    else:
        raise ValueError(f'Unsupported metric {metric}, options are: IAE, ISE, ITAE')
        
    print(f'Iteration finished, error ({metric.upper()}): {objective:.2f} for x={np.array2string(parameters, precision=3, floatmode="fixed")}')
        
    return objective


def calculate_itae(predicted:np.array, actual:np.array):
    """
    Calculate the Integral Time-weighted Absolute Error (ITAE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.
        time (array-like): Time values.

    Returns:
        float: ITAE value.
    """
    time = np.arange(0, len(predicted))
    
    error = np.abs(predicted - actual)
    itae = np.nansum(error * time[:, np.newaxis])
        
    return itae

def calculate_ise(predicted, actual):
    """
    Calculate the Integral Square Error (ISE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: ISE value.
    """
    error = predicted - actual
    ise = np.nansum(np.square(error))
    return ise

def calculate_iae(predicted, actual):
    """
    Calculate the Integral Absolute Error (IAE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: IAE value.
    """
    error = np.abs(predicted - actual)
    iae = np.nansum(error)
    return iae

#%%
    
