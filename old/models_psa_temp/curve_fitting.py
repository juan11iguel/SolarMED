#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:05:57 2023

@author: patomareao
"""
import numpy as np
from scipy.optimize import curve_fit

def polynomial_interpolation(x, y, x_interp, order=2):
    # Perform second-order polynomial interpolation
    coefficients = np.polyfit(x, y, order)
    interp_values = np.polyval(coefficients, x_interp)

    return interp_values


# def sigmoid(x, L, x0, k, b):
#     """
#     The parameters optimized are L, x0, k, b, who are initially assigned in p0,
#     the point the optimization starts from.
#
#     - L is responsible for scaling the output range from [0,1] to [0,L]
#     - b adds bias to the output and changes its range from [0,L] to [b,L+b]
#     - k is responsible for scaling the input, which remains in (-inf,inf)
#     - x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
#
#     """
#     y = L / (1 + np.exp(-k * (x - x0))) + b
#
#     return y
#
#
# def sigmoid_interpolation(xdata, ydata, x_interp):
#     # Perform sigmoid interpolation
#     # Should have kept the reference
#
#     p0 = [max(ydata), np.median(xdata), 1, min(ydata)]  # this is a mandatory initial guess
#     popt = curve_fit(sigmoid, xdata, ydata, p0, method='lm')
#
#     interp_values = sigmoid(x_interp, *popt)
#
#     # return interp_values
#     return interp_values

def sigmoid(x, L, x0, k):
    """
    The parameters optimized are L, x0, k, who are initially assigned in p0,
    the point the optimization starts from.

    - L is responsible for scaling the output range from [0,1] to [0,L]
    - k is responsible for scaling the input, which remains in (-inf,inf)
    - x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].

    """
    y = L / (1 + np.exp(-k * (x - x0)))

    return y

def sigmoid_interpolation(xdata, ydata, x_interp):
    # Perform sigmoid interpolation
    # Should have kept the reference

    p0 = [max(ydata), np.median(xdata), 1]  # this is a mandatory initial guess
    popt, _ = curve_fit(sigmoid, xdata, ydata, p0, method='lm')

    interp_values = sigmoid(x_interp, *popt)

    # return interp_values
    return interp_values

def exponential_curve(x, a, b, c):
    return a * np.exp(-b * x) + c

def quadratic_curve(x, a, b, c):
    return a * x**2 + b * x + c

def logarithmic_curve(x, a, b, c):
    return a * np.log(b * x) + c

def linear_fit(x, a, b):
    return a * x + b

def spline_fit(x, knots, coeffs, order):

    tk = (knots, coeffs, order)
    return splev(x, tk, der=0)

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