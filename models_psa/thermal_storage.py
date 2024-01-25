from utils.curve_fitting import sigmoid_interpolation
from iapws import IAPWS97 as w_props
import numpy as np

def calculate_stored_energy(Ti, V_i, Tmin):
    # T in ºC, V_i in m³ 
    # Perform polynomial interpolation for temperatures
    interp_volumes = np.linspace(np.min(V_i), np.max(V_i), num=100)  # Generate interpolated volume points
    interp_temperatures = sigmoid_interpolation(V_i, Ti, interp_volumes)
    
    # Calculate the energy stored above Tmin
    # Estimate specific heat capacity (cp) and density (rho)
    cp  = np.array([w_props(T=T+273.15, P=0.1).cp if T>=Tmin else 0 for T in interp_temperatures])
    rho = np.array([w_props(T=T+273.15, P=0.1).rho if T>=Tmin else 0 for T in interp_temperatures])
    
    # Calculate the temperature differences above Tmin
    temperature_diff = np.maximum(0, interp_temperatures - Tmin)
    
    # Calculate the energy
    energy = np.sum(interp_volumes * rho * cp * temperature_diff)/3600 # m³·kg/m³·KJ/KgK·K == kWh

    return energy