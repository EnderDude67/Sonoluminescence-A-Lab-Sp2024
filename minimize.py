import numpy as np
import astropy.units as u
from scipy.optimize import minimize

from blackbody_energy import bb_color_flux

def dimless_color_flux(temperature_K, distance_ratio):
    color_flux = bb_color_flux(temperature_K * u.K, distance_ratio)
    
    dimless = (color_flux / (u.erg / (u.s * u.cm**2))).to(u.dimensionless_unscaled)
    return dimless


def objective_function(params, data_to_match):
    temperature_K, distance_ratio = params
    color_flux = dimless_color_flux(temperature_K, distance_ratio)
    
    return np.linalg.norm(color_flux - data_to_match)


def get_bb_params(color_flux):
    data_to_match = color_flux / ((u.erg / (u.s * u.cm**2)))
    data_to_match = data_to_match.to(u.dimensionless_unscaled)
    
    initial_guess = [2700, 1]
    
    bounds=[(0, None), (0, None)]
    
    result = minimize(objective_function, initial_guess, args=(data_to_match,), bounds=bounds)
    
    # Extract optimized parameters
    optimized_temp, optimized_ratio = result.x
    
    return optimized_temp * u.K, optimized_ratio


def main():
    data_to_match = bb_color_flux(2701.0 * u.K, 0.5)
    temp, ratio = get_bb_params(data_to_match)
    
    print(temp)


if __name__ == "__main__":
    main()
