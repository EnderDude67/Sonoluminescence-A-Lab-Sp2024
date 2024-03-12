import astropy.units as u
import astropy.constants as const
import numpy as np

from astropy.units import Quantity

from blackbody_energy import bb_image_energy
from laser_energy import laser_image_energy
from constants import COUNTS_PER_ERG


def bb_expected_sum(
        exposure_time: Quantity["time"],
        temperature: Quantity["temperature"],
        distance_ratio: Quantity["dimensionless"]
) -> Quantity["dimensionless"]:
    """
    Calculates the expected image sum based on the blackbody spectrum and the sensitivity function
    
    Parameters
    ----------
    exposure_time: time
        The exposure time of the image
    temperature: temperature
        The temperature of the blackbody emitter
    distance_ratio: dimensionless
        The ratio radius / distance, where `radius` is the radius of the object and `distance` is the
        distance from the emitter to the observer
    
    Returns
    -------
    sum: [R, G, B] dimensionless
        The estimated value of the sum of each of the RGB channels
    """
    
    energy = bb_image_energy(exposure_time, temperature, distance_ratio)
    
    # Estimates the sum based on the calibrated COUNTS_PER_ERG value
    sum = (COUNTS_PER_ERG * energy).to(u.dimensionless_unscaled)
    
    return sum


def laser_expected_sum(
        exposure_time: Quantity["time"],
        laser_power: Quantity["power"],
        laser_wavelength: Quantity["length"]
) -> Quantity["dimensionless"]:
    """
    Calculates the expected image sum based on the laser spectrum and the sensitivity function
    
    Parameters
    ----------
    exposure_time: time
        The exposure time of the image
    laser_power: power
        The output power of the laser
    laser_wavelength: length
        The output wavelength of the laser
    
    Returns
    -------
    sum: [R, G, B] dimensionless
        The estimated value of the sum of each of the RGB channels
    """
    energy = laser_image_energy(exposure_time, laser_power, laser_wavelength)
    
    return (COUNTS_PER_ERG * energy).to(u.dimensionless_unscaled)


if __name__ == "__main__":
    print(f"{COUNTS_PER_ERG=:e}")
    
    sum = laser_expected_sum(1/1017 * u.s, 450 * 10**-5 * u.uW, 632.8 * u.nm)
    
    print(sum)
    print(sum / np.max(sum) * 255)
    
    # sum = bb_expected_sum(1/1017 * u.s, 2700 * u.K, 50 * u.um / (1 * u.m))
    # print(sum)
    # print(sum / np.max(sum) * 255)
