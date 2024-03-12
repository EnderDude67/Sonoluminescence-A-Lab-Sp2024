import astropy.units as u
from astropy.units import Quantity

from sensor_sensitivity import sensitivity_func


def laser_image_energy(
        exposure_time: Quantity["time"],
        laser_power: Quantity["power"],
        laser_wavelength: Quantity["length"]
) -> Quantity[u.erg]:
    """
    Calculates the total energy absorbed in a photo of a laser source
    
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
    absorbed_energy: [R, G, B] erg
        The total energy absorbed by the sensor for each color channel
    """
    
    # Calculate the output energy of the laser during the photo
    laser_energy = laser_power * exposure_time
    
    # Multiply the laser energy by the sensitivity function to get the absorbed energy
    absorbed_energy = sensitivity_func(laser_wavelength) * laser_energy
    absorbed_energy = absorbed_energy.to(u.erg)
    
    return absorbed_energy