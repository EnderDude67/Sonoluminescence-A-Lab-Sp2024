import astropy.units as u
from astropy.modeling import models
import numpy as np

from astropy.units import Quantity

from sensor_sensitivity import sensitivity_func
from constants import SENSOR_AREA


def bb_image_energy(
        exposure_time: Quantity["time"],
        temperature: Quantity["temperature"],
        distance_ratio: Quantity["dimensionless"]
) -> Quantity[u.erg]:
    """
    Calculates the total energy absorbed in a photo
    
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
    energy: [R, G, B] erg
        The total energy absorbed by the sensor for each color channel
    """
    
    color_flux = bb_color_flux(temperature, distance_ratio)
    
    energy = color_flux * SENSOR_AREA * exposure_time
    
    return energy.to(u.erg)


def bb_color_flux(
        temperature: Quantity["temperature"],
        distance_ratio: Quantity["dimensionless"]
) -> Quantity[u.erg / (u.s * u.cm**2)]:
    """
    Returns the total detected energy per color channel
    This is done by integrating the spectral sensitivity function multiplied by the blackbody spectrum
    
    Parameters
    ----------
    temperature: temperature
        The temperature of the blackbody emitter
    distance_ratio: dimensionless
        The ratio `radius / distance`, where `radius` is the radius of the object and `distance` is the
        distance from the emitter to the observer
    
    Returns
    -------
    color_flux: [R, G, B] energy flux
        An array of the energy fluxes of the three color channels
    """
    
    # Define the interval over which the spectrum * sensitivity will be integrated
    sample_density = 0.1 * u.nm
    start = 400 * u.nm
    stop = 700 * u.nm
    
    # Generates the points that the function will be integrated on
    num_samples = (stop - start) / sample_density + 1
    wavelengths = np.linspace(start, stop, int(num_samples), endpoint=True)
    
    # Evaluates the blackbody spectrum on the range of wavelengths
    sample_points = bb_specific_absorbed_flux(wavelengths, temperature, distance_ratio)
    
    # Evaluates the integral
    color_flux = np.trapz(sample_points, x=wavelengths, axis=0)
    color_flux.to(u.erg / (u.s * u.cm**2))
    
    return color_flux


def bb_specific_absorbed_flux(
        wavelength: Quantity["length"],
        temperature: Quantity["temperature"],
        distance_ratio: Quantity["dimensionless"]
) -> Quantity[u.erg / (u.s * u.cm**2 * u.nm)]:
    """
    Calculates the energy flux absorbed by the sensor for a small wavelength interval
    
    Parameters
    ----------
    wavelength: length
        The wavelength of light for which to calculate the specific flux
    temperature: temperature
        The temperature of the blackbody emitter
    distance_ratio: dimensionless
        The ratio radius / distance, where `radius` is the radius of the object and `distance` is the
        distance from the emitter to the observer
    
    Returns
    -------
    detected_flux: [R, G, B] energy flux per wavelength interval
        For each color channel, the absorbed specific flux of the given wavelength
    """
    
    # Evaluate the [R, G, B] sensitivity function on the wavelength
    sensitivity = sensitivity_func(wavelength)
    
    # Calculate the specific energy flux of the wavelength
    flux = bb_specific_flux(wavelength, temperature, distance_ratio)
    flux = np.array([1]) * flux
    flux = flux[:, np.newaxis]
    
    # Multiply the sensitivity function by the energy flux to get the detected flux
    detected_flux = flux * sensitivity
    detected_flux.to(u.erg / (u.s * u.cm**2 * u.nm))
    
    return detected_flux


def bb_specific_flux(
        wavelength: Quantity["length"],
        temperature: Quantity["temperature"],
        distance_ratio: Quantity["dimensionless"]
) -> Quantity[u.erg / (u.s * u.cm**2 * u.nm)]:
    """
    Calculates the specific flux of a blackbody, as seen by an external observer
    
    Parameters
    ----------
    wavelength: length
        The wavelength of light for which to calculate the specific flux
    temperature: temperature
        The temperature of the blackbody emitter
    distance_ratio: dimensionless
        The ratio radius / distance, where `radius` is the radius of the object and `distance` is the
        distance from the emitter to the observer
    
    Returns
    -------
    specific_flux: energy flux per wavelength interval
        The specific flux of the given wavelength
    """
    
    # Sets the units of the distribution and handles the jacobian factor
    scale = 1 * u.erg / (u.s * u.cm**2 * u.nm * u.sr)
    
    # Makes the BlackBody model
    bb = models.BlackBody(temperature=temperature, scale=scale)
    
    specific_intensity = bb(wavelength)
    
    # Calculates the external flux, which falls off with the inverse square law
    specific_flux = np.pi * u.sr * specific_intensity * distance_ratio**2
    specific_flux = specific_flux.to(u.erg / (u.s * u.cm**2 * u.nm))
    
    return specific_flux
