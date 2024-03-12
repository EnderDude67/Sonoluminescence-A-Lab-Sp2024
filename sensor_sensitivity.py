import csv
import numpy as np
import astropy.units as u
import scipy.interpolate as interp

from astropy.units import Quantity


def read_csv(filename):
    """
    Returns the contents of an unlabled CSV file as a numpy array
    Data is returned as floats
    
    Parameters
    ----------
    filename: string
        The relative path of the CSV file
    
    Returns
    -------
    rows: NDArray[np.float32]
        The retrieved float data in row major order
    """
    
    rows = []
    with open(filename, mode ='r')as file:
        csvFile = csv.reader(file)
        
        for lines in csvFile:
                rows.append(lines)

    rows = np.array(rows).astype(np.float32)
    return rows


# Reads the sensitivity values from the csv file
# Global variable so the file needs to only be read once
sensitivities = read_csv("Pixel4_Sensitivity.csv")


def sensitivity_func(
        wavelength: Quantity["length"]
) -> Quantity["dimensionless"]:
    """
    Calculates the relative response of the sensor to the given wavelength of light
    Interpolates the CSV data
    
    Parameters
    ----------
    wavelength: length
        The wavelength of light for which to estimate the sensitivity
    
    Returns
    -------
    sensitivity: [R, G, B] dimensionless
        An array of numbers between 0 and 1 representing the sensor's sensitivity to the given
        wavelength on each color channel
    """
    
    # Make the wavelength values that pair with the CSV data
    wavelengths = np.linspace(400, 700, 31) * u.nm
    
    # Create the interpolator
    interp_func = interp.PchipInterpolator(wavelengths, sensitivities)
    
    sensitivity = interp_func(wavelength.to(u.nm))
    
    return sensitivity
     


def main():
    laser_activation = sensitivity_func(632.8 * u.nm)
    #  print(laser_activation)
    print(laser_activation / np.max(laser_activation))


if __name__ == "__main__":
     main()
