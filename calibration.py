import astropy.units as u
import numpy as np

from laser_energy import laser_image_energy
from sum_image import get_source_sum, simple_debayer

path = 'Images/IMG_20240305_142245.dng'
noise_location = "1011 x 1497 @ (1005, 15)"
source_location = "155 x 136 @ (619, 881)"

img = simple_debayer(path)

actual_sum = get_source_sum(img, noise_location, source_location)

laser_energy = laser_image_energy(1/1017 * u.s, 450 * 10**-5 * u.uW, 632.8 * u.nm)

print(np.mean(actual_sum / laser_energy) * u.count)