import astropy.units as u
import numpy as np

import sum_image as sum
import minimize
from blackbody_energy import bb_color_flux

path = "Images/Lightbulb/IMG_20240308_225556.dng"
noise_location = "1931 x 3024 @ (2101, 0)"
source_location = "277 x 259 @ (1770, 1369)"

source_count = sum.get_source_sum(path, noise_location, source_location)

img_flux = sum.sum_to_flux(source_count, 1/73470 * u.s)
img_flux = img_flux.to(u.erg / (u.s * u.cm**2))

print(img_flux)

bb_params = minimize.get_bb_params(img_flux)

print(bb_params)

color_flux = bb_color_flux(*bb_params)

color = color_flux / np.max(color_flux) * 255

print(color)