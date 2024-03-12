import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import rawpy
import re

from constants import COUNTS_PER_ERG, SENSOR_AREA
from debayer import simple_debayer, rescale_debayer


def sum_to_flux(sum, exposure_time):
    img_energy = sum / COUNTS_PER_ERG
    
    flux = img_energy / (SENSOR_AREA * exposure_time)
    
    return flux


def parse_crop_string(crop_string):
    return np.array(re.findall(r'\d+', crop_string)).astype(int)


def crop_img(img, crop_string = None):
    if crop_string is None or crop_string == "":
        crop_location = img.shape[1], img.shape[0], 0, 0
    else:
        crop_location = parse_crop_string(crop_string)
        
    w, h, x, y = crop_location
    img = img[y:y+h, x:x+w]

    return img, crop_location


def get_source_sum(img, noise_location, source_location):
    # Get data from a part of the image with just noise    
    noise_crop, crop_location = crop_img(img, noise_location)
    noise_sum = np.sum(noise_crop, axis=(0,1))
    w, h, _, _ = crop_location
    
    noise_per_pixel = noise_sum / (w * h)
    
    laser_crop, crop_location = crop_img(img, source_location)
    w, h, _, _ = crop_location
    laser_sum = np.sum(laser_crop, axis=(0,1))
        
    corrected_sum = laser_sum - noise_per_pixel * (w * h)
    
    return corrected_sum.astype(int)
    

def main():
    path = 'Images/IMG_20240305_142245.dng'
    noise_location = "1011 x 1497 @ (1005, 15)"
    source_location = "155 x 136 @ (619, 881)"
    
    img = simple_debayer(path)
    imageio.imwrite("result.png", rescale_debayer(img))
    
    sum = get_source_sum(img, noise_location, source_location)
    
    print(sum / np.max(sum) * 255)
    

if __name__ == "__main__":
    main()
