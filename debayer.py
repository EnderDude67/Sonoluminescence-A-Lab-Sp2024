import numpy as np
import rawpy
import imageio


def simple_debayer(img_path):
    with rawpy.imread(img_path) as raw:
        raw_img = np.copy(raw.raw_image)
       
    # Breaks the image into 2x2 chunks, each with one red, two green, and one blue sensor
    bayer_chunks = raw_img.reshape(raw_img.shape[0] // 2, 2, raw_img.shape[1] // 2, 2)
    bayer_chunks = np.swapaxes(bayer_chunks, 1, 2)
    
    # Constructs the empty array to fill with values
    result_img = np.zeros((raw_img.shape[0] // 2, raw_img.shape[1] // 2, 3), dtype=np.float64)
    
    # Directly copy the red and blue values from the bayer chunks
    result_img[:, :, 0] = bayer_chunks[:, :, 0, 0]
    result_img[:, :, 2] = bayer_chunks[:, :, 1, 1]
    
    # Average the values of the two green pixels - matches the methodology of the paper
    result_img[:, :, 1] = (bayer_chunks[:, :, 0, 1] + bayer_chunks[:, :, 1, 0]) / 2
        
    return result_img


def rescale_debayer(img):    
    img //= 4
    img = img.astype(np.uint8)

    return img
    


def cool_bayer(img_path):
    """
    Not useful but makes cool images
    """
    
    with rawpy.imread(img_path) as raw:
        raw_img = np.copy(raw.raw_image)
        colors = np.copy(raw.raw_colors)
    
    red_channel = raw_img * (colors == 0)

    green_channel = raw_img * ((colors == 1) | (colors == 3))

    blue_channel = raw_img * (colors == 2)

    # Create the bayered image
    result_img = np.zeros((raw_img.shape[0], raw_img.shape[1], 3))

    result_img[:, :, 0] = red_channel
    result_img[:, :, 1] = green_channel
    result_img[:, :, 2] = blue_channel
    
    return result_img


def main():
    img_path = "Images/Lightbulb/IMG_20240308_232001.dng"
    
    # rescale_debayer(img_path)


if __name__ == "__main__":
    main()