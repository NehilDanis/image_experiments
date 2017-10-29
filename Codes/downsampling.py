import numpy as np
import math
import gaussian_derivative


def downsample_image(image, downsampling_factor):
    mid = downsampling_factor - 1
    size = len(image) - mid
    #new_image = np.zeros((len(image), len(image)))
    new_image = np.zeros((size, size))
    row = 0
    for i in range(0, len(image) - mid):
        column = 0
        for j in range(0, len(image) - mid):
            sum_sample = (calc_sum(i, j, image, int(downsampling_factor/2)))
            new_image[row][column] += sum_sample
            column += 1
        row += 1

    return new_image


def calc_sum(i, j, image, downsampling_filter_size):
    sum_square = 0
    num_elements = 0
    for row in range(0, downsampling_filter_size):
       for column in range(0, downsampling_filter_size):
            sum_square += image[i+row][j+column]
            num_elements += 1
    return sum_square/num_elements

if __name__ == "__main__":
    image = gaussian_derivative.load_image("lena_gray_scale.png")
    image = downsample_image(image, 8)
    gaussian_derivative.save_image(image, "lena_downsample.png")
