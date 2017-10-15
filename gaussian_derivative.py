import numpy as np
from scipy import misc
import math

def load_image(image_file):
    row_image = misc.imread(image_file)
    return row_image


def save_image(new_image, file_name):
    misc.toimage(new_image).save(file_name)


def gaussian_derivative_filter(sigma):
    mid = int(sigma/2)
    mask = np.zeros(sigma)
    result_mask = np.zeros(sigma)
    c = 0
    for i in range(0, sigma):
        i_pos = i - mid
        mask[i] += ((-1) * i_pos * math.exp((-1/2) * math.pow((i_pos / sigma), 2)))
        c += (i_pos * mask[i])

    for i in range(0, sigma):
        result_mask[i] += (mask[i]/c)
    return result_mask, np.transpose(result_mask)


def gaussian_conv_x_axes(mask, image, mid):
    result_image = image
    for i in range(0, len(image)):
        for j in range(mid, len(image) - mid):
            sum_kernel = 0
            for c in range(0, len(mask)):
                c_pos = c - mid
                sum_kernel += (image[i][j - c_pos] * mask[c])
            result_image[i][j] = sum_kernel
    return result_image

def gaussian_conv_y_axes(mask, image, mid):
    result_image = image
    for i in range(mid, len(image)-mid):
        for j in range(0, len(image)):
            sum_kernel = 0
            for c in range(0, len(mask)):
                c_pos = c - mid
                sum_kernel += (image[i - c_pos][j] * mask[c])
            result_image[i][j] = sum_kernel
    return result_image

def gaussian_conv(mask, image, mid):
    result_image = image
    for i in range(mid, len(image)-mid):
        for j in range(mid, len(image)-mid):
            sum_kernel = 0
            for c in range(0, len(mask)):
                c_pos = c - mid
                sum_kernel += (image[i][j - c_pos] * mask[c])
            for c in range(0, len(mask)):
                c_pos = c - mid
                sum_kernel += (image[i - c_pos][j] * mask[c])
            result_image[i][j] = sum_kernel
    return result_image

if __name__ == "__main__":
    image = load_image("lena_noisy.png")
    mask_x, mask_y = gaussian_derivative_filter(3)
    #image = gaussian_conv_x_axes(mask_x, image, int(3/2))
    #image = gaussian_conv_y_axes(mask_y, image, int(3/2))
    image = gaussian_conv(mask_x, image, int(3/2))
    save_image(image, "lena_xy.png")
