import numpy as np
from scipy import misc
import math


def load_image(image_file):
    row_image = misc.imread(image_file)
    return row_image


def save_image(new_image, file_name):
    misc.toimage(new_image).save(file_name)


def conv_mask_separately(sigma, filter_size):
    g1_x = np.zeros(filter_size)
    gx_res = np.zeros(filter_size)
    g1_y = np.zeros(filter_size)
    gy_res = np.zeros(filter_size)
    sum_g1_x = 0
    sum_g1_y = 0
    dist_to_origin = int(filter_size / 2)
    for i in range(0, filter_size):
        x_pos = i - dist_to_origin
        dist_power_2 = math.pow((x_pos/sigma), 2)
        dist_power_2 *= (-1/2)
        g1_x[i] += math.exp(dist_power_2)
        sum_g1_x += g1_x[i]

    for i in range(0, filter_size):
        y_pos = i - dist_to_origin
        dist_power_2 = math.pow((y_pos/sigma), 2)
        dist_power_2 *= (-1/2)
        g1_y[i] += math.exp(dist_power_2)
        sum_g1_y += g1_y[i]
    for i in range(0, filter_size):
        gx_res[i] = g1_x[i]/sum_g1_x
        gy_res[i] = g1_y[i]/sum_g1_y

    return gx_res, gy_res


def separate_mask_gaussian_conv(mask_x, mask_y, image, mid):
    result_image = image
    for im_x in range(mid, len(image) - mid):
        for im_y in range(mid, len(image) - mid):
            sum_image = 0
            for i in range(0, len(mask_x)):
                sum_mask = 0
                i_x = i - mid
                for j in range(0, len(mask_y)):
                        j_y = j - mid
                        sum_mask += (mask_y[j] * image[im_x - i_x][im_y - j_y])
                sum_image += (sum_mask * mask_x[i])
            result_image[im_x][im_y] = sum_image
    return result_image


if __name__ == "__main__":
    image = load_image("lena_n.gif")
    sigma = 3
    filter_size = 3
    mask_x, mask_y = conv_mask_separately(sigma, filter_size)
    image = separate_mask_gaussian_conv(mask_x, mask_y, image, int(filter_size/2))
    save_image(image, "lena_gaussian.png")
