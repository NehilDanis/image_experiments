import numpy as np
import math


def conv_mask_separately(sigma):
    g1_x = np.zeros(sigma)
    gx_res = np.zeros(sigma)
    g1_y = np.zeros(sigma)
    gy_res = np.zeros(sigma)
    sum_g1_x = 0
    sum_g1_y = 0
    dist_to_origin = int(sigma / 2)
    for i in range(0, sigma):
        x_pos = i - dist_to_origin
        dist_power_2 = math.pow((x_pos/sigma), 2)
        dist_power_2 *= (-1/2)
        g1_x[i] += math.exp(dist_power_2)
        sum_g1_x += g1_x[i]

    for i in range(0, sigma):
        y_pos = i - dist_to_origin
        dist_power_2 = math.pow((y_pos/sigma), 2)
        dist_power_2 *= (-1/2)
        g1_y[i] += math.exp(dist_power_2)
        sum_g1_y += g1_y[i]
    for i in range(0, sigma):
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
