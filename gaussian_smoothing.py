import numpy as np
from scipy import misc
import math

def load_image(image_file):
    row_image = misc.imread(image_file)
    return row_image


def save_image(new_image, file_name):
    misc.toimage(new_image).save(file_name)

def gaussian_kernel(sigma):
    dist_to_origin = int(sigma / 2)
    g0 = np.zeros((sigma, sigma))
    g = np.zeros((sigma, sigma))
    sum_g0 = 0
    for i in range(0, sigma):
        for j in range(0, sigma):
            x_pos = i - dist_to_origin
            y_pos = j - dist_to_origin
            dist_power_2 = math.pow(x_pos,2) + math.pow(y_pos,2)
            dist_power_2 /= math.pow(sigma,2)
            dist_power_2 *= (-1/2)
            g0[i][j] += math.exp(dist_power_2)
            sum_g0 += g0[i][j]
    for i in range(0, sigma):
        for j in range(0, sigma):
            g[i][j] += g0[i][j] / sum_g0
    return g

def gaussian_conv(mask, image, mid):
    result_image = image
    for im_x in range(mid, len(image)-mid):
        for im_y in range(mid, len(image)-mid):
            sum_image = 0
            for i in range(0, len(mask)):
                for j in range(0, len(mask)):
                    i_x = i - mid
                    j_y = j - mid
                    sum_image += (mask[i][j] * image[im_x - i_x][im_y - j_y])
            result_image[im_x][im_y] = sum_image
    return result_image





if __name__ == "__main__":
    image = load_image("lena_noisy.png")
    sigma = 3
    mask = gaussian_kernel(sigma)
    image = gaussian_conv(mask, image, int(sigma/2))
    save_image(image, "new_lena1.png")
