import numpy as np
from scipy import misc
import math

def load_image(image_file):
    row_image = misc.imread(image_file)
    return row_image


def save_image(new_image, file_name):
    misc.toimage(new_image).save(file_name)


def change_brightness(image_to_change, bias):
    brighter_image = np.empty_like(image_to_change)
    for i in range(0, len(image_to_change)):
        for j in range(0, len(image_to_change[i])):
            #for color in range(0, len(image_to_change[i][j])): #first check whether its rgb image or not.
            brightened_val = image_to_change[i][j] + bias
            brighter_image[i][j] = np.uint8(brightened_val)
    return brighter_image

def conv_filter():
    s = (3, 3)
    mask = np.zeros(s)
    #mask[0] = [0, 1, 0]
    #mask[1]  = [1, 4, 1]
    #mask[2] = [0, 1, 0]
    mask[0] = [-1, 0, 1]
    mask[1]  = [-1, 0, 1]
    mask[2] = [-1, 0, 1]
    return mask

def calc_derivative(image):
    # I(i+1, j) - I(i-1, j) derivative of image with respect to x.
    # I(i, j+1) - I(i, j-1) derivative of image with respect to y.
    new_image = np.zeros(len(image))
    new_image = image
    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if i+1 < len(image) and i-1 >= 0:
                new_image[i][j] = np.uint8(image[i+1][j] - image[i-1][j])
    return new_image

def control_pixel(i, j, size):
    if i == 0 or j == 0:
        return False
    elif i == size or j == size:
        return False
    return True

def calc_mult(image, filter, i, j):
    start_i = i - 1
    sum = 0
    filter_sum = 0
    for index_i in range(0, len(filter)):
        start_j = j -1
        for index_j in range(0, len(filter)):
            sum += (image[start_i][start_j]*filter[index_i][index_j])
            start_j += 1
            filter_sum += filter[index_i][index_j]
        start_i += 1
    if filter_sum == 0:
        filter_sum += 1
    return np.uint8(sum / filter_sum)

def convolution(image, filter):
    size = len(filter)
    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if control_pixel(i, j, len(image) - 1) is True:
                image[i][j] = calc_mult(image, filter, i, j)
    return image

def homogenous_diff(iteration_num, image):
    if iteration_num == 0:
        return image
    image_temp = np.zeros(len(image))
    image_temp = image

    for i in range(1,len(image)-1):
        for j in range(1,len(image)-1):
            sum = image_temp[i-1][j] + image_temp[i+1][j] \
                  + image_temp[i][j-1] + image_temp[i][j+1]
            image[i][j] = (sum/4) + (3 * image_temp[i][j]/4)
    return homogenous_diff(iteration_num-1, image)

def derivative_mask(mask_x, mask_y, mid):
    d_x = np.zeros(len(mask_x))
    d_y = np.zeros(len(mask_y))
    result = 0
    for i in range(0, len(mask_x)):
        x_pos = i - mid
        y_pos = i - mid
        d_x[i] += ((-1) * x_pos * math.exp((-1/2) * math.pow((x_pos)/len(mask_x), 2)))
        d_y[i] += ((-1) * y_pos * math.exp((-1/2) * math.pow((y_pos)/len(mask_y), 2)))
        result += x_pos * d_x[i]

    for i in range (0, len(mask_x)):
        d_x[i] = result * d_x[i]
        d_y[i] = result * d_y[i]
    return d_x, d_y

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

def conv_mask_seperately(sigma):
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

def seperate_mask_gaussian_conv(mask_x, mask_y, image, mid):
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

def derivative_gaussian_conv(mask, mid):
    for i in range(0, len(mask)):
        for j in range(0, len(mask)):
            i_x = i - mid
            j_y = j - mid
            mask[i_x][j_y] = (i_x * math.exp((-1/2)* (math.pow(i_x, 2) + math.pow(j_y, 2)) / math.pow(len(mask), 2)))
    return mask

def gaussian_derivative_conv(mask_x, mask_y, d_x, d_y, image, mid):
    result_image = image
    for im_x in range(mid, len(image)-mid):
        for im_y in range(mid, len(image)-mid):
            sum_image = 0
            for i in range(0, len(mask_x)):
                sum_mask = 0
                i_y = i - mid
                for j in range(0, len(mask_y)):
                    j_x = j - mid
                    sum_mask += (image[im_x - i_y][im_y - j_x] * d_x[j_x])
                sum_image += (sum_mask * mask_y[i_y])
            result_image[im_x][im_y] = sum_image
    return result_image

'''if __name__ == "__main__":
    image = load_image("new_lena1.png")
    save_image(image, "new_lena1.png")'''




