import numpy as np
from scipy import misc
import math
import gaussian_smoothing

def load_image(image_file):
    row_image = misc.imread(image_file)
    return row_image


def save_image(new_image, file_name):
    misc.toimage(new_image).save(file_name)


def derivative_with_respect_to_x(image):
    x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.int32)
    result_image = image
    result_image = result_image.astype('int32')
    for i in range(1, len(image) - 1):
        for j in range(1, len(image) - 1):
            sum_result = 0
            for m_i in range(0, len(x)):
                for m_j in range(0, len(x)):
                    im_i = m_i - 1
                    im_j = m_j -1
                    sum_result += x[m_i][m_j] * image[i - im_i][j - im_j]
            result_image[i][j] = sum_result
    save_image(result_image ,"sobel_x.png")
    return result_image


def derivative_with_respect_to_y(image):
    y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.int32)
    result_image = image
    result_image = result_image.astype('int32')
    for i in range(1, len(image) - 1):
        for j in range(1, len(image) - 1):
            sum_result = 0
            for m_i in range(0, len(y)):
                for m_j in range(0, len(y)):
                    im_i = m_i - 1
                    im_j = m_j -1
                    sum_result += y[m_i][m_j] * image[i - im_i][j - im_j]
            result_image[i][j] = sum_result
    save_image(result_image ,"sobel_y.png")
    return result_image


def apply_sobel_filter(image):
    return derivative_with_respect_to_x(image), derivative_with_respect_to_y(image)


def r_treshold(matrix):
    det_matrix = np.linalg.det(matrix)
    trace_matrix = np.trace(matrix)
    k  = 0.05 # scales between 0.04 and 0.06
    return det_matrix - (k * math.pow(trace_matrix, 2))


def create_gradient_matrix(sobel_im_x, sobel_im_y, lena_smooth, image, size):
    res_x = np.zeros((size, size))
    res_y = np.zeros((size, size))
    res_xy = np.zeros((size, size))
    mid = int(size / 2)
    for i in range(0, size):
        for j in range(0, size):
            res_x[i][j] += math.pow(sobel_im_x[i][j], 2)
            res_y[i][j] += math.pow(sobel_im_y[i][j], 2)
            res_xy[i][j] += sobel_im_x[i][j] * sobel_im_y[i][j]
            '''s_x = lena_smooth[i][j] * res_x[i][j]
            s_y = lena_smooth[i][j] * res_y[i][j]
            s_xy = lena_smooth[i][j] * res_xy[i][j]
            matrix = np.array([[s_x, s_xy], [s_xy, s_y]], np.int32)
            r_trashold_val = r_treshold(matrix)
            if r_trashold_val > 10000:
                print "nehil"
                image[i][j] = 255'''
    mask_x, mask_y = gaussian_smoothing.conv_mask_separately(1, 3)
    gauss_res_x = gaussian_smoothing.separate_mask_gaussian_conv(mask_x, mask_y, res_x, int(3/2))
    gauss_res_y = gaussian_smoothing.separate_mask_gaussian_conv(mask_x, mask_y, res_y, int(3/2))
    gauss_res_xy = gaussian_smoothing.separate_mask_gaussian_conv(mask_x, mask_y, res_xy, int(3/2))

    for i in range(0, size):
        for j in range(0, size):
            s_x = gauss_res_x[i][j]
            s_y = gauss_res_y[i][j]
            s_xy = gauss_res_xy[i][j]
            matrix = np.array([[s_x, s_xy], [s_xy, s_y]], np.int32)
            r_trashold_val = r_treshold(matrix)
            if r_trashold_val < -10000:
                image[i][j] = [0, 0, 255]
    return image


if __name__ == "__main__":
    image = load_image("lena_gray_scale.png")
    image_rgb = load_image("lena.png")
    sobel_x, sobel_y = apply_sobel_filter(image)
    lena_smooth = load_image("lena_smooth.png")
    image  = create_gradient_matrix(sobel_x, sobel_y, lena_smooth, image_rgb, len(image))
    save_image(image, "harris.png")

