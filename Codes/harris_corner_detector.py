import numpy as np
from scipy import misc
import math

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


def create_gradient_matrix(sobel_im_x, sobel_im_y, size):
    res_x = np.zeros((size, size))
    res_y = np.zeros((size, size))
    res_xy = np.zeros((size, size))

    for i in range(0, size):
        for j in range(0, size):
            res_x[i][j] += math.pow(sobel_im_x[i][j], 2)
            res_y[i][j] += math.pow(sobel_im_y[i][j], 2)
            res_xy[i][j] += sobel_im_x[i][j] * sobel_im_y[i][j]
            gaussian_val = math.exp((-1/2) * (math.pow(i, 2) + math.pow(j, 2)))



if __name__ == "__main__":
    image = load_image("lena_gray_scale.png")
    sobel_x, sobel_y = apply_sobel_filter(image)


