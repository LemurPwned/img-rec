import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pprint as pp
import time


def gaussian_filter(k, sigma):
    """
    :param k: is the semi-size of kernel matrix
    :param sigma: spread over the values of the kernel
    :return: return ready Gaussian kernel
    """
    size = 2*k + 1
    B = [[0 for i in range(size)] for j in range(size)]
    factor = 1/(2*np.pi*(sigma**2))

    for i in range(1, size+1):
        for j in range(1, size+1):
            a = (i-(k+1))**2
            b = (j-(k+1))**2
            c = (2*(sigma**2))
            h = np.exp(-(a+b)/c)
            B[i-1][j-1] = factor*h
    return B


sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
gaussian = gaussian_filter(1, 1.4)
kernels = {
    'SobelX': (sobelX, 0),
    'SobelY': (sobelY, 0),
    'Gaussian': (gaussian, np.sum(gaussian))
}


def convolution(A, kernel):
    """
    :param A: matrix to be convoluted
    :param kernel: kernel of the convolution, as a tuple (kernel, kernel_sum)
    :return: return value of the convolution
    """
    result = 0
    kernel_sum = kernel[1]
    kernel = kernel[0]
    for i in range(0, len(kernel)):
        for j in range(0, len(kernel)):
            result += A[i][j]*kernel[i][j]
    if kernel_sum == 0:
        return result
    else:
        return result/float(kernel_sum)


def remap_kernel(img):
    max_val = np.max(img)
    for row in img:
        row[:] = np.ceil(np.interp(row, [0, max_val], [0, 255]))
    return img


def iterate_img(img, h, w, kernel):
    """
    :param img: image to be iterated over
    :param h: height of the image
    :param w: width of the image
    :param kernel: kernel to create a mask
    :return: new image with kernel mask put over
    """

    new_img = [[0 for i in range(0, w)] for j in range(0, h)]
    for i in range(1, h-1):
        for j in range(1, w-1):
            a = np.array(img[i-1:i+2, j-1:j+2])
            new_img[i][j] = convolution(a, kernel)
    return new_img


def sobel_mask(Gx, Gy):
    if Gx != 0:
        return np.sqrt(Gx ** 2 + Gy ** 2), np.arctan(Gy / Gx)
    else:
        return np.sqrt(Gx**2 + Gy**2), np.pi/2


def compose_sobel(imgA, imgB):
    """
    Puts together the vertical and horizontal Sobel operators
    :param imgA: Horizontal
    :param imgB: Vertical
    :return: Edge detection trigger
    """
    result_img = [[0 for i in range(0, len(imgA[0]))] for j in range(0, len(imgA))]
    print(len(imgA), len(imgB), len(imgA[0]), len(imgB[0]), len(result_img), len(result_img[0]))
    for i in range(0, len(imgA)):
        for j in range(0, len(imgA[0])):
            result_img[i][j] = sobel_mask(imgA[i][j], imgB[i][j])[0]
    return result_img


def rescale(A):
    corner_val = A[0][0]    
    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] /= float(corner_val)
    return A


def normalize(A):
    s = 0
    for i in range(len(A)):
        for j in range(len(A)):
            s += A[i][j]
    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] /= s
    return A


def print_matrix(A):
    for row in A:
        print(row)

img = cv.imread('D:\\Dokumenty\\image-recognition\\track\\finish1.JPG', 0)
img2 = cv.Canny(img, 520, 160)

cv.namedWindow("name", cv.WINDOW_NORMAL)
cv.imshow("name", img2)
cv.waitKey(0)
cv.destroyAllWindows()

h, w = img.shape
img = np.array(img)
# apply Gaussian blur
start1 = time.time()
new_img = iterate_img(img, h, w, kernels['Gaussian'])
end1 = time.time()
# Remap the kernel to standarize
new_img = remap_kernel(new_img)
# apply SobelX
new_imgX = iterate_img(img, h, w, kernels['SobelX'])
# apply SobelY
new_imgY = iterate_img(img, h, w, kernels['SobelY'])
# put together
final_img = compose_sobel(new_imgX, new_imgY)
end2 = time.time()

print("Single operation {}; multiples operation : {}".format(end1 - start1, end2 - start1))
# idk why not in greyscale, but assume it is ok
# imgplot = plt.imshow(final_img)


plt.subplot(1, 2, 1)
plt.imshow(new_imgX)
plt.title("Sobel X")
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(new_imgY)
plt.title("Sobel Y")
plt.xticks([]), plt.yticks([])
plt.show()

# some Tkinker shit pops out sometimes, but generally everthing should be o.k.
