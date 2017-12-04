import cv2 as cv

def apply_canny_filter(img):
    canny_img = cv.Canny(img, 100, 200)
    return canny_img


def apply_laplace_filter(img):
    laplace_img = cv.Laplacian(img, cv.CV_64F)
    return laplace_img


def test_img():
    filename = "C:\\Users\\Jakub\\Desktop\\fog1.jpg"
    img = cv.imread(filename)
    img = apply_laplace_filter(img)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

test_img()