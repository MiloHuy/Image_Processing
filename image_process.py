import math
import cv2 as cv
import numpy as np

# from scipy.ndimage import convolve

# # cv.imshow("OGR",img)
# # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # F = np.fft.fft2(img)
# # F = np.fft.fftshift(F)
# # M, N = img.shape[:2]
# # D0 = 30
# # u = np.arange(0, M) - M / 2
# # v = np.arange(0, N) - N / 2
# # [V, U] = np.meshgrid(v, u)
# # D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
# # H = np.array(D <= D0, 'float')
# # G = H * F
# # G = np.fft.ifftshift(G)
# # imgOut = np.real(np.fft.ifft2(G))
# # cv.imshow("Image_Out",imgOut)


# # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # cv.imshow("OGR",img)
# # F = np.fft.fft2(img)
# # F = np.fft.fftshift(F)
# # M, N = img.shape[:2]
# # D0 = 75
# # u = np.arange(0, M) - M / 2
# # v = np.arange(0, N) - N / 2
# # [V, U] = np.meshgrid(v, u)
# # D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
# # H = 1/np.power(1 + (D0/D),(2*2))
# # G = H * F
# # G = np.fft.ifftshift(G)
# # imgOut = np.real(np.fft.ifft2(G))
# # cv.imshow("Image_Out",imgOut)


# # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # cv.imshow("OGR",img)
# # F = np.fft.fft2(img)
# # F = np.fft.fftshift(F)
# # M, N = img.shape[:2]
# # D0 = 75
# # u = np.arange(0, M) - M / 2
# # v = np.arange(0, N) - N / 2
# # [V, U] = np.meshgrid(v, u)
# # D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
# # H = np.exp((-D**2)/((2*D0)**2))
# # G = H * F
# # G = np.fft.ifftshift(G)
# # imgOut = np.real(np.fft.ifft2(G))
# # cv.imshow("Image_Out",imgOut)


# # img = cv.imread(".\images\\1.jpg",0)
# # cv.imshow("Ori",img)
# # threshval = 200; n = 255
# # retval,imgB = cv.threshold(img,threshval, n,cv.THRESH_BINARY)
# # kenrel = np.ones((6,6),np.uint8)
# # img_ero1 = cv.erode(imgB,kenrel,iterations=1)
# # cv.imshow("New",img_ero1)
# # cv.waitKey(0)


# img = cv.imread("new_image.jpg")


# kernel = np.ones((15,15),dtype=np.uint8)


# closing = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)


# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))


# dilate = cv.dilate(closing,kernel,iterations = 1)

# output = closing - dilate

# cv.imshow("Image",output)


# cv.waitKey(0)


# import the turtle modules


# src = cv.imread('./images/imageline.png', 0)
# img = cv.GaussianBlur(src, (3, 3), 0)
# threshval = 50
# n = 255
# retval, imB = cv.threshold(img, threshval, n, cv.THRESH_BINARY)
# ddepth = cv.CV_16S
# kernel_size = 3
# img_lap = cv.Laplacian(imB, ddepth, ksize=kernel_size)
# img_lap_abs = cv.convertScaleAbs(img_lap)
# img_lap_pos = (img_lap > 0) + 1


# cv.imshow("Image1", imB)
# cv.imshow("Image2", img_lap)
# cv.imshow("Image3", img_lap_abs)
# cv.imshow("Image4", img_lap_pos)


# img = cv.imread('./images/imagebuilding.jpg', 0)
# img = cv.medianBlur(img, 5)
# ddepth = cv.CV_16S
# grad_x = cv.Sobel(img, ddepth, dx=1, dy=0, ksize=3)
# grad_y = cv.Sobel(img, ddepth, dx=0, dy=1, ksize=3)
# try:
#     grad = np.sqrt((grad_x**2+grad_y**2))
# except:
#     pass
# thres = 60
# edge = grad > thres
# titles = ['Image', 'grad_x', 'grad_y', 'grad', 'Egde']
# images = [img, grad_x, grad_y, grad, edge]

# for image in images:
#     cv.imshow("image", image)
#     cv.waitKey(0)


# img = cv.imread('./images/soduku.jpg', 0)
# img = cv.medianBlur(img, 5)
# ret, th1 = cv.threshold(img, 132, 255, cv.THRESH_BINARY)
# th2 = cv.adaptiveThreshold(img, 255,

#                            cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

# th3 = cv.adaptiveThreshold(img, 255,

#                            cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# titles = ['Original Image', 'Thresholding (v = 132)',
#           'Mean Thresholding', 'Gaussian Thresholding']

# images = [img, th1, th2, th3]
# for image in images:
#     cv.imshow("image", image)
#     cv.waitKey(0)


# A = [[574.0, 446.99999999999994], [355.0, 561.0], [815.0, 582.0]]
# B = []
# d = 0
# for i in range(len(A)):
#     for j in range(2):
#         t = A[i]
#         B.append(t[j])
#     print(B[d])
#     print(B[d + 1])
#     d = d + 2


# img_input = cv.imread("new_image.png", 0)
# # cv.imshow("Image", img_input)


# h, w = img_input.shape[:2]
# mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
# imfloodfill = img_input.copy()
# cv.floodFill(imfloodfill, mask, (0, 0), 255)
# imfloodfill_not = cv.bitwise_not(imfloodfill)
# img_output = img_input | imfloodfill_not
# (th, img_output) = cv.threshold(img_output, 127, 255, cv.THRESH_BINARY_INV)

# cv.imshow("Image_output", img_output)

# i = 1
# print("new_image" + str(i) + ".jpg")


# kernel = np.ones((15, 15, 3), dtype=np.uint8)
# print(kernel)

string = "Car detection using image processing and pattern recognition techniques."
print(string.upper())
