import cv2 as cv
import os
import numpy as np


class GrayImage:
    def convert(self, count, file_path):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == file_path):
                img = cv.imread(imgPath)
                print(imgPath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite("new_image" + str(count) + ".jpg", gray)
        pass

    def Rotage(self, count, file_path, counter):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == file_path):
                img = cv.imread(file_path)
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv.getRotationMatrix2D((cX, cY), counter, 1.0)
        rotated = cv.warpAffine(img, M, (w, h))
        cv.imwrite("new_image" + str(count) + ".jpg", rotated)


class SeventScreen:
    def save_img(self, img, count, name, argument):
        cv.imwrite(
            "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\" + name + str(count) + ".png", img)
        pass

    def line_detection(self, file_path7, threshval, count):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == file_path7):
                src = cv.imread(imgPath, 0)
        img = cv.GaussianBlur(src, (3, 3), 0)
        n = 255
        retval, imB = cv.threshold(
            img, threshval, n, cv.THRESH_BINARY)
        ddepth = cv.CV_16S
        kernel_size = 3
        img_lap = cv.Laplacian(imB, ddepth, ksize=kernel_size)
        img_lap_abs = cv.convertScaleAbs(img_lap)
        img_lap_pos = img_lap > 0
        self.save_img(imB,         count, "LineDetection", count)
        self.save_img(img_lap,     count + 1,
                      "LineDetection", count)
        self.save_img(img_lap_abs, count + 2,
                      "LineDetection", count)

    def edge_detection(self, file_path7, count):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == file_path7):
                src = cv.imread(imgPath, 0)
        img = cv.medianBlur(src, 5)
        ddepth = cv.CV_16S
        grad_x = cv.Sobel(img, ddepth, dx=1, dy=0, ksize=3)
        grad_y = cv.Sobel(img, ddepth, dx=0, dy=1, ksize=3)
        try:
            grad = np.sqrt((grad_x**2+grad_y**2))
        except:
            pass
        thres = 60
        edge = grad > thres
        self.save_img(img,    count, "Edge_Detection", count)
        self.save_img(grad_x, count + 1, "Edge_Detection", count)
        self.save_img(grad_y, count + 2, "Edge_Detection", count)
        self.save_img(grad,   count + 3, "Edge_Detection", count)
        # self.save_img(edge,   5, "Edge_Detection", self.count)

    def thresholding(self, file_path7, threshval, count):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == file_path7):
                src = cv.imread(imgPath, 0)
        img = cv.medianBlur(src, 5)
        ret, th1 = cv.threshold(img, threshval, 255, cv.THRESH_BINARY)
        th2 = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        th3 = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        self.save_img(img,    count,
                      "Thresholding_Detection", count)
        self.save_img(th1,    count + 1,
                      "Thresholding_Detection", count)
        self.save_img(th2,    count + 2,
                      "Thresholding_Detection", count)
        self.save_img(th3,    count + 3,
                      "Thresholding_Detection", count)

    def optimum_thresholding(self, file_path7, threshval, count):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == file_path7):
                src = cv.imread(imgPath, 0)
        img = cv.medianBlur(src, 5)
        ret, th1 = cv.threshold(
            img, threshval, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        th2 = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2)
        self.save_img(
            img,    count, "Optimum_Thresholding_Detection", count)
        self.save_img(
            th1,    count + 1, "Optimum_Thresholding_Detection", count)
        self.save_img(
            th2,    count + 2, "Optimum_Thresholding_Detection", count)
        pass
