from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color, Line, Ellipse, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, StringProperty
import cv2 as cv
import random
import numpy as np
import os
import module
# import matplotlib.pyplot as plt

LabelBase.register(name="font2", fn_regular="font2.ttf")


class WindowManager(ScreenManager):
    pass


class FirstScreen(Screen):
    def open_popup(self):
        self.the_popup = WindowExit()
        content_exit = NotificationExitApp()
        self.the_popup.add_widget(content_exit)
        self.the_popup.open()

    def CloseApp(self):
        content_exit = NotificationExitApp()
        content_exit.exit_app()
        pass
    pass


class WindowExit(Popup):
    pass


class NotificationExitApp(FloatLayout):

    def exit_app(self):
        App.get_running_app().stop()
        Window.close()
    pass

    def cancel_popup(self):
        self.the_popup_exit = WindowExit()
        self.the_popup_exit.dismiss()
    pass


class Content1(FloatLayout):
    load = ObjectProperty()

    def dismiss(self):
        self.absadasdasd = WindowPopup()
        self.absadasdasd.dismiss()
    pass


class Content4(FloatLayout):
    load = ObjectProperty()
    cancel = ObjectProperty()
    pass


class Content6(FloatLayout):
    load = ObjectProperty()

    def cancel_popup(self):
        self.the_popup_exit = WindowPopup()
        self.the_popup_exit.dismiss()
        print("cancel")
    pass


class WindowPopup(Popup):
    pass


class SecondScreen(Screen):
    file_path = StringProperty("No file chosen")
    the_popup = ObjectProperty(None)
    count = 0
    grayimage = module.GrayImage()
    counter = 0

    def Reset(self):
        self.ids.my_image.source = 'backgroundsecondscreen_1.png'
        self.ids.new_image.source = "backgroundsecondscreen_2.png"

    def open_popup(self):
        self.the_popup = WindowPopup()
        content = Content1(load=self.load)
        self.the_popup.add_widget(content)
        self.the_popup.open()

    def cancel_popup(self):
        self.the_popup = WindowPopup()
        self.the_popup.dismiss()
        pass

    def load(self, selection):
        self.file_path = str(selection[0])
        self.the_popup.dismiss()
        # print(self.file_path)
        try:
            self.ids.my_image.source = self.file_path
        except:
            pass

    def ClearImage(self):
        os.remove("new_image.jpg")
        self.ids.new_image.source = ""

    def RotateImage(self):
        self.grayimage.Rotage(self.count, self.file_path, self.counter)
        self.ids.new_image.source = "new_image" + str(self.count) + ".jpg"
        self.count += 1
        self.counter += 10
        pass

    def ConverGrayImage(self):
        self.ids.new_image.source = ""
        self.grayimage.convert(self.count, self.file_path)
        self.ids.new_image.source = "new_image" + str(self.count) + ".jpg"
        self.count += 1
        pass


class ThirdScreen(Screen):
    def Reset(self):
        self.ids.my_image3.source = "image3.png"
        self.ids.new_image3.source = "image3.png"

    def selected(self, filename):
        try:
            self.ids.my_image3.source = filename[0]
            print(self.ids.my_image3.source)
            self.ids.new_image3.remove_from_cache()
        except:
            pass
        # self.ids.my_image3.source = filename[0]

    def ImageNegative(self, filename):
        self.ids.new_image3.source = ""
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        img2 = 255 - img
        cv.imwrite("new_image.jpg", img2)
        self.ids.new_image3.source = "new_image.jpg"
        pass

    def LogTransformation(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        c = 255/np.log(1 + np.log(img + 1))
        log_image = c * (np.log(img + 1))
        log_image = np.array(log_image, dtype=np.uint8)
        cv.imwrite("new_image.jpg", log_image)
        self.ids.new_image3.source = "new_image.jpg"
        pass

    def PowerLaw(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
                gama = np.array(255*(img/255)**0.5, dtype='uint8')
                cv.imwrite("new_image.jpg", gama)
                self.ids.new_image3.source = "new_image.jpg"
        pass

    def MinFilter(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        height, width = img.shape[:2]
        img1, img2, img3 = cv.split(img)
        for i in range(0, height):
            for j in range(0, width):
                img1[i, j] = np.min(img1[i:i + 2, j:j + 2])
                img2[i, j] = np.min(img2[i:i + 2, j:j + 2])
                img3[i, j] = np.min(img3[i:i + 2, j:j + 2])
        img = cv.merge([img1, img2, img3])
        cv.imwrite("new_image.jpg", img)
        self.ids.new_image3.source = "new_image.jpg"
        pass

    def MaxFilter(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        height, width = img.shape[:2]
        img1, img2, img3 = cv.split(img)
        for i in range(0, height):
            for j in range(0, width):
                img1[i, j] = np.max(img1[i:i + 2, j:j + 2])
                img2[i, j] = np.max(img2[i:i + 2, j:j + 2])
                img3[i, j] = np.max(img3[i:i + 2, j:j + 2])
        img = cv.merge([img1, img2, img3])
        cv.imwrite("new_image.jpg", img)
        self.ids.new_image3.source = "new_image.jpg"
        pass

    def Max_Min_Filter(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        height, width = img.shape[:2]
        img1, img2, img3 = cv.split(img)
        for i in range(0, height):
            for j in range(0, width):
                img1[i, j] = np.max(img1[i:i + 2, j:j + 2]) - \
                    np.min(img1[i:i + 2, j:j + 2])
                img2[i, j] = np.max(img2[i:i + 2, j:j + 2]) - \
                    np.min(img2[i:i + 2, j:j + 2])
                img3[i, j] = np.max(img3[i:i + 2, j:j + 2]) - \
                    np.min(img3[i:i + 2, j:j + 2])
        img = cv.merge([img1, img2, img3])
        cv.imwrite("new_image.jpg", img)
        self.ids.new_image3.source = "new_image.jpg"
        pass

    def MidPoint_Filter(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        height, width = img.shape[:2]
        img1, img2, img3 = cv.split(img)
        for i in range(0, height):
            for j in range(0, width):
                img1[i, j] = (np.max(img1[i:i + 2, j:j + 2]) +
                              np.min(img1[i:i + 2, j:j + 2]))/2
                img2[i, j] = (np.max(img2[i:i + 2, j:j + 2]) +
                              np.min(img2[i:i + 2, j:j + 2]))/2
                img3[i, j] = (np.max(img3[i:i + 2, j:j + 2]) +
                              np.min(img3[i:i + 2, j:j + 2]))/2
        img = cv.merge([img1, img2, img3])
        cv.imwrite("new_image.jpg", img)
        self.ids.new_image3.source = "new_image.jpg"
        pass

    def Average_Filter(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        img1, img2, img3 = cv.split(img)
        height, width = img.shape[:2]
        for i in range(0, height - 3):
            for j in range(0, width - 3):
                avg1 = np.average(img1[i:i + 3, j:j + 3])
                img1[i, j] = avg1
                avg2 = np.average(img2[i:i + 3, j:j + 3])
                img2[i, j] = avg2
                avg3 = np.average(img3[i:i + 3, j:j + 3])
                img3[i, j] = avg3
        new_image = cv.merge([img1, img2, img3])
        new_image = new_image.astype(np.uint8)
        cv.imwrite("new_image.jpg", new_image)
        self.ids.new_image3.source = "new_image.jpg"
        pass

    def Median_Filter(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        self.ids.new_image3.source = ""
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath)
        img1, img2, img3 = cv.split(img)
        height, width = img.shape[:2]
        new_image1 = np.ones([height, width])
        new_image2 = np.ones([height, width])
        new_image3 = np.ones([height, width])
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                temp1 = [img1[i-1, j-1],   img1[i-1, j],   img1[i-1,   j + 1],
                         img1[i, j-1],     img1[i, j],     img1[i,     j + 1],
                         img1[i + 1, j-1],  img1[i + 1, j], img1[i + 1, j + 1]]
                temp1 = sorted(temp1)
                new_image1[i, j] = temp1[4]
                temp2 = [img2[i-1, j-1],   img2[i-1, j],   img2[i-1,   j + 1],
                         img2[i, j-1],    img2[i, j],     img2[i,     j + 1],
                         img2[i + 1, j-1], img2[i + 1, j], img2[i + 1, j + 1]]
                temp2 = sorted(temp2)
                new_image2[i, j] = temp2[4]
                temp3 = [img3[i-1, j-1],   img3[i-1, j],   img3[i-1,   j + 1],
                         img3[i, j-1],    img3[i, j],     img3[i,     j + 1],
                         img3[i + 1, j-1], img3[i + 1, j], img3[i + 1, j + 1]]
                temp3 = sorted(temp3)
                new_image3[i, j] = temp3[4]
        new_image1 = new_image1.astype(np.uint8)
        new_image2 = new_image2.astype(np.uint8)
        new_image3 = new_image3.astype(np.uint8)
        new_image = cv.merge([new_image1, new_image2, new_image3])
        cv.imwrite("new_image.jpg", new_image)
        self.ids.new_image3.source = "new_image.jpg"
        pass


class FourthScreen(Screen):
    count = 0

    def Reset(self):
        self.ids.my_image4.source = "backgroundscreen_1_4.png"
        self.ids.new_image4.source = "backgroundscreen_2_4.png"

    def Reset_new_image(self):
        self.ids.new_image4.source = ""

    def selected(self, filename):
        try:
            self.ids.my_image4.source = filename[0]
            # print(self.ids.my_image3.source)
            self.ids.new_image4.remove_from_cache()
        except:
            pass

    def Ideal_LowPass(self, filename):
        self.count += 1
        self.Active = True
        if (self.ids.new_image4.source != None):
            self.ids.new_image4.source = "" and os.remove(
                "new_image.jpg") and self.ids.new_image.remove_from_cache()
        if (self.Active == True):
            DirPath = ".\images"
            Files = os.listdir(DirPath)
            self.ids.new_image4.source = ""
            for File in Files:
                imgPath = os.path.join(DirPath, File)
                if (os.path.abspath(imgPath) == filename[0]):
                    img = cv.imread(imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            F = np.fft.fft2(img)
            F = np.fft.fftshift(F)
            M, N = img.shape[:2]
            D0 = 30
            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
            H = np.array(D <= D0, 'float')
            G = H * F
            G = np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            cv.imwrite("new_image" + str(self.count) + ".jpg", imgOut)
            self.ids.new_image4.source = "new_image" + str(self.count) + ".jpg"
            self.Active = False
        else:
            self.Active = False

    def ButterWorth_LowPass(self, filename):
        self.count += 1
        self.Active = True
        if (self.ids.new_image4.source != None):
            self.ids.new_image4.source = "" and os.remove(
                "new_image.jpg") and self.ids.new_image.remove_from_cache()
        if (self.Active == True):
            DirPath = ".\images"
            Files = os.listdir(DirPath)
            self.ids.new_image4.source = ""
            for File in Files:
                imgPath = os.path.join(DirPath, File)
                if (os.path.abspath(imgPath) == filename[0]):
                    img = cv.imread(imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            F = np.fft.fft2(img)
            F = np.fft.fftshift(F)
            M, N = img.shape[:2]
            D0 = 75
            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
            H = 1/np.power(1 + (D/D0), (2*2))
            G = H * F
            G = np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            cv.imwrite("new_image" + str(self.count) + ".jpg", imgOut)
            self.ids.new_image4.source = "new_image" + str(self.count) + ".jpg"
            self.Active = False
        else:
            self.Active = False

    def Gaussian_LowPass(self, filename):
        self.count += 1
        self.Active = True
        if (self.ids.new_image4.source != None):
            self.ids.new_image4.source = "" and os.remove(
                "new_image.jpg") and self.ids.new_image.remove_from_cache()
        if (self.Active == True):
            DirPath = ".\images"
            Files = os.listdir(DirPath)
            self.ids.new_image4.source = ""
            for File in Files:
                imgPath = os.path.join(DirPath, File)
                if (os.path.abspath(imgPath) == filename[0]):
                    img = cv.imread(imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            F = np.fft.fft2(img)
            F = np.fft.fftshift(F)
            M, N = img.shape[:2]
            D0 = 5
            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
            H = np.exp((-D ** 2) / ((2 * D0) ** 2))
            G = H * F
            G = np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            cv.imwrite("new_image" + str(self.count) + ".jpg", imgOut)
            self.ids.new_image4.source = "new_image" + str(self.count) + ".jpg"
            self.Active = False
        else:
            self.Active = False

    def Ideal_HighPass(self, filename):
        self.count += 1
        self.Active = True
        if (self.ids.new_image4.source != None):
            self.ids.new_image4.source = "" and os.remove(
                "new_image.jpg") and self.ids.new_image.remove_from_cache()
        if (self.Active == True):
            DirPath = ".\images"
            Files = os.listdir(DirPath)
            self.ids.new_image4.source = ""
            for File in Files:
                imgPath = os.path.join(DirPath, File)
                if (os.path.abspath(imgPath) == filename[0]):
                    img = cv.imread(imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            F = np.fft.fft2(img)
            F = np.fft.fftshift(F)
            M, N = img.shape[:2]
            D0 = 30
            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
            H = np.array(D > D0, 'float')
            G = H * F
            G = np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            cv.imwrite("new_image" + str(self.count) + ".jpg", imgOut)
            self.ids.new_image4.source = "new_image" + str(self.count) + ".jpg"
            self.Active = False
        else:
            self.Active = False

    def ButterWorth_HighPass(self, filename):
        self.count += 1
        self.Active = True
        if (self.ids.new_image4.source != None):
            self.ids.new_image4.source = "" and os.remove(
                "new_image.jpg") and self.ids.new_image.remove_from_cache()
        if (self.Active == True):
            DirPath = ".\images"
            Files = os.listdir(DirPath)
            self.ids.new_image4.source = ""
            for File in Files:
                imgPath = os.path.join(DirPath, File)
                if (os.path.abspath(imgPath) == filename[0]):
                    img = cv.imread(imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            F = np.fft.fft2(img)
            F = np.fft.fftshift(F)
            M, N = img.shape[:2]
            D0 = 75
            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
            H = 1/np.power(1 + (D0/D), (2*2))
            G = H * F
            G = np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            cv.imwrite("new_image" + str(self.count) + ".jpg", imgOut)
            self.ids.new_image4.source = "new_image" + str(self.count) + ".jpg"
            self.Active = False
        else:
            self.Active = False

    def Gaussian_HighPass(self, filename):
        self.count += 1
        self.Active = True
        if (self.ids.new_image4.source != None):
            self.ids.new_image4.source = "" and os.remove(
                "new_image.jpg") and self.ids.new_image.remove_from_cache()
        if (self.Active == True):
            DirPath = ".\images"
            Files = os.listdir(DirPath)
            self.ids.new_image4.source = ""
            for File in Files:
                imgPath = os.path.join(DirPath, File)
                if (os.path.abspath(imgPath) == filename[0]):
                    img = cv.imread(imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            F = np.fft.fft2(img)
            F = np.fft.fftshift(F)
            M, N = img.shape[:2]
            D0 = 5
            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt((np.power(U, 2) + np.power(V, 2)))
            H = 1 - np.exp((-D ** 2) / ((2 * D0) ** 2))
            G = H * F
            G = np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            cv.imwrite("new_image" + str(self.count) + ".jpg", imgOut)
            self.ids.new_image4.source = "new_image" + str(self.count) + ".jpg"
            self.Active = False
        else:
            self.Active = False
            pass
    pass


class Content_Check(FloatLayout):
    dismiss = ObjectProperty()
    pass


class WindowCheck(Popup):
    pass


class FifthScreen(Screen):
    file_path5 = StringProperty("No file chosen")
    the_popup5 = ObjectProperty(None)
    the_popup_check = ObjectProperty(None)
    kernel = np.zeros((3, 3), np.uint8)
    threshval = 0
    Active = False
    count = 0

    def Reset(self):
        self.ids.my_image5.source = "image5.png"
        self.ids.new_image5.source = "image5.png"

    def Reset_new_image(self):
        self.ids.new_image5.source = ""

    def open_popup_check(self):
        self.the_popup_check = WindowCheck()
        content_check = Content_Check(dismiss=self.dismiss)
        self.the_popup_check.add_widget(content_check)
        self.the_popup_check.open()

    def dismiss(self):
        self.the_popup_check.dismiss()

    def open_popup(self):
        self.the_popup5 = WindowPopup()
        content = Content4(load=self.load, cancel=self.cancel)
        self.the_popup5.add_widget(content)
        self.the_popup5.open()

    def cancel(self):
        self.the_popup5.dismiss()

    def load(self, selection):
        self.file_path5 = str(selection[0])
        self.the_popup5.dismiss()
        try:
            self.ids.my_image5.source = self.file_path5
        except:
            pass

    def checkbox_click(self, instance, value, topping):
        if (value == True):
            if (topping == "k1 = 6x6"):
                self.kernel = np.ones((6, 6), np.uint8)
            elif (topping == "k2 = 11x11"):
                self.kernel = np.ones((11, 11), np.uint8)
            elif (topping == "k3 = 15x15"):
                self.kernel = np.ones((15, 15), np.uint8)
        print("False")
        pass

    def slide_it(self, *args):
        self.threshval = int(args[1])
        self.slider_text.text = f"Threshval: {int(args[1])}"

    def check(self, source):
        if (self.ids.new_image5.source != None):
            self.ids.new_image5.source = ""
            os.remove("new_image.jpg")
            self.ids.new_image5.remove_from_cache()
        pass

    def Erosion(self):
        self.count += 1
        self.Active = True
        if (self.Active == True):
            DirPath = ".\images"
            Files = os.listdir(DirPath)
            for File in Files:
                imgPath = os.path.join(DirPath, File)
                if (os.path.abspath(imgPath) == self.file_path5):
                    img = cv.imread(imgPath)
            n = 255
            retval, imgB = cv.threshold(
                img, self.threshval, n, cv.THRESH_BINARY)
            if (self.kernel.all() == 0) or (self.kernel.all() == None):
                self.open_popup_check()
            else:
                print("Kernel : " + str(self.kernel))
                img_ero1 = cv.erode(imgB, self.kernel, iterations=1)
                cv.imwrite("new_image" + str(self.count) + ".jpg", img_ero1)
                self.ids.new_image5.source = "new_image" + \
                    str(self.count) + ".jpg"
                self.count += 1
                self.Active = False
        else:
            self.Acive = False

    def Delation(self):
        self.count += 1
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == self.file_path5):
                img = cv.imread(imgPath, 0)
        n = 255
        retval, imgB = cv.threshold(img, self.threshval, n, cv.THRESH_BINARY)
        img_ero1 = cv.dilate(imgB, self.kenrel, iterations=1)
        cv.imwrite("new_image" + str(self.count) + ".jpg", img_ero1)
        self.ids.new_image5.source = "new_image" + str(self.count) + ".jpg"
        print(self.count)

    def openning(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath, 0)
        n = 255
        imgB = cv.threshold(img, self.threshval, n, cv.THRESH_BINARY)
        openning = cv.morphologyEx(imgB, cv.MORPH_OPEN, self.kenrel)
        cv.imwrite("new_image" + str(self.count) + ".jpg", openning)
        self.ids.new_image5.source = "new_image" + str(self.count) + ".jpg"
        self.count += 1

    def closing(self, filename):
        DirPath = ".\images"
        Files = os.listdir(DirPath)
        for File in Files:
            imgPath = os.path.join(DirPath, File)
            if (os.path.abspath(imgPath) == filename[0]):
                img = cv.imread(imgPath, 0)
        n = 255
        imgB = cv.threshold(img, self.threshval, n, cv.THRESH_BINARY)
        closing = cv.morphologyEx(imgB, cv.MORPH_CLOSE, self.kenrel)
        cv.imwrite("new_image.jpg", closing)
        self.ids.new_image5.source = "new_image.jpg"
    pass


class Paint(Widget):
    flag = False
    A = []
    B = []
    count = 0

    def __init__(self, **kwargs):
        super(Paint, self).__init__(**kwargs)
        pass

    def on_touch_down(self, touch):
        super(Paint, self).on_touch_down(touch)
        random_shape = random.randint(1, 4)
        random_color = random.randint(100, 255)
        print(touch)
        if (self.flag == True and touch.pos[0] >= 0 and touch.pos[1] < 700 and touch.pos[1] > 244 and random_shape == 1):
            with self.canvas:
                Color(random_color/255, random_color/255, random_color/255)
                self.rect = RoundedRectangle(size=(120, 100),
                                             pos=(touch.pos[0] - 50,
                                                  touch.pos[1] - 50),
                                             border=(40, 40, 40, 40),
                                             group=u"rect")
                self.A.append([touch.pos[0], touch.pos[1]])
                self.ids.img2.canvas.add(self.rect)
        elif (self.flag == True and touch.pos[0] >= 0 and touch.pos[1] < 700 and touch.pos[1] > 244 and random_shape == 2):
            with self.canvas:
                Color(random_color/255, random_color/255, random_color/255)
                self.rect = Ellipse(size=(150, 150),
                                    pos=(touch.pos[0] - 50,
                                         touch.pos[1] - 50),
                                    segments=4,
                                    group=u"rect")
                self.A.append([touch.pos[0], touch.pos[1]])
                self.ids.img2.canvas.add(self.rect)
        elif (self.flag == True and touch.pos[0] >= 0 and touch.pos[1] < 700 and touch.pos[1] > 244 and random_shape == 3):
            with self.canvas:
                Color(random_color/255, random_color/255, random_color/255)
                self.rect = Ellipse(size=(150, 150),
                                    pos=(touch.pos[0] - 50,
                                         touch.pos[1] - 50),
                                    segments=3,
                                    group=u"rect")
                self.A.append([touch.pos[0], touch.pos[1]])
                self.ids.img2.canvas.add(self.rect)
        elif (self.flag == True and touch.pos[0] >= 0 and touch.pos[1] < 700 and touch.pos[1] > 244 and random_shape == 4):
            with self.canvas:
                Color(random_color/255, random_color/255, random_color/255)
                self.rect = Ellipse(size=(150, 150),
                                    pos=(touch.pos[0] - 50,
                                         touch.pos[1] - 50),
                                    segments=5,
                                    group=u"rect")
                self.A.append([touch.pos[0], touch.pos[1]])
                self.ids.img2.canvas.add(self.rect)
        pass

    def remove_rect(self):
        try:
            self.canvas.remove_group(u"rect")
        except:
            pass
    pass

    def add_rect(self):
        for i in range(len(self.A)):
            for j in range(2):
                t = self.A[i]
                self.B.append(t[j])
            pos_x = self.B[self.count]
            pos_y = self.B[self.count + 1]
            self.count = self.count + 2
            with self.canvas:
                Color(0, 0, 0)
                self.rect = Rectangle(size=(80, 80),
                                      pos=(pos_x - 40,
                                           pos_y - 40),
                                      group=u"rect")
        try:
            self.canvas.add_group(u"rect")
        except:
            pass


class Sixth(Screen):
    show = False
    count = 0

    def Reset(self):
        self.paint.canvas.clear()
        self.paint.ids.img.remove_from_cache()
        self.paint.ids.img2.remove_from_cache()

    def save_img(self, img, count):
        cv.imwrite(
            "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\new_image" + str(count) + ".png", img)
        pass

    def add_rect(self):
        self.paint.add_rect()

    def __init__(self, **kwargs):
        super(Sixth, self).__init__(**kwargs)
        self.boxlayout = BoxLayout(
            orientation='vertical', size=(50, 50))
        self.paint = Paint()
        self.boxlayout.add_widget(self.paint)
        self.add_widget(self.boxlayout)

    def export_img(self, *args):

        self.paint.ids.img2.export_to_png("shape.png")
        self.delete_shape()
        self.paint.ids.img2.canvas.clear()
        self.show = True

    def delete_shape(self):
        self.paint.remove_rect()

    def draw_shape(self):
        self.paint.flag = True
        self.show = False

    def Boundary(self):
        self.count += 1
        if self.show == True:
            while self.show:
                self.paint.flag = False
                img = cv.imread("shape.png")
                kernel = np.ones((15, 15), dtype=np.uint8)
                closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
                erosion = cv.erode(closing, kernel, iterations=1)
                output = closing - erosion
                self.save_img(output, self.count)
                if self.paint.ids.img.source == "Black.jpg":
                    self.paint.ids.img.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\new_image" + \
                        str(self.count) + ".png"
                self.show = False
                break
        elif self.show == False:
            os.remove("new_image.png")
            self.paint.ids.img2.source = "Black.jpg"
            self.show = True

    def holefilling(self):
        img_input = cv.imread("D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\new_image" +
                              str(self.count) + ".png", 0)
        self.count += 1
        h, w = img_input.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        imfloodfill = img_input.copy()
        cv.floodFill(imfloodfill, mask, (0, 0), 255)
        imfloodfill_not = cv.bitwise_not(imfloodfill)
        img_output = img_input | imfloodfill_not
        (th, img_output) = cv.threshold(
            img_output, 127, 255, cv.THRESH_BINARY_INV)
        self.save_img(img_output, self.count)
        self.paint.ids.img.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\new_image" + \
            str(self.count) + ".png"
    pass


class Seventh(Screen):
    file_path7 = StringProperty("No file chosen")
    the_popup7 = ObjectProperty(None)
    count = 0
    threshval = 0
    threshold = 1
    line = 1
    edge = 1
    optimum_threshold = 1
    A = []
    nA = 0
    image_sevent = module.SeventScreen()

    def Reset(self):
        DirPath_NewImage = ".\\new_img"
        Files_NewImage = os.listdir(DirPath_NewImage)
        self.delete_new_img(DirPath_NewImage, Files_NewImage)

    def delete_new_img(self, DirPath_NewImage, Files_NewImage):
        for f in Files_NewImage:
            os.remove(os.path.join(DirPath_NewImage, f))

    def open_popup(self):
        self.the_popup7 = WindowPopup()
        content = Content6(load=self.load)
        self.the_popup7.add_widget(content)
        self.the_popup7.open()

    def load(self, selection):
        self.file_path7 = str(selection[0])
        self.the_popup7.dismiss()
        try:
            self.ids.my_image7.source = self.file_path7
        except:
            pass

    def slide_it(self, *args):
        self.threshval = int(args[1])
        self.slider_text.text = f"Threshval: {int(args[1])}"

    def line_detection(self):
        self.count += 1
        self.nA += 1
        self.A.append(self.file_path7)
        print(self.A)
        if len(self.A) == 1 or self.A[self.nA - 2] == self.A[self.nA - 1]:
            self.image_sevent.line_detection(
                self.file_path7, self.threshval, self.count)
            if self.line == 1:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\LineDetection" + \
                    str(self.count) + ".png"
                self.line += 1
                self.ids.label_img_after.text = "imB"
            elif self.line == 2:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\LineDetection" + \
                    str(self.count + 1) + ".png"
                self.ids.label_img_after.text = "img_lap"
                self.line += 1
            elif self.line == 3:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\LineDetection" + \
                    str(self.count + 2) + ".png"
                self.ids.label_img_after.text = "img_lap_abs"
                self.line = 1
        elif (self.A[self.nA - 2] != self.A[self.nA - 1]):
            DirPath_NewImage = ".\\new_img"
            Files_NewImage = os.listdir(DirPath_NewImage)
            self.delete_new_img(DirPath_NewImage, Files_NewImage)
            self.nA = 0
            self.A.clear()
        print(self.count)
        pass

    def edge_detection(self):
        self.count += 1
        self.nA += 1
        self.A.append(self.file_path7)
        if len(self.A) == 1 or self.A[self.nA - 2] == self.A[self.nA - 1]:
            self.image_sevent.edge_detection(self.file_path7, self.count)
            if self.edge == 1:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Edge_Detection" + \
                    str(self.count) + ".png"
                self.ids.label_img_after.text = "img"
                self.edge += 1
            elif self.edge == 2:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Edge_Detection" + \
                    str(self.count + 1) + ".png"
                self.ids.label_img_after.text = "grad_x"
                self.edge += 1
            elif self.edge == 3:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Edge_Detection" + \
                    str(self.count + 2) + ".png"
                self.ids.label_img_after.text = "grad_y"
                self.edge += 1
            elif self.edge == 4:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Edge_Detection" + \
                    str(self.count + 3) + ".png"
                self.ids.label_img_after.text = "grad"
                self.edge = 1
        elif (self.A[self.nA - 2] != self.A[self.nA - 1]):
            DirPath_NewImage = ".\\new_img"
            Files_NewImage = os.listdir(DirPath_NewImage)
            self.delete_new_img(DirPath_NewImage, Files_NewImage)
            self.nA = 0
            self.A.clear()
        print(self.count)
        pass

    def thresholding(self):
        self.count += 1
        self.nA += 1
        self.A.append(self.file_path7)
        if len(self.A) == 1 or self.A[self.nA - 2] == self.A[self.nA - 1]:
            self.image_sevent.thresholding(
                self.file_path7, self.threshval, self.count)
            if self.threshold == 1:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Thresholding_Detection" + \
                    str(self.count) + ".png"
                self.ids.label_img_after.text = "Original Image Blur"
                self.threshold += 1
            elif self.threshold == 2:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Thresholding_Detection" + \
                    str(self.count + 1) + ".png"
                self.ids.label_img_after.text = "Thresholding"
                self.threshold += 1
            elif self.threshold == 3:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Thresholding_Detection" + \
                    str(self.count + 2) + ".png"
                self.ids.label_img_after.text = "Mean Thresholding"
                self.threshold += 1
            elif self.threshold == 4:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Thresholding_Detection" + \
                    str(self.count + 3) + ".png"
                self.ids.label_img_after.text = "Gaussian Thresholding"
                self.threshold = 1
        elif (self.A[self.nA - 2] != self.A[self.nA - 1]):
            DirPath_NewImage = ".\\new_img"
            Files_NewImage = os.listdir(DirPath_NewImage)
            self.delete_new_img(DirPath_NewImage, Files_NewImage)
            self.nA = 0
            self.A.clear()
            pass

    def optimum_thresholding(self):
        self.count += 1
        self.nA += 1
        self.A.append(self.file_path7)
        if len(self.A) == 1 or self.A[self.nA - 2] == self.A[self.nA - 1]:
            self.image_sevent.optimum_thresholding(
                self.file_path7, self.threshval, self.count)
            if self.optimum_threshold == 1:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Optimum_Thresholding_Detection" + \
                    str(self.count) + ".png"
                self.ids.label_img_after.text = "Original Image Blur"
                self.optimum_threshold += 1
            elif self.optimum_threshold == 2:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Optimum_Thresholding_Detection" + \
                    str(self.count + 1) + ".png"
                self.ids.label_img_after.text = "OTSU thres"
                self.optimum_threshold += 1
            elif self.optimum_threshold == 3:
                self.ids.new_image7.source = "D:\Hk2_Nam3\Image_Processing\Kivy_tutorial\\new_img\\Optimum_Thresholding_Detection" + \
                    str(self.count + 2) + ".png"
                self.ids.label_img_after.text = "Local Thres"
                self.optimum_threshold = 1
        elif (self.A[self.nA - 2] != self.A[self.nA - 1]):
            DirPath_NewImage = ".\\new_img"
            Files_NewImage = os.listdir(DirPath_NewImage)
            self.delete_new_img(DirPath_NewImage, Files_NewImage)
            self.nA = 0
            self.A.clear()
            pass


class InformMySelf(Screen):
    pass


kv = Builder.load_file("image.kv")


class ImageApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    ImageApp().run()
