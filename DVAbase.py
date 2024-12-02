import cv2
import numpy as np
import pydicom
import cupy as cp
import keyboard
from matplotlib import pyplot as plt

class DVAbase:
    def __unit8__(self):
        pass

    def calculate_windowed_variance(self, image_stack, threshold = 0, x_left = 0, x_right = 1, y_up = 0, y_down = 1):
        """
        窗口化方差
        :param image_stack: numpy.array, uint8, 3-dim
        :return var_img_stack: numpy.array, uint8, 3-dim
        """
        num_frames, height, width = image_stack.shape
        # 比例化窗口赋值
        x_start = int(x_left * width)
        x_end = int(x_right * width)
        y_start = int(y_up * height)
        y_end = int(y_down * height)
        image_stack = image_stack[:, y_start:y_end, x_start:x_end]
        # gpu方差运算
        # 分段阈值
        image_stack_gpu = cp.asarray(image_stack)
        avg_image = cp.mean(image_stack_gpu, axis=0)
        variance_image = cp.var(image_stack_gpu - avg_image, axis=0)
        zero_array = cp.zeros_like(variance_image)
        sqrt_val = cp.sqrt(variance_image)
        doubled = variance_image * 2
        variance_image = cp.where(variance_image < 0.5 * threshold, zero_array,
                                  cp.where(variance_image < threshold, sqrt_val,
                                           cp.where(variance_image < 2 * threshold, doubled, variance_image)))
        variance_image = cp.asnumpy(variance_image)
        var_img_stack = variance_image.astype(np.uint8)
        return var_img_stack

    def normalize_image(self, img):
        """
        归一化
        :param img: numpy.array, uint8, 2-dim
        :return mor_img: numpy.array, uint8, 2-dim
        """
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        normalized_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        nor_img = normalized_image.astype(np.uint8)
        return nor_img

    def adjust_brightness_contrast(self, img, brightness = 0, contrast = 0):
        """
        调整亮度和对比度 ### 后面考虑换成其他方法
        :param img: numpy.array, uint8, 2-dim
        :return ctr_img: numpy.array, uint8, 2-dim
        """
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            ctr_img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
        return ctr_img

    def histogram_equalization(self, img):
        """
        直方图均衡化，已被HEplan取缔
        :param img: numpy.array, uint8, 2-dim
        :return hist_img: numpy.array, uint8, 2-dim
        """
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        hist_img = cv2.equalizeHist(img)
        return hist_img

    def apply_bilateral_filter(self, img):
        """
        双边滤波，已被EVM取缔
        :param img: numpy.array, uint8, 2-dim
        :return filtered_img: numpy.array, uint8, 2-dim
        """
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        filter_img = cv2.bilateralFilter(img, d = 5, sigmaColor = 150, sigmaSpace = 150)
        return filter_img

    def edge_detection(self, img, threshold1 = 0):
        """
        边缘检测，Canny算子
        :param img: numpy.array, uinit8, 2-dim
        :return edge: numpy.array, uinit8, 2-dim
        """
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        threshold2 = min(10 * threshold1, threshold1 + 100)
        edge = cv2.Canny(img, threshold1, threshold2)
        return edge

    def plot_images(self, img1, img2, img3, img4):
        """
        图像循环展示
        :param img1: 原始图像
        :param img2: 处理后图像
        :param img3: 平均颜色掩码
        :param img4: 绿色掩码
        :return:
        """
        plt.figure()
        plt.subplot(221)
        plt.imshow(img1, cmap='gray')
        plt.title('Original Image')

        plt.subplot(222)
        plt.imshow(img2)
        plt.title('Processed Image')

        plt.subplot(223)
        plt.imshow(img3)
        plt.title('Black Masked Image')

        plt.subplot(224)
        plt.imshow(img4)
        plt.title('Green Masked Image')

        plt.show()

    def get_key(self):
        # 等待按键事件
        event = keyboard.read_event(suppress=True)
        # 返回按键的 ASCII 码
        return event.name

    def preprocess_image(self, image):
        # 平滑
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def reprocess_image(self, image):
        # 对比度，亮度
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # alpha控制对比度，beta控制亮度
        return image

    def read_frames(self, file_path):
        if file_path.endswith('.dcm'):
            # 读取 DICOM 文件
            ds = pydicom.dcmread(file_path)
            frames = ds.pixel_array
        elif file_path.endswith('.mp4'):
            # 读取 MP4 视频文件
            cap = cv2.VideoCapture(file_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frames = np.array(frames)
            cap.release()
        else:
            raise ValueError("Unsupported file format")
        return frames