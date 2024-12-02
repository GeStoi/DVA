from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt

class HE():
    def __init__( self ):
        pass

    def enhance_contrast(self, img, method="CLAHE", level=256, blocks=8, threshold=10):
        """
        对比增强，类主函数
        :param img: 输入图像 
        :param method: 处理方法，CLAHE/ LRSHE
        :param level: 彩色图或灰度图的值域
        :param blocks: CLAHE当中分割行列次数
        :param threshold: CLAHE当中限制对比度的阈值
        :return image: 均衡化结果
        """
        # 选择处理方法
        if method in ["CLAHE", "clahe"]:
           he_func = self.contrast_limited_ahe  # CLAHE
        elif method in ["Bright", "bright", "bright_level"]:
           he_func = self.bright_wise_histequal  # LRSHE

        # 检验灰度图还是彩色图
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            channel_num = 1
        elif len(img_arr.shape) == 3:
            channel_num = img_arr.shape[2]

        if channel_num == 1:
            # 灰度图
            arr = he_func(img_arr, level=level, blocks=blocks, threshold=threshold)
            img_res = Image.fromarray(arr)
        elif channel_num == 3 or channel_num == 4:
            # RGB/RGBA(png)
            rgb_arr = [None] * 3
            rgb_img = [None] * 3
            # 分别处理
            for k in range(3):
                rgb_arr[k] = he_func(img_arr[:, :, k], level=level, blocks=blocks, threshold=threshold)
                rgb_img[k] = Image.fromarray(rgb_arr[k])
            img_res = Image.merge("RGB", tuple(rgb_img))

        return img_res

    def contrast_limited_ahe(self, img_arr, level=256, blocks=8, threshold=10, **args):
        """
        CLAHE: Contrast Limited Adaptive Histogram Equalization
        :param img_arr: numpy.array, uint8, 2-dim
        :param level: 灰度值值域
        :param blocks: 分割次数
        :param threshold: 对比度限制阈值
        :param args: 杂项
        :return arr: image array
        """
        (m, n) = img_arr.shape
        print(m, n)
        block_m = int(m / blocks)
        block_n = int(n / blocks)

        # 区域分割并逐个计算CDF，存入2维列表maps
        maps = []
        for i in range(blocks):
            row_maps = []
            for j in range(blocks):
                # block border
                si, ei = i * block_m, (i + 1) * block_m
                sj, ej = j * block_n, (j + 1) * block_n

                # block image array
                block_img_arr = img_arr[si: ei, sj: ej]

                # calculate histogram and cdf
                hists = self.calc_histogram_(block_img_arr)
                clip_hists = self.clip_histogram_(hists, threshold=threshold)  # clip histogram
                hists_cdf = self.calc_histogram_cdf_(clip_hists, block_m, block_n, level)

                # save
                row_maps.append(hists_cdf)
            maps.append(row_maps)

        # 像素插值，使用四个临近函数
        # 区分大小写
        arr = img_arr.copy()
        for i in range(m):
            for j in range(n):
                r = int((i - block_m / 2) / block_m)  # 行左起始
                c = int((j - block_n / 2) / block_n)  # 列左起始

                x1 = (i - (r + 0.5) * block_m) / block_m  # 左起始图中心距x轴
                y1 = (j - (c + 0.5) * block_n) / block_n  # 左起始图中心距y轴

                lu = 0  # mapping value of the left up cdf
                lb = 0  # 左终止
                ru = 0  # 起始
                rb = 0  # 右终止

                # 四角近邻值
                if r < 0 and c < 0:
                    arr[i][j] = maps[r + 1][c + 1][img_arr[i][j]]
                elif r < 0 and c >= blocks - 1:
                    arr[i][j] = maps[r + 1][c][img_arr[i][j]]
                elif r >= blocks - 1 and c < 0:
                    arr[i][j] = maps[r][c + 1][img_arr[i][j]]
                elif r >= blocks - 1 and c >= blocks - 1:
                    arr[i][j] = maps[r][c][img_arr[i][j]]
                # 四边线性插值
                elif r < 0 or r >= blocks - 1:
                    if r < 0:
                        r = 0
                    elif r > blocks - 1:
                        r = blocks - 1
                    left = maps[r][c][img_arr[i][j]]
                    right = maps[r][c + 1][img_arr[i][j]]
                    arr[i][j] = (1 - y1) * left + y1 * right
                elif c < 0 or c >= blocks - 1:
                    if c < 0:
                        c = 0
                    elif c > blocks - 1:
                        c = blocks - 1
                    up = maps[r][c][img_arr[i][j]]
                    bottom = maps[r + 1][c][img_arr[i][j]]
                    arr[i][j] = (1 - x1) * up + x1 * bottom
                # 区域内像素线性插值
                else:
                    lu = maps[r][c][img_arr[i][j]]
                    lb = maps[r + 1][c][img_arr[i][j]]
                    ru = maps[r][c + 1][img_arr[i][j]]
                    rb = maps[r + 1][c + 1][img_arr[i][j]]
                    arr[i][j] = (1 - y1) * ((1 - x1) * lu + x1 * lb) + y1 * ((1 - x1) * ru + x1 * rb)
        arr = arr.astype("uint8")
        return arr

    def bright_wise_histequal(self, img_arr, level=256, **args):
        """
        LRSHE: Local Region Stretch Histogram Equalization
        :param img_arr: numpy.array, uint8, 2-dim
        :param level: 灰度值值域
        :param args: 杂项
        :return arr: image array
        """
        def special_histogram(img_arr, min_v, max_v):
            """
            限定值域的直方图
            :param img_arr: numpy.array, 1-dim
            :param min_v: min gray scale
            :param max_v: max gray scale
            :return hists: list, length = max_v - min_v + 1
            """
            hists = [0 for _ in range(max_v - min_v + 1)]
            for v in img_arr:
                hists[v - min_v] += 1
            return hists

        def special_histogram_cdf(hists, min_v, max_v):
            """
            限定值域的直方图cdf
            :param hists: list
            :param min_v: min gray scale
            :param max_v: max gray scale
            :return hists_cdf: numpy.array
            """
            hists_cumsum = np.cumsum(np.array(hists))
            hists_cdf = (max_v - min_v) / hists_cumsum[-1] * hists_cumsum + min_v
            hists_cdf = hists_cdf.astype("uint8")
            return hists_cdf

        def pseudo_variance(arr):
            """
            特殊形式的方差（平均绝对差）
            :param arr: numpy.array, 1-dim
            :return np.mean(arr_abs): numpy, 1-dim
            """
            arr_abs = np.abs(arr - np.mean(arr))
            return np.mean(arr_abs)

        # 按灰度值划分三块等大小区域
        (m, n) = img_arr.shape
        hists = self.calc_histogram_(img_arr)
        hists_arr = np.cumsum(np.array(hists))
        hists_ratio = hists_arr / hists_arr[-1]

        scale1 = None
        scale2 = None
        for i in range(len(hists_ratio)):
            if hists_ratio[i] >= 0.333 and scale1 == None:
                scale1 = i
            if hists_ratio[i] >= 0.667 and scale2 == None:
                scale2 = i
                break

        # 划分区域
        dark_index = (img_arr <= scale1)
        mid_index = (img_arr > scale1) & (img_arr <= scale2)
        bright_index = (img_arr > scale2)

        # 平均绝对差
        dark_variance = pseudo_variance(img_arr[dark_index])
        mid_variance = pseudo_variance(img_arr[mid_index])
        bright_variance = pseudo_variance(img_arr[bright_index])

        # 构建三块图像
        dark_img_arr = np.zeros_like(img_arr)
        mid_img_arr = np.zeros_like(img_arr)
        bright_img_arr = np.zeros_like(img_arr)

        # 分别均衡化
        dark_hists = special_histogram(img_arr[dark_index], 0, scale1)
        dark_cdf = special_histogram_cdf(dark_hists, 0, scale1)

        mid_hists = special_histogram(img_arr[mid_index], scale1, scale2)
        mid_cdf = special_histogram_cdf(mid_hists, scale1, scale2)

        bright_hists = special_histogram(img_arr[bright_index], scale2, level - 1)
        bright_cdf = special_histogram_cdf(bright_hists, scale2, level - 1)

        def plot_hists(arr):
            hists = [0 for i in range(256)]
            for a in arr:
                hists[a] += 1
            self.draw_histogram_(hists)

        # 映射
        dark_img_arr[dark_index] = dark_cdf[img_arr[dark_index]]
        mid_img_arr[mid_index] = mid_cdf[img_arr[mid_index] - scale1]
        bright_img_arr[bright_index] = bright_cdf[img_arr[bright_index] - scale2]

        # 加权和
        # fractor = dark_variance + mid_variance + bright_variance
        # arr = (dark_variance * dark_img_arr + mid_variance * mid_img_arr + bright_variance * bright_img_arr)/fractor
        arr = dark_img_arr + mid_img_arr + bright_img_arr
        arr = arr.astype("uint8")
        return arr

    def calc_histogram_(self, gray_arr, level=256):
        """
        计算灰度图的直方图
        :param gray_arr: numpy,array, unit8, 2-dim
        :param level: 灰度值值域
        :return hist: list
        """
        hists = [0 for _ in range(level)]
        for row in gray_arr:
            for p in row:
                hists[p] += 1
        return hists

    def calc_histogram_cdf_(self, hists, block_m, block_n, level=256):
        """
        计算直方图的cdf
        :param hists: list
        :param block_m: height of histogram block
        :param block_n: width of histogram block
        :param level: 灰度值值域
        :return hist_cdf: numpy.array
        """
        hists_cumsum = np.cumsum(np.array(hists))
        const_a = (level - 1) / (block_m * block_n)
        hists_cdf = (const_a * hists_cumsum).astype("uint8")
        return hists_cdf

    def clip_histogram_(self, hists, threshold=10):
        """
        截取直方图的峰值，均匀分配至全部层次
        :param hists: list
        :param threshold: 直方图最大值阈值
        :return clip_hist: list
        """
        all_sum = sum(hists)
        threshold_value = all_sum / len(hists) * threshold
        total_extra = sum([h - threshold_value for h in hists if h >= threshold_value])
        mean_extra = total_extra / len(hists)

        clip_hists = [0 for _ in hists]
        for i in range(len(hists)):
            if hists[i] >= threshold_value:
                clip_hists[i] = int(threshold_value + mean_extra)
            else:
                clip_hists[i] = int(hists[i] + mean_extra)

        return clip_hists

    def draw_histogram_(self, hists):
        """
        绘制直方图
        :param hists: list
        :return:
        """
        plt.figure()
        plt.bar(range(len(hists)), hists)
        plt.show()
