import numpy as np
import cv2

class Flow:
    def __init__(self, ws=10, amp_con=5, arror_amp_con=3):
        self.ws = ws
        self.amp_con = amp_con
        self.arror_amp_con = arror_amp_con
        self.pi = np.pi

    def calculate_flow(self, prev_gray, next_gray):
        h, w = prev_gray.shape
        Vx = np.zeros((h // 8, w // 8), dtype=np.float32)
        Vy = np.zeros((h // 8, w // 8), dtype=np.float32)

        for m in range(Vx.shape[0]):
            for n in range(Vx.shape[1]):
                pixel_x = 8 // 2 + m * 8
                pixel_y = 8 // 2 + n * 8

                lig_gra_x = (prev_gray[pixel_x + 1, pixel_y] - prev_gray[pixel_x, pixel_y] +
                             prev_gray[pixel_x + 1, pixel_y + 1] - prev_gray[pixel_x, pixel_y + 1] +
                             next_gray[pixel_x + 1, pixel_y] - next_gray[pixel_x, pixel_y] +
                             next_gray[pixel_x + 1, pixel_y + 1] - next_gray[pixel_x, pixel_y + 1]) / 4.0
                lig_gra_y = (prev_gray[pixel_x, pixel_y + 1] - prev_gray[pixel_x, pixel_y] +
                             prev_gray[pixel_x + 1, pixel_y + 1] - prev_gray[pixel_x + 1, pixel_y] +
                             next_gray[pixel_x, pixel_y + 1] - next_gray[pixel_x, pixel_y] +
                             next_gray[pixel_x + 1, pixel_y + 1] - next_gray[pixel_x + 1, pixel_y]) / 4.0
                lig_gra_t = (next_gray[pixel_x, pixel_y] - prev_gray[pixel_x, pixel_y] +
                             next_gray[pixel_x + 1, pixel_y] - prev_gray[pixel_x + 1, pixel_y] +
                             next_gray[pixel_x, pixel_y + 1] - prev_gray[pixel_x, pixel_y + 1] +
                             next_gray[pixel_x + 1, pixel_y + 1] - prev_gray[pixel_x + 1, pixel_y + 1]) / 4.0

                Vx[m, n] = -1 * lig_gra_t * lig_gra_x / (self.ws + lig_gra_x**2 + lig_gra_y**2)
                Vy[m, n] = -1 * lig_gra_t * lig_gra_y / (self.ws + lig_gra_x**2 + lig_gra_y**2)

        flow_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        for m in range(Vx.shape[0]):
            for n in range(Vx.shape[1]):
                amp_x, amp_y = int(Vx[m, n]), int(Vy[m, n])
                amp = int(np.sqrt(Vx[m, n]**2 + Vy[m, n]**2))
                if amp != 0:
                    self.optical_paint(flow_img, pixel_x, pixel_y, amp_x, amp_y, amp)

        return flow_img, Vx, Vy

    def optical_paint(self, frame, pixel_x, pixel_y, amp_x, amp_y, amp):
        p = (pixel_x, pixel_y)
        a, b = int(amp_x * self.amp_con), int(amp_y * self.amp_con)
        q = (p[0] + a, p[1] + b)
        angle = np.arctan2(p[1] - q[1], p[0] - q[0])
        cv2.line(frame, p, q, (0, 0, 255), 1)
        cv2.arrowedLine(frame, q, (int(q[0] + self.arror_amp_con * np.cos(angle + self.pi / 4)),
                                   int(q[1] + self.arror_amp_con * np.sin(angle + self.pi / 4))), (0, 0, 255), 1)
        cv2.arrowedLine(frame, q, (int(q[0] + self.arror_amp_con * np.cos(angle - self.pi / 4)),
                                   int(q[1] + self.arror_amp_con * np.sin(angle - self.pi / 4))), (0, 0, 255), 1)