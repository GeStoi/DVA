import sys
import subprocess
import cv2
import numpy as np
import os
import keyboard
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from HEplan import HE
from DVAbase import DVAbase
from Flow import Flow

# 检查并安装 cupy
def install_cupy():
    try:
        import cupy as cp
    except ImportError:
        print("cupy 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda12x"])
        print("cupy 安装完成。")
        import cupy as cp
    return cp
cp = install_cupy()

class DVA_App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.processed_frames = None
        self.green_masked_images = None
        self.black_masked_images = None

    def initUI(self):
        self.setWindowTitle('DVA')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.video_path_label = QLabel('视频文件路径:')
        self.layout.addWidget(self.video_path_label)

        self.video_path_input = QLineEdit()
        self.layout.addWidget(self.video_path_input)

        self.browse_button = QPushButton('浏览')
        self.browse_button.clicked.connect(self.browse_file)
        self.layout.addWidget(self.browse_button)

        self.threshold_label = QLabel('阈值:')
        self.layout.addWidget(self.threshold_label)

        self.threshold_input = QLineEdit('5')
        self.layout.addWidget(self.threshold_input)

        self.select_frame_label = QLabel('选择帧:')
        self.layout.addWidget(self.select_frame_label)

        self.select_frame_input = QLineEdit('68')
        self.layout.addWidget(self.select_frame_input)

        self.process_button = QPushButton('处理')
        self.process_button.clicked.connect(self.process_video)
        self.layout.addWidget(self.process_button)

        self.save_frame_button = QPushButton('保存指定帧')
        self.save_frame_button.clicked.connect(self.save_selected_frame)
        self.layout.addWidget(self.save_frame_button)

        self.result_label = QLabel('处理结果:')
        self.layout.addWidget(self.result_label)

        self.result_display = QLabel()
        self.layout.addWidget(self.result_display)

        self.log_text = QTextEdit()
        self.layout.addWidget(self.log_text)

    def browse_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择视频文件", "", "All Files (*);;DCM Files (*.dcm)")
        self.video_path_input.setText(file_path)

    def process_video(self):
        video_path = self.video_path_input.text()
        threshold = int(self.threshold_input.text())
        select_frame = int(self.select_frame_input.text())

        image_stack = DVAbase.read_frames(video_path)
        if image_stack is None:
            self.log_text.append("无法读取DICOM文件或文件中没有帧。")
            return

        num_frames, frame_height, frame_width = image_stack.shape
        window_size = min(10, int(np.ceil(num_frames / 4)))
        self.log_text.append(f"窗口大小设置为：{window_size}")

        if num_frames < 20:
            self.log_text.append("输入视频时长过短")
            return

        processed_frames = np.zeros((0, frame_height, frame_width), dtype=np.uint8)
        mask_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.uint8)
        frame_index = 0

        pbar = tqdm(total=num_frames - window_size + 1, desc="Processing frames")
        while frame_index < num_frames - window_size + 1:
            image = image_stack[frame_index:frame_index + window_size]
            image = np.array([DVAbase.preprocess_image(frame) for frame in image])
            image = DVAbase.calculate_windowed_variance(image, threshold, 0, 1, 0, 1)
            image = HE.enhance_contrast(image, method="Bright", blocks=12, threshold=10.0)
            full_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
            full_frame[0:frame_height, 0:frame_width] = image
            processed_frames = np.concatenate((processed_frames, [full_frame]), axis=0)

            mask_frames[frame_index] = full_frame
            frame_index += 1
            pbar.update(1)
        pbar.close()

        avg_image = np.mean(np.mean(processed_frames, axis=0))
        self.log_text.append(f"当前平均像素值：{avg_image}")
        if avg_image > 10:
            self.log_text.append("当前平均值过大，请调大阈值threshold")
        elif avg_image < 5:
            self.log_text.append("当前平均值过小，请调小阈值threshold")

        green_masked_images = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
        black_masked_images = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
        masked_images = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)

        for i in range(window_size - 1):
            green_masked_images[i] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            black_masked_images[i] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            masked_images[i] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        pbar = tqdm(total=num_frames - (window_size - 1), desc="Creating masked images")
        for i in range(window_size - 1, num_frames):
            mask = mask_frames[i - window_size + 1]
            original_frame = image_stack[i].astype(np.uint8)
            original_frame_rgb = np.stack([original_frame] * 3, axis=-1)

            green_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            green_mask[mask > 10] = [0, 255, 0]
            green_masked_image = np.where(mask[..., None] == 0, original_frame_rgb, green_mask)

            black_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            black_masked_image = np.where(mask[..., None] == 0, original_frame_rgb, black_mask)

            green_masked_images[i] = green_masked_image
            black_masked_images[i] = black_masked_image
            masked_images[i] = np.stack((processed_frames[i - (window_size - 1)],) * 3, axis=-1)

            pbar.update(1)
        pbar.close()

        self.processed_frames = processed_frames
        self.green_masked_images = green_masked_images
        self.black_masked_images = black_masked_images

        output_folder = "demo1"
        os.makedirs(output_folder, exist_ok=True)
        green_output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_{threshold}_{select_frame}_G.png")
        black_output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_{threshold}_{select_frame}_B.png")
        cv2.imwrite(green_output_path, green_masked_images[select_frame])
        cv2.imwrite(black_output_path, black_masked_images[select_frame])

        self.log_text.append(f"已保存第{select_frame}帧的绿色掩码图像到 {green_output_path}")
        self.log_text.append(f"已保存第{select_frame}帧的黑色掩码图像到 {black_output_path}")

        # 显示处理结果
        green_pixmap = QPixmap(green_output_path)
        self.result_display.setPixmap(green_pixmap.scaledToWidth(400))

    def save_selected_frame(self):
        if self.green_masked_images is None or self.black_masked_images is None:
            self.log_text.append("请先处理视频以生成处理结果。")
            return

        video_path = self.video_path_input.text()
        threshold = int(self.threshold_input.text())
        select_frame = int(self.select_frame_input.text())

        output_folder = "demo1"
        os.makedirs(output_folder, exist_ok=True)
        green_output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_{threshold}_{select_frame}_G.png")
        black_output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_{threshold}_{select_frame}_B.png")

        cv2.imwrite(green_output_path, self.green_masked_images[select_frame])
        cv2.imwrite(black_output_path, self.black_masked_images[select_frame])

        self.log_text.append(f"已保存第{select_frame}帧的绿色掩码图像到 {green_output_path}")
        self.log_text.append(f"已保存第{select_frame}帧的黑色掩码图像到 {black_output_path}")

        # 显示处理结果
        green_pixmap = QPixmap(green_output_path)
        self.result_display.setPixmap(green_pixmap.scaledToWidth(400))

if __name__ == '__main__':
    DVAbase = DVAbase()
    HE = HE()
    Flow = Flow()
    app = QApplication(sys.argv)
    ex = DVA_App()
    ex.show()
    sys.exit(app.exec_())