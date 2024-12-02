#%%
import cv2
import numpy as np
import cupy as cp
import os
import keyboard
from tqdm import tqdm
from HEplan import HE
from DVAbase import DVAbase
from Flow import Flow

DVAbase = DVAbase()
HE = HE()
Flow = Flow()
video_path = "Input/IMG-0024-00001.dcm"  # 输入视频文件路径
threshold = 5 # 限高，要根据输出的平均方差等统计值调整，一般要使得平均值在 10以内
select_frame = 55  # 保存第10帧
x_left = 0
x_right = 1
y_up = 0
y_down = 1

image_stack = DVAbase.read_frames(video_path)
if image_stack is None:
    print("无法读取DICOM文件或文件中没有帧。")
print('image_stack shape', image_stack.shape)
print('image_stack type', type(image_stack))

num_frames, frame_height, frame_width = image_stack.shape
window_size = min(10, int(np.ceil(num_frames / 4)))      # 设置窗口大小，动态性要求越高数值越小
print(f"窗口大小设置为：{window_size}")
if num_frames < 20:
    print("输入视频时长过短")

processed_frames = np.zeros((0, frame_height, frame_width), dtype=np.uint8)
mask_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.uint8)
frame_index = 0

pbar = tqdm(total=num_frames - window_size + 1, desc="Processing frames")
frame_index = 0
while frame_index < num_frames - window_size + 1:
    image = image_stack[frame_index:frame_index + window_size]
    image = np.array([DVAbase.preprocess_image(frame) for frame in image])
    image = DVAbase.calculate_windowed_variance(image, threshold, x_left, x_right, y_up, y_down)
    image = HE.enhance_contrast(image, method="Bright", blocks=12, threshold=10.0)
    full_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
    x_start = int(x_left * frame_width)
    x_end = int(x_right * frame_width)
    y_start = int(y_up * frame_height)
    y_end = int(y_down * frame_height)
    full_frame[y_start:y_end, x_start:x_end] = image
    processed_frames = np.concatenate((processed_frames, [full_frame]), axis=0)

    mask_frames[frame_index] = full_frame
    frame_index += 1
    pbar.update(1)
pbar.close()
print(f"processed_frames形状：{processed_frames.shape}")
avg_image = np.mean(np.mean(processed_frames, axis=0))
print(f"当前平均像素值：{avg_image}")
if avg_image > 10:
    print("当前平均值过大，请调大阈值threshold")
elif avg_image < 5:
    print("当前平均值过小，请调小阈值threshold")

#%%
# 应用掩码
# 初始化存储应用黑色掩码和绿色掩码后的图像数组
green_masked_images = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
black_masked_images = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
masked_images = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)

# 前window_size - 1帧添加空图像
for i in range(window_size - 1):
    green_masked_images[i] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    black_masked_images[i] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    masked_images[i] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# 应用掩码到剩余的帧
pbar = tqdm(total=num_frames - (window_size - 1), desc="Creating masked images")
for i in range(window_size - 1, num_frames):
    mask = mask_frames[i - window_size + 1]
    original_frame = image_stack[i].astype(np.uint8)
    original_frame_rgb = np.stack([original_frame] * 3, axis=-1)  # 转 RGB

    # 应用绿色掩码
    green_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    green_mask[mask > 10] = [0, 255, 0]  # 绿色掩码
    green_masked_image = np.where(mask[..., None] == 0, original_frame_rgb, green_mask)

    # 应用黑色掩码
    black_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    black_masked_image = np.where(mask[..., None] == 0, original_frame_rgb, black_mask)

    # 存储结果
    green_masked_images[i] = green_masked_image
    black_masked_images[i] = black_masked_image
    masked_images[i] = np.stack((processed_frames[i - (window_size - 1)],) * 3, axis=-1)  # 存储processed_frames转为三通道

    pbar.update(1)
pbar.close()
print('masked_images shape', black_masked_images.shape)   #   掩码图像
print('masked_images shape', green_masked_images.shape)
print('masked_images shape', masked_images.shape)
#%%

output_folder = "demo1"
os.makedirs(output_folder, exist_ok=True)
green_output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_{threshold}_{select_frame}_G.png")
black_output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_{threshold}_{select_frame}_B.png")
cv2.imwrite(green_output_path, green_masked_images[select_frame])
cv2.imwrite(black_output_path, black_masked_images[select_frame])
print(f"已保存第{select_frame}帧的绿色掩码图像到 {green_output_path}")
print(f"已保存第{select_frame}帧的黑色掩码图像到 {black_output_path}")

#%%
# 循环展示 Q下一帧 E上一帧 Esc退出
i = 0
while True:
    DVAbase.plot_images(
        image_stack[i % len(image_stack)],masked_images[i % len(masked_images)],
        black_masked_images[i % len(black_masked_images)],
        green_masked_images[i % len(green_masked_images)]
    )
    key = DVAbase.get_key()
    if key == 'q':
        i += 1
    elif key == 'e':
        i -= 1
    elif key == 'esc':
        break
