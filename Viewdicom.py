import matplotlib.pyplot as plt
import pydicom
import os
import cv2
import numpy as np

file_path = 'IMG-0024-00001.dcm'
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

cv2.imwrite('IMG24-180.png', frames[180])
frames_index = 0

while True:
    cv2.imshow('Origin', frames[frames_index % len(frames)])

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        frames_index = (frames_index + 1) % len(frames)
    elif key == ord('e'):
        frames_index = (frames_index - 1) % len(frames)
    elif key == 27:  # Esc key
        break

cv2.destroyAllWindows()