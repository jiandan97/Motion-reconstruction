# 这个版本可以自由调整单帧图片的像素数

import cv2
import numpy as np
import os
import glob
from PIL import Image  # 使用 Pillow 读取和保存帧

# === 参数配置 ===
image_folder = "frames"
image_paths = sorted(glob.glob(os.path.join(image_folder, 'frame_*.png')))

print("Looking for frames in:", image_folder)
print("Found these files:", image_paths[:5], "… total:", len(image_paths))

# —— 用 PIL 读取第一帧 —— 
pil0 = Image.open(image_paths[0]).convert("RGB")
frame0 = cv2.cvtColor(np.array(pil0), cv2.COLOR_RGB2BGR)
print("PIL→cv2 read returned:", frame0 is not None)

prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

# === HSV for dense flow 可视化准备 ===
hsv = np.zeros_like(frame0)
hsv[..., 1] = 255

# 输出目录
save_path = os.path.join(image_folder, "optical_flow_frames")
os.makedirs(save_path, exist_ok=True)

mode = "save"  # or "show"

# === 光流处理循环 ===
for i, path in enumerate(image_paths[1:], start=1):
    # 用 PIL 读取后转 cv2
    pil = Image.open(path).convert("RGB")
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Farneback 稠密光流 ===
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.2, levels=3, winsize=30,
        iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # === HSV 可视化（可选） ===
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # === 在原图上画稀疏箭头 ===
    step = 8
    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            fx, fy = flow[y, x]
            end_point = (int(x + fx), int(y + fy))
            cv2.arrowedLine(
                frame, (x, y), end_point,
                color=(0, 255, 0), thickness=1, tipLength=0.3
            )

    # === 保存或展示 ===
    if mode == "save":
        out_file = os.path.join(save_path, f'dense8_flow_quiver_{i:04d}.png')
        # BGR -> RGB，然后用 Pillow 保存
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            img.save(out_file)
            print(f"[Saved] {out_file}")
        except Exception as e:
            print(f"[Error] 无法写入 {out_file}: {e}")
    else:
        cv2.imshow("Optical Flow", frame)
        if cv2.waitKey(1000) & 0xFF == 27:
            break

    prev_gray = gray

if mode == "show":
    cv2.destroyAllWindows()
