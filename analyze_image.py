
from pathlib import Path

import cv2
import mediapipe as mp

# 找到当前项目文件夹
folder = Path(__file__).resolve().parent

# 图片、AI模型、输出图片的路径
image_path = folder / "person.jpg"
model_path = folder / "pose_landmarker_lite.task"
output_path = folder / "person_skeleton.jpg"

# 检查文件是否存在
if not image_path.exists():
    raise FileNotFoundError(
        "没有找到 person.jpg，请检查图片是否放在 squat_ai 文件夹里。"
    )

if not model_path.exists():
    raise FileNotFoundError(
        "没有找到 pose_landmarker_lite.task，请检查AI模型文件。"
    )

# 读取人体照片
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("图片无法读取，请确认图片格式正确。")

height, width = image.shape[:2]

# OpenCV读取的图片转换成AI需要的格式
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

# 人体骨架的连接关系
connections = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

# 加载AI模型并识别照片
with mp.tasks.vision.PoseLandmarker.create_from_model_path(
    str(model_path)
) as landmarker:

    result = landmarker.detect(mp_image)

# 检查有没有检测到人体
if not result.pose_landmarks:
    raise RuntimeError(
        "没有识别到人体，请换一张身体更清晰的全身照片。"
    )

# 取出第一个人体的33个关键点
landmarks = result.pose_landmarks[0]

points = []

for landmark in landmarks:
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    visibility = getattr(landmark, "visibility", 1.0)
    if visibility is None:
        visibility = 1.0

    points.append((x, y, visibility))

# 把关键点之间的骨架画出来
for start, end in connections:
    x1, y1, v1 = points[start]
    x2, y2, v2 = points[end]

    if v1 > 0.3 and v2 > 0.3:
        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

# 在关键点位置画圆点
for x, y, visibility in points:
    if visibility > 0.3:
        cv2.circle(
            image,
            (x, y),
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

# 保存识别结果
success = cv2.imwrite(str(output_path), image)

if not success:
    raise RuntimeError("结果图片保存失败。")

print("人体识别成功！")
print("识别到的关键点数量：", len(landmarks))
print("结果图片保存在：", output_path)