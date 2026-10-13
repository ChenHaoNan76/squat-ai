
from pathlib import Path
import math

import cv2
import mediapipe as mp


# 1. 找到项目文件夹和相关文件
folder = Path(__file__).resolve().parent

image_path = folder / "person.jpg"
model_path = folder / "pose_landmarker_lite.task"
output_path = folder / "person_angles.jpg"


# 2. 检查文件是否存在
if not image_path.exists():
    raise FileNotFoundError("没有找到 person.jpg，请检查项目文件夹。")

if not model_path.exists():
    raise FileNotFoundError("没有找到 pose_landmarker_lite.task，请检查模型文件。")


# 3. 读取图片
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("图片读取失败，请检查图片格式。")

height, width = image.shape[:2]

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# 4. 使用 MediaPipe 识别人体关键点
with mp.tasks.vision.PoseLandmarker.create_from_model_path(
    str(model_path)
) as landmarker:
    result = landmarker.detect(mp_image)

if not result.pose_landmarks:
    raise RuntimeError(
        "没有识别到人体，请换一张清晰的全身照片。"
    )

landmarks = result.pose_landmarks[0]


# 5. 把归一化坐标转换为图片像素坐标
points = []

for landmark in landmarks:
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    visibility = getattr(landmark, "visibility", 1.0)
    if visibility is None:
        visibility = 1.0

    points.append((x, y, float(visibility)))


# 6. 定义人体骨架的连接关系
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


# 7. 选择可见度较高的一侧
sides = {
    "左侧": {
        "shoulder": 11,
        "hip": 23,
        "knee": 25,
        "ankle": 27
    },
    "右侧": {
        "shoulder": 12,
        "hip": 24,
        "knee": 26,
        "ankle": 28
    }
}

def side_score(side):
    ids = sides[side]
    return sum(points[ids[key]][2] for key in ids) / 4


best_side = max(sides, key=side_score)
ids = sides[best_side]

if any(points[ids[key]][2] < 0.25 for key in ids):
    raise RuntimeError(
        "肩、髋、膝或踝关键点不够清晰，请换一张全身照片。"
    )


# 8. 计算三点形成的夹角
def joint_angle(point_a, vertex, point_c):
    ax, ay = point_a[0], point_a[1]
    bx, by = vertex[0], vertex[1]
    cx, cy = point_c[0], point_c[1]

    vector_a = (ax - bx, ay - by)
    vector_c = (cx - bx, cy - by)

    dot = (
        vector_a[0] * vector_c[0]
        + vector_a[1] * vector_c[1]
    )

    length_a = math.hypot(*vector_a)
    length_c = math.hypot(*vector_c)

    if length_a == 0 or length_c == 0:
        raise ValueError("关键点重合，无法计算角度。")

    cosine = dot / (length_a * length_c)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


# 9. 计算髋、膝的二维屈曲角估计值
shoulder = points[ids["shoulder"]]
hip = points[ids["hip"]]
knee = points[ids["knee"]]
ankle = points[ids["ankle"]]

hip_included = joint_angle(shoulder, hip, knee)
knee_included = joint_angle(hip, knee, ankle)

hip_flexion = 180.0 - hip_included
knee_flexion = 180.0 - knee_included


# 10. 在照片上画出人体骨架
for start, end in connections:
    x1, y1, v1 = points[start]
    x2, y2, v2 = points[end]

    if v1 > 0.3 and v2 > 0.3:
        cv2.line(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 3, cv2.LINE_AA
        )

for x, y, visibility in points:
    if visibility > 0.3:
        cv2.circle(
            image, (x, y), 5,
            (0, 0, 255), -1, cv2.LINE_AA
        )


# 11. 把角度数字画在髋、膝附近
def draw_label(text, anchor):
    x, y = anchor[0], anchor[1]
    text_width = 230

    if width > text_width + 20:
        if x + text_width < width:
            x += 10
        else:
            x -= text_width + 10

        x = max(5, min(x, width - text_width - 5))
    else:
        x = 5

    y = max(25, min(y - 10, height - 10))

    cv2.putText(
        image, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 255), 2, cv2.LINE_AA
    )


draw_label(
    f"HIP FLEX: {hip_flexion:.1f} deg",
    hip
)

draw_label(
    f"KNEE FLEX: {knee_flexion:.1f} deg",
    knee
)


# 12. 保存结果并显示测量数据
success = cv2.imwrite(str(output_path), image)

if not success:
    raise RuntimeError("结果图片保存失败。")

print("人体识别成功！")
print("显示侧：", best_side)
print(f"髋关节二维屈曲角估计：{hip_flexion:.1f}°")
print(f"膝关节二维屈曲角估计：{knee_flexion:.1f}°")
print("结果图片：", output_path)
print("注意：这是二维图片估计，不等同于临床三维测量。")