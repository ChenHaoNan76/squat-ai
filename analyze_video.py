
from pathlib import Path
import math
import csv
import cv2
import mediapipe as mp


# ========= 1. 文件路径 =========
folder = Path(__file__).resolve().parent

video_path = folder / "squat.mp4"
model_path = folder / "pose_landmarker_lite.task"
output_path = folder / "squat_analyzed.mp4"
csv_path = folder / "squat_angles.csv"
if not video_path.exists():
    raise FileNotFoundError(
        "没有找到 squat.mp4，请把视频放进 squat_ai 文件夹。"
    )

if not model_path.exists():
    raise FileNotFoundError(
        "没有找到 pose_landmarker_lite.task，请检查模型文件。"
    )


# ========= 2. 打开视频 =========
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(
        "无法打开 squat.mp4，请确认视频能正常播放且格式受支持。"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not math.isfinite(fps) or fps <= 0 or fps > 240:
    fps = 30.0

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("无法读取视频尺寸。")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        "无法创建输出视频，请检查文件夹权限或视频编码支持。"
    )


# ========= 3. 人体骨架连接关系 =========
connections = [
    # 右侧手臂
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),

    # 右侧躯干
    (12, 24),

    # 右侧腿部和足部
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
]
# MediaPipe 关键点编号
sides = {
    "LEFT": {
    "shoulder": 11,
    "hip": 23,
    "knee": 25,
    "ankle": 27,
    "foot": 31
    }, 
   "RIGHT": {
    "shoulder": 12,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
    "foot": 32
    },
}


# ========= 4. 计算二维关节屈曲角 =========
def flexion_angle(point_a, vertex, point_c):
    ax, ay = point_a[0], point_a[1]
    bx, by = vertex[0], vertex[1]
    cx, cy = point_c[0], point_c[1]

    vector_a = (ax - bx, ay - by)
    vector_c = (cx - bx, cy - by)

    length_a = math.hypot(*vector_a)
    length_c = math.hypot(*vector_c)

    if length_a == 0 or length_c == 0:
        return None

    dot = (
        vector_a[0] * vector_c[0]
        + vector_a[1] * vector_c[1]
    )

    cosine = dot / (length_a * length_c)
    cosine = max(-1.0, min(1.0, cosine))

    included_angle = math.degrees(math.acos(cosine))

    # 直腿时内夹角约为180度，屈曲估计值约为0度
    return 180.0 - included_angle


# ========= 5. 初始化视频模式的 AI =========
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

frame_index = 0
detected_frames = 0
angle_rows = []
try:
    with PoseLandmarker.create_from_options(options) as landmarker:

        # ========= 6. 逐帧处理视频 =========
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp_ms = int(round(frame_index * 1000.0 / fps))

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]
                points = []

                # 将AI输出的归一化坐标转换成像素坐标
                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))

                    visibility = getattr(
                        landmark, "visibility", 1.0
                    )

                    if visibility is None:
                        visibility = 1.0

                    points.append(
                        (x, y, float(visibility))
                    )

                # 绘制骨架
                for start, end in connections:
                    x1, y1, v1 = points[start]
                    x2, y2, v2 = points[end]

                    if v1 > 0.3 and v2 > 0.3:
                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA
                        )

                # 只绘制右侧的关键点
                right_point_ids = [
                    12, 14, 16, 18, 20, 22,
                    24, 26, 28, 30, 32
                ]

                for point_index in right_point_ids:
                    x, y, visibility = points[point_index]

                    if visibility > 0.3:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1,
                            cv2.LINE_AA
                        )

                # 选择关键点可见度较高的一侧
                def side_score(side):
                    ids = sides[side]
                    names = ("shoulder", "hip", "knee", "ankle")

                    return sum(
                        points[ids[name]][2]
                        for name in names
                    ) / len(names)

                best_side = "RIGHT"
                ids = sides[best_side]

                required_points = (
                    ids["shoulder"],
                    ids["hip"],
                    ids["knee"],
                    ids["ankle"]
                )

                if all(
                    points[index][2] > 0.3
                    for index in required_points
                ):
                    shoulder = points[ids["shoulder"]]
                    hip = points[ids["hip"]]
                    knee = points[ids["knee"]]
                    ankle = points[ids["ankle"]]

                    hip_flexion = flexion_angle(
                        shoulder, hip, knee
                    )

                    knee_flexion = flexion_angle(
                        hip, knee, ankle
                    )
                    # 计算踝背屈二维估计值
                    foot = points[ids["foot"]]
                    ankle_dorsiflexion = None

                    if (
                        knee[2] > 0.3
                        and ankle[2] > 0.3
                        and foot[2] > 0.3
                    ):
                        ankle_dorsiflexion = (
                            flexion_angle(knee, ankle, foot) - 90.0
                        )
                        # 在对应关节旁边显示角度
                    def draw_joint_label(name, value, point, color):
                        if value is None:
                            return

                        x = max(5, min(point[0] + 8, width - 190))
                        y = max(25, min(point[1] - 10, height - 5))

                        cv2.putText(
                            frame,
                            f"{name}: {value:.1f} deg",
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA
                        )


                    draw_joint_label(
                        "HIP", hip_flexion, hip, (0, 255, 255)
                    )

                    draw_joint_label(
                        "KNEE", knee_flexion, knee, (0, 255, 255)
                    )

                    draw_joint_label(
                        "ANKLE DF",
                        ankle_dorsiflexion,
                        ankle,
                        (255, 255, 0)
                    )
                    if hip_flexion is not None and knee_flexion is not None:
                        angle_rows.append({
                            "frame": frame_index,
                            "time_seconds": round(timestamp_ms / 1000.0, 3),
                            "side": best_side,
                            "hip_flexion_deg": round(hip_flexion, 2),
                            "knee_flexion_deg": round(knee_flexion, 2),
                            "ankle_dorsiflexion_deg": (
                                round(ankle_dorsiflexion, 2)
                                if ankle_dorsiflexion is not None
                                else ""
                            ),
                        })
                        
                    cv2.rectangle(
                        frame,
                        (5, 5),
                        (min(width - 5, 410), 80),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"{best_side} HIP FLEX: "
                        f"{hip_flexion:.1f} deg",
                        (12, 33),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        frame,
                        f"{best_side} KNEE FLEX: "
                        f"{knee_flexion:.1f} deg",
                        (12, 63),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                detected_frames += 1

            else:
                cv2.putText(
                    frame,
                    "Pose not detected",
                    (12, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # 把处理后的画面写入新视频
            writer.write(frame)

            frame_index += 1

            # 每处理约一秒的视频，输出一次进度
            if frame_index % max(1, int(fps)) == 0:
                print(
                    f"已处理 {frame_index} 帧，"
                    f"已识别 {detected_frames} 帧"
                )

finally:
    cap.release()
    writer.release()


# ========= 7. 输出结果 =========
if frame_index == 0:
    raise RuntimeError("视频没有读取到任何有效画面。")
# 保存每一帧的关节角度
with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
    fieldnames = [
        "frame",
        "time_seconds",
        "side",
        "hip_flexion_deg",
        "knee_flexion_deg",
        "ankle_dorsiflexion_deg",
    ]

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )
    csv_writer.writeheader()
    csv_writer.writerows(angle_rows)

print("角度数据已保存：", csv_path)

# 计算记录到的最大膝屈曲二维估计值
if angle_rows:
    max_knee_row = max(
        angle_rows,
        key=lambda row: row["knee_flexion_deg"]
    )

    print(
        f"最大膝屈曲二维估计值："
        f"{max_knee_row['knee_flexion_deg']:.1f}°"
    )

    print(
        f"出现时间：{max_knee_row['time_seconds']:.2f} 秒"
    )

    print(
        f"对应识别侧：{max_knee_row['side']}"
    )
else:
    print("没有获得有效的髋膝角度记录，请检查视频中的人体是否清晰。")
print("\n视频分析完成！")
print("总帧数：", frame_index)
print("检测到人体的帧数：", detected_frames)
print("输出视频：", output_path)
print("注意：这是二维角度估计，不等同于临床三维测量。")