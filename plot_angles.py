from pathlib import Path
import csv
import matplotlib.pyplot as plt

# 1. 找到项目文件夹和角度数据表
folder = Path(__file__).resolve().parent
csv_path = folder / "squat_angles.csv"
output_path = folder / "squat_angle_curve.png"

if not csv_path.exists():
    raise FileNotFoundError(
        "没有找到 squat_angles.csv，请先运行 analyze_video.py。"
    )

# 2. 读取每一帧的角度数据
times = []
hip_angles = []
knee_angles = []
ankle_angles = []

with csv_path.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    reader = csv.DictReader(csv_file)

    required_columns = {
        "time_seconds",
        "hip_flexion_deg",
        "knee_flexion_deg",
        "ankle_dorsiflexion_deg",
    }

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError(
            "CSV表格缺少需要的关节角度列，请先运行 analyze_video.py。"
        )

    for row in reader:
        try:
            time_value = float(row["time_seconds"])
            hip_value = float(row["hip_flexion_deg"])
            knee_value = float(row["knee_flexion_deg"])
        except (ValueError, TypeError):
            continue

        # 踝关节有时会因为脚部遮挡而没有有效数据
        ankle_text = (
            row.get("ankle_dorsiflexion_deg") or ""
        ).strip()

        try:
            ankle_value = (
                float(ankle_text)
                if ankle_text
                else float("nan")
            )
        except (ValueError, TypeError):
            ankle_value = float("nan")

        times.append(time_value)
        hip_angles.append(hip_value)
        knee_angles.append(knee_value)
        ankle_angles.append(ankle_value)

if not times:
    raise ValueError("表格里没有可用于绘图的角度数据。")

# 3. 找到记录到的最大膝屈曲估计值
max_index = max(
    range(len(knee_angles)),
    key=lambda i: knee_angles[i]
)

max_knee = knee_angles[max_index]
max_time = times[max_index]

# 4. 绘制髋、膝关节角度变化曲线
plt.figure(figsize=(11, 6))

plt.plot(
    times,
    hip_angles,
    label="Hip flexion estimate",
    linewidth=2
)

plt.plot(
    times,
    knee_angles,
    label="Knee flexion estimate",
    linewidth=2
)

plt.plot(
    times,
    ankle_angles,
    label="Ankle dorsiflexion estimate",
    linewidth=2
)

# 标记膝关节角度估计值最大的时刻
plt.scatter(
    [max_time],
    [max_knee],
    s=65,
    zorder=3,
    label="Maximum recorded knee value"
)

plt.annotate(
    f"{max_knee:.1f} deg at {max_time:.2f} s",
    (max_time, max_knee),
    xytext=(10, 12),
    textcoords="offset points"
)

plt.title("Squat Joint Angle over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Estimated angle (degrees)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 5. 保存图片
plt.savefig(output_path, dpi=160)
plt.close()

print("角度曲线生成成功！")
print("数据点数量：", len(times))
print(f"最大膝屈曲二维估计值：{max_knee:.1f}°")
print(f"对应时间：{max_time:.2f} 秒")
print("曲线图片保存位置：", output_path)