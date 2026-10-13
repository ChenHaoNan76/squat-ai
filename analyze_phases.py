
from pathlib import Path
import csv
from collections import Counter
import statistics

import matplotlib.pyplot as plt


# 1. 文件路径
folder = Path(__file__).resolve().parent
csv_path = folder / "squat_angles.csv"
output_csv = folder / "squat_phases.csv"
output_png = folder / "squat_phase_curve.png"

if not csv_path.exists():
    raise FileNotFoundError(
        "没有找到 squat_angles.csv，请先运行 analyze_video.py。"
    )


# 2. 读取角度数据
records = []

with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        try:
            records.append({
                "frame": int(float(row["frame"])),
                "time_seconds": float(row["time_seconds"]),
                "side": row["side"],
                "hip_flexion_deg": float(row["hip_flexion_deg"]),
                "knee_flexion_deg": float(row["knee_flexion_deg"]),
                "ankle_dorsiflexion_deg": (
                    float(row["ankle_dorsiflexion_deg"])
                    if (row.get("ankle_dorsiflexion_deg") or "").strip()
                    else None
                ),
            })
        except (ValueError, TypeError, KeyError):
            continue

if len(records) < 5:
    raise ValueError(
        "有效角度数据太少，请检查 squat_angles.csv。"
    )


# 3. 固定使用记录最多的一侧，避免左右侧数据混在一起
side_counts = Counter(row["side"] for row in records)
chosen_side = side_counts.most_common(1)[0][0]

rows = [
    row for row in records
    if row["side"] == chosen_side
]
rows.sort(key=lambda row: row["time_seconds"])

if len(rows) < 5:
    raise ValueError("该侧有效数据太少，无法分析动作阶段。")

times = [row["time_seconds"] for row in rows]
knee_raw = [row["knee_flexion_deg"] for row in rows]


# 4. 使用前后相邻数据的平均值，减少曲线小幅抖动
window = 5
half_window = window // 2

knee_smooth = []

for i in range(len(knee_raw)):
    start = max(0, i - half_window)
    end = min(len(knee_raw), i + half_window + 1)

    average = sum(knee_raw[start:end]) / (end - start)
    knee_smooth.append(average)


# 5. 找到膝屈曲估计值最大的帧
peak_index = max(
    range(len(knee_smooth)),
    key=lambda i: knee_smooth[i]
)

peak_angle = knee_smooth[peak_index]
peak_time = times[peak_index]

# 用视频两端的角度估计起始/结束的站立水平
edge_count = max(1, len(knee_smooth) // 10)

edge_values = (
    knee_smooth[:edge_count]
    + knee_smooth[-edge_count:]
)

baseline = statistics.median(edge_values)

if peak_angle - baseline < 5:
    raise ValueError(
        "视频中的膝关节角度变化较小。请检查视频是否包含完整深蹲，"
        "以及开始和结束时是否接近站立姿势。"
    )

# 超过起始/结束水平的15%，暂时视为进入明显动作
stand_threshold = baseline + 0.15 * (peak_angle - baseline)

# 最高角度附近的一小段，标记为最低点附近
bottom_tolerance = max(
    3.0,
    0.05 * (peak_angle - baseline)
)

phases = []

for i, angle in enumerate(knee_smooth):
    if angle >= peak_angle - bottom_tolerance:
        phase = "最低点附近"

    elif angle <= stand_threshold:
        phase = "站立/接近站立"

    elif i < peak_index:
        phase = "下蹲阶段"

    else:
        phase = "起身阶段"

    phases.append(phase)


# 6. 保存带有动作阶段的数据表
with output_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "frame",
        "time_seconds",
        "side",
        "hip_flexion_deg",
        "knee_flexion_deg",
        "ankle_dorsiflexion_deg",
        "knee_flexion_smoothed_deg",
        "phase",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for row, smooth_value, phase in zip(
        rows, knee_smooth, phases
    ):
        writer.writerow({
            **row,
            "knee_flexion_smoothed_deg": round(smooth_value, 2),
            "phase": phase,
        })


# 7. 绘制曲线，并用背景色区分动作阶段
plt.figure(figsize=(12, 6))

phase_colors = {
    "站立/接近站立": "#E5E7EB",
    "下蹲阶段": "#BFDBFE",
    "最低点附近": "#BBF7D0",
    "起身阶段": "#FED7AA",
}

# 将连续相同阶段的帧画成一段背景
segment_start = 0

for i in range(1, len(phases) + 1):
    if i == len(phases) or phases[i] != phases[segment_start]:
        plt.axvspan(
            times[segment_start],
            times[i - 1],
            color=phase_colors[phases[segment_start]],
            alpha=0.35
        )
        segment_start = i

plt.plot(
    times,
    knee_raw,
    alpha=0.35,
    linewidth=1,
    label="Raw knee estimate"
)

plt.plot(
    times,
    knee_smooth,
    linewidth=2,
    label="Smoothed knee estimate"
)

plt.axvline(
    peak_time,
    linestyle="--",
    color="red",
    label="Estimated bottom"
)

plt.scatter(
    [peak_time],
    [peak_angle],
    color="red",
    zorder=5
)

plt.annotate(
    f"{peak_angle:.1f} deg at {peak_time:.2f} s",
    (peak_time, peak_angle),
    xytext=(10, 12),
    textcoords="offset points"
)

plt.title("Squat Phases and Knee Flexion Estimate")
plt.xlabel("Time (seconds)")
plt.ylabel("Estimated knee flexion (degrees)")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(output_png, dpi=160)
plt.close()


# 8. 输出结果
print("深蹲阶段分析完成！")
print("采用分析侧：", chosen_side)
print(f"估计最低点时间：{peak_time:.2f} 秒")
print(f"该时刻膝屈曲估计值：{peak_angle:.1f}°")
print("阶段数据表：", output_csv)
print("阶段曲线图：", output_png)
print("提示：阶段判断属于初步算法，不是经过验证的临床测量。")