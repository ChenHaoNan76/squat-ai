from pathlib import Path
import csv
from collections import Counter
from html import escape

# 找到项目文件夹
folder = Path(__file__).resolve().parent

csv_path = folder / "squat_phases.csv"
curve_path = folder / "squat_phase_curve.png"
report_path = folder / "squat_report.html"

# 检查数据文件
if not csv_path.exists():
    raise FileNotFoundError(
        "没有找到 squat_phases.csv，请先运行 analyze_phases.py。"
    )

# 读取深蹲分析数据
rows = []

with csv_path.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:
    reader = csv.DictReader(file)

    for row in reader:
        try:
            row["time_seconds"] = float(row["time_seconds"])
            row["hip_flexion_deg"] = float(row["hip_flexion_deg"])
            row["knee_flexion_smoothed_deg"] = float(
                row["knee_flexion_smoothed_deg"]
            )
        except (ValueError, TypeError, KeyError):
            continue
        ankle_text = (
            row.get("ankle_dorsiflexion_deg") or ""
        ).strip()

        try:
            row["ankle_dorsiflexion_deg"] = (
                float(ankle_text) if ankle_text else None
            )
        except ValueError:
            row["ankle_dorsiflexion_deg"] = None
        rows.append(row)

if not rows:
    raise ValueError("没有可用于生成报告的有效数据。")

# 找到记录到的最大膝屈曲估计值
peak = max(
    rows,
    key=lambda row: row["knee_flexion_smoothed_deg"]
)
peak_ankle = peak.get("ankle_dorsiflexion_deg")

if peak_ankle is not None:
    peak_ankle_text = f"{peak_ankle:.1f}°"
else:
    peak_ankle_text = "暂无有效数据"

duration = max(row["time_seconds"] for row in rows) - min(
    row["time_seconds"] for row in rows
)

sides = Counter(row["side"] for row in rows)
main_side = sides.most_common(1)[0][0]

phase_counts = Counter(row["phase"] for row in rows)

# 引用已生成的曲线图
if curve_path.exists():
    chart_html = """
    <img class="chart"
         src="squat_phase_curve.png"
         alt="深蹲膝关节角度变化曲线">
    """
else:
    chart_html = "<p>暂未找到角度曲线图片。</p>"

# 生成阶段汇总
phase_items = ""

for phase, count in phase_counts.items():
    phase_items += (
        "<tr><td>"
        + escape(str(phase))
        + "</td><td>"
        + str(count)
        + "</td></tr>"
    )

# 生成报告网页
html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>深蹲动作分析报告</title>

<style>
body {{
    font-family: "Microsoft YaHei", sans-serif;
    max-width: 900px;
    margin: 32px auto;
    padding: 0 20px;
    color: #253247;
    background: #f6f8fb;
    line-height: 1.7;
}}

h1, h2 {{ color: #183b56; }}

.subtitle {{ color: #617286; }}

.metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin: 24px 0;
}}

.card {{
    background: white;
    border: 1px solid #dce4ec;
    border-radius: 12px;
    padding: 18px;
}}

.label {{ color: #65758b; font-size: 14px; }}

.value {{
    font-size: 28px;
    font-weight: bold;
    margin-top: 4px;
}}

.chart {{
    width: 100%;
    height: auto;
    border-radius: 10px;
    background: white;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
}}

th, td {{
    padding: 10px;
    border: 1px solid #dce4ec;
    text-align: left;
}}

.notice {{
    margin-top: 24px;
    padding: 16px;
    background: #fff8e8;
    border-radius: 10px;
    color: #624b21;
}}

@media print {{
    body {{ background: white; }}
}}
</style>
</head>

<body>
<h1>深蹲动作分析报告</h1>
<p class="subtitle">人体姿态识别与二维关节角度估计</p>

<div class="metrics">
    <div class="card">
        <div class="label">分析侧</div>
        <div class="value">{escape(str(main_side))}</div>
    </div>

    <div class="card">
        <div class="label">记录时长</div>
        <div class="value">{duration:.2f} 秒</div>
    </div>

    <div class="card">
        <div class="label">最大膝屈曲估计值</div>
        <div class="value">
            {peak["knee_flexion_smoothed_deg"]:.1f}°
        </div>
    </div>
    <div class="card">
        <div class="label">踝关节二维角度估计（同一时刻）</div>
        <div class="value">{peak_ankle_text}</div>
    </div>
    
    <div class="card">
        <div class="label">对应时间</div>
        <div class="value">{peak["time_seconds"]:.2f} 秒</div>
    </div>
</div>

<h2>关节角度变化曲线</h2>
{chart_html}

<h2>动作阶段记录</h2>
<table>
    <thead>
        <tr><th>阶段</th><th>记录帧数</th></tr>
    </thead>
    <tbody>
        {phase_items}
    </tbody>
</table>

<div class="notice">
<strong>使用说明</strong><br>
本报告用于动作可视化和康复评估辅助讨论。
关节角度为二维视频估计值，可能受到拍摄方向、遮挡、
关键点定位误差和相机透视影响。
本报告不构成独立诊断，也不能替代经过验证的临床测量。
请由康复专业人员结合实际情况解读。
</div>

</body>
</html>
"""

report_path.write_text(
    html_content,
    encoding="utf-8"
)

print("深蹲分析报告生成成功！")
print("报告文件：", report_path)
print("请用浏览器打开 squat_report.html 查看。")
print("曲线图片需要与报告文件放在同一个文件夹。")