
from pathlib import Path
import subprocess
import sys
import uuid

import imageio_ffmpeg
import pandas as pd
import streamlit as st


# 1. 基本设置
st.set_page_config(
    page_title="深蹲 AI 动作分析",
    page_icon="🏋️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

st.title("深蹲 AI 动作分析")
st.caption("无标记点人体姿态识别 · 关节角度可视化")

st.info(
    "这是个人开发原型。当前角度属于二维视频估计，"
    "不能直接替代经过验证的临床测量。"
)

st.markdown(
    """
    ### 使用方法
    1. 上传一段 MP4 或 MOV 深蹲视频。
    2. 点击开始分析。
    3. 查看处理后的视频、角度曲线和数据表。
    """
)

uploaded_video = st.file_uploader(
    "选择深蹲视频（MP4 或 MOV）",
    type=["mp4", "mov"],
)

# 2. 所有分析步骤都放在按钮内部执行
if st.button(
    "开始分析",
    type="primary",
    disabled=uploaded_video is None,
):

    # 分析脚本固定读取 squat.mp4，因此将上传文件统一准备为这个路径。
    # MP4 直接保存；MOV 先转码成 H.264 MP4，兼容 iPhone 常见的 MOV/HEVC 视频。
    input_video = BASE_DIR / "squat.mp4"
    uploaded_suffix = Path(uploaded_video.name).suffix.lower()
    uploaded_bytes = uploaded_video.getvalue()

    if uploaded_suffix not in {".mp4", ".mov"}:
        st.error("目前只支持 MP4 或 MOV 视频，请重新选择文件。")
        st.stop()

    if uploaded_suffix == ".mov":
        source_video = BASE_DIR / f"uploaded_source_{uuid.uuid4().hex}.mov"
        st.info("已收到 MOV 视频，正在转换为兼容性更好的 MP4，请稍候……")
        try:
            source_video.write_bytes(uploaded_bytes)
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            with st.spinner("正在转换 MOV 视频……"):
                input_conversion = subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-loglevel", "error",
                        "-i", str(source_video),
                        "-map", "0:v:0",
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "18",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-an",
                        str(input_video),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
        except Exception as error:
            st.error(f"MOV 转换失败：{error}")
            st.stop()
        finally:
            source_video.unlink(missing_ok=True)

        if input_conversion.returncode != 0:
            st.error("MOV 转换失败，云端无法解码该视频。请尝试把手机视频转换为 H.264 MP4 后再上传。")
            if input_conversion.stderr.strip():
                st.code(input_conversion.stderr[-3000:])
            st.stop()
        if not input_video.exists() or input_video.stat().st_size == 0:
            st.error("MOV 转换没有生成有效的 MP4 文件，请重新上传视频。")
            st.stop()
    else:
        try:
            input_video.write_bytes(uploaded_bytes)
        except OSError as error:
            st.error(f"无法保存上传的视频：{error}")
            st.stop()

    # 3. 依次运行已有的分析程序
    scripts = [
        "analyze_video.py",
        "plot_angles.py",
        "analyze_phases.py",
        "make_report.py",
    ]
    logs = []
    failed = False
    progress = st.progress(0)
    status = st.empty()

    for index, script_name in enumerate(scripts, start=1):
        script_path = BASE_DIR / script_name

        if not script_path.exists():
            st.error(f"找不到程序文件：{script_name}")
            failed = True
            break

        status.info(
            f"正在执行第 {index}/{len(scripts)} 步："
            f"{script_name}"
        )

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as error:
            st.error(f"运行 {script_name} 时出错：{error}")
            failed = True
            break

        logs.append(
            (
                script_name,
                result.stdout,
                result.stderr,
            )
        )

        if result.returncode != 0:
            st.error(
                f"{script_name} 运行失败，请查看下方日志。"
            )
            failed = True
            break

        progress.progress(index / len(scripts))

    with st.expander("查看程序运行日志"):
        for name, stdout_text, stderr_text in logs:
            st.markdown(f"**{name}**")

            st.code(
                stdout_text or "程序没有标准输出"
            )

            if stderr_text.strip():
                st.text("警告或错误输出：")
                st.code(stderr_text[-3000:])

    if failed:
        status.error("分析没有全部完成。")
        st.stop()

    # 4. 检查 AI 生成的文件
    video_out = BASE_DIR / "squat_analyzed.mp4"
    phase_csv = BASE_DIR / "squat_phases.csv"
    angle_curve = BASE_DIR / "squat_angle_curve.png"
    report_html = BASE_DIR / "squat_report.html"

    required_files = [
        video_out,
        phase_csv,
        angle_curve,
        report_html,
    ]

    missing = [
        path.name
        for path in required_files
        if not path.exists() or path.stat().st_size == 0
    ]

    if missing:
        st.error(
            "分析结果不完整，缺少文件："
            + "、".join(missing)
        )
        st.stop()

    # 5. 转换成适合浏览器播放的视频
    browser_video_out = (
        BASE_DIR / "squat_analyzed_browser.mp4"
    )

    status.info("正在转换视频格式，请稍等……")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        conversion = subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel", "error",
                "-i", str(video_out),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(browser_video_out),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as error:
        st.error(f"启动视频转换程序失败：{error}")
        st.stop()

    if conversion.returncode != 0:
        st.error("视频格式转换失败。")
        st.code(conversion.stderr[-3000:])
        st.stop()

    if (
        not browser_video_out.exists()
        or browser_video_out.stat().st_size == 0
    ):
        st.error("没有生成有效的浏览器播放视频。")
        st.stop()

    status.success("分析完成！")

    # 6. 读取角度数据
    data = pd.read_csv(phase_csv)

    if data.empty:
        st.warning("没有可显示的角度记录。")
        st.stop()

    peak = data.loc[
        data["knee_flexion_smoothed_deg"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "分析侧",
        str(peak["side"]),
    )

    col2.metric(
        "最大膝屈曲二维估计值",
        f'{peak["knee_flexion_smoothed_deg"]:.1f}°',
    )

    col3.metric(
        "对应时间",
        f'{peak["time_seconds"]:.2f} 秒',
    )

    # 7. 显示视频
    st.subheader("处理后的视频")

    st.video(browser_video_out.read_bytes())

    # 8. 显示角度曲线
    st.subheader("深蹲关节角度曲线")

    st.image(
        str(angle_curve),
        caption="膝关节角度变化与初步动作阶段识别",
        use_container_width=True,
    )

    # 9. 显示数据表
    st.subheader("角度数据表")

    st.dataframe(
        data.head(100),
        use_container_width=True,
    )

    # 10. 提供下载
    st.download_button(
        "下载完整角度数据 CSV",
        data=phase_csv.read_bytes(),
        file_name="squat_phases.csv",
        mime="text/csv",
    )

    st.download_button(
        "下载分析后的视频",
        data=browser_video_out.read_bytes(),
        file_name="squat_analyzed.mp4",
        mime="video/mp4",
    )

    st.success("结果已生成，可以查看或下载。")

    st.caption(
        "当前角度和动作阶段均为初步二维估计，"
        "不能替代经过验证的临床测量。"
        "正式用于康复评估前，需要进一步验证测量准确性。"
    )