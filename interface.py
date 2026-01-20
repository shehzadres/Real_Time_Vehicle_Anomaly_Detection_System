import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
import cv2
from FYP import initialize_pipeline, process_frame

st.set_page_config(page_title="Campus Traffic Anomaly Dashboard", layout="wide")
st.title("🚦 Campus Traffic Anomaly Detection Dashboard")

# --------------------------------------------------
# SIDEBAR: USER CONFIGURATION
# --------------------------------------------------
st.sidebar.header("⚙️ Detection Parameters")

fps = st.sidebar.number_input("Video FPS", min_value=5, max_value=60, value=25)

speed_limit = st.sidebar.slider("Speed Limit (km/h)", 10, 120, 30)
speed_window = st.sidebar.slider("Speed Smoothing Window (frames)", 3, 30, 10)
speeding_frames = st.sidebar.slider("Consecutive Speeding Frames", 1, 20, 6)

stall_seconds = st.sidebar.slider("Stall Duration (seconds)", 1, 10, 3)
stall_speed_thresh = st.sidebar.slider("Stall Speed Threshold (km/h)", 0.0, 2.0, 0.5)

road_length = st.sidebar.slider("Road Length (meters)", 10.0, 100.0, 30.0)
road_width = st.sidebar.slider("Road Width (meters)", 5.0, 30.0, 12.0)

lane_divisions = st.sidebar.slider("Number of Lanes", 1, 6, 3)
lane_violation_seconds = st.sidebar.slider("Lane Violation Duration (seconds)", 0.2, 2.0, 0.5)

st.sidebar.subheader("📐 Homography Source Points (pixels)")
st.sidebar.caption("Top-left, Top-right, Bottom-right, Bottom-left")

src_points = []
for i, label in enumerate(["TL", "TR", "BR", "BL"]):
    col1, col2 = st.sidebar.columns(2)
    x = col1.number_input(f"{label} X", value=100 + i*50)
    y = col2.number_input(f"{label} Y", value=100 + i*50)
    src_points.append([x, y])

# Lane directions (simple toggle)
st.sidebar.subheader("↕️ Lane Directions")
lane_direction = {}
for lane in range(lane_divisions):
    lane_direction[lane] = st.sidebar.selectbox(
        f"Lane {lane} direction",
        options={+1: "Forward", -1: "Reverse"},
        format_func=lambda x: "Forward" if x == 1 else "Reverse"
    )

# --------------------------------------------------
# UPLOAD VIDEO
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload a campus video (MP4)", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    st.info("Click ▶️ Start Processing to detect anomalies")

    if st.button("▶️ Start Processing"):
        with st.spinner("Processing video and generating annotated output..."):
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            output_video_path = output_dir / "RESULT_OUTPUT.mp4"
            anomaly_csv_path = output_dir / "anomalies_log.csv"

            # --------------------------------------------------
            # BUILD CONFIG FROM USER INPUT
            # --------------------------------------------------
            pipeline_config = {
                "fps": fps,
                "vehicle_model_path": "yolo11s.pt",
                "vehicle_classes": [2, 3, 5, 7],

                "road_length_m": road_length,
                "road_width_m": road_width,
                "lane_divisions": lane_divisions,

                "homography_src": src_points,

                "speed_limit_kmh": speed_limit,
                "speed_smoothing_window": speed_window,
                "speeding_consec_frames": speeding_frames,

                "stall_speed_thresh": stall_speed_thresh,
                "stall_seconds": stall_seconds,

                "lane_violation_seconds": lane_violation_seconds,

                "lane_direction": lane_direction
            }

            # Initialize pipeline
            state = initialize_pipeline(pipeline_config)

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

            anomalies_list = []
            st_frame = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, anomalies, state = process_frame(frame, state)
                anomalies_list.extend(anomalies)

                out.write(annotated_frame)
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

            cap.release()
            out.release()

            # Save anomalies
            df_anomalies = pd.DataFrame(anomalies_list)
            df_anomalies.to_csv(anomaly_csv_path, index=False)

            st.success("✅ Video processing complete!")

            st.subheader("Annotated Video")
            st.video(str(output_video_path))

            st.subheader("Detected Anomalies")
            st.dataframe(df_anomalies)

            st.subheader("Anomaly Statistics")
            st.write(f"Total anomalies: {len(df_anomalies)}")
            st.write(f"Speeding: {df_anomalies['event'].str.contains('SPEEDING', na=False).sum()}")
            st.write(f"Stalled: {df_anomalies['event'].str.contains('STALLED', na=False).sum()}")
            st.write(f"Wrong-way: {df_anomalies['event'].str.contains('WRONG WAY', na=False).sum()}")
            st.write(f"Lane violations: {df_anomalies['event'].str.contains('LANE VIOL', na=False).sum()}")

            st.download_button(
                "📥 Download Anomalies CSV",
                df_anomalies.to_csv(index=False).encode("utf-8"),
                file_name="anomalies_log.csv",
                mime="text/csv"
            )
