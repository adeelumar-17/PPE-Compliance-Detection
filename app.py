"""
PPE Compliance Detection — Streamlit Application
==================================================
Real-time construction-site PPE compliance monitoring using a fine-tuned YOLO11m model.
Supports image upload, video upload, and live webcam inference via WebRTC.
"""

import io
import tempfile
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from ppe_engine import (
    PPEDetector,
    PersonTracker,
    ViolationEventLog,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    analyze_frame,
    annotate_frame_with_tracking,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PPE Compliance Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e8eaed !important;
    }

    /* ── Hero header ── */
    .hero-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #161b22 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #f97316, #ef4444, #8b5cf6);
    }
    .hero-header h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f97316 0%, #ef4444 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #161b22, #1a1f2e);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .metric-card.total .metric-value { color: #58a6ff; }
    .metric-card.compliant .metric-value { color: #3fb950; }
    .metric-card.violation .metric-value { color: #f85149; }
    .metric-card.unknown .metric-value { color: #8b949e; }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-compliant {
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .badge-violation {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .badge-unknown {
        background: rgba(139, 148, 158, 0.15);
        color: #8b949e;
        border: 1px solid rgba(139, 148, 158, 0.3);
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e8eaed;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(249, 115, 22, 0.4);
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Violation event cards ── */
    .violation-event {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.08), rgba(248, 81, 73, 0.03));
        border: 1px solid rgba(248, 81, 73, 0.2);
        border-left: 3px solid #f85149;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .violation-event .event-title {
        font-weight: 600;
        color: #f85149;
        font-size: 0.9rem;
    }
    .violation-event .event-detail {
        color: #8b949e;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* ── Info panel ── */
    .info-panel {
        background: linear-gradient(135deg, #161b22, #1a1f2e);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .info-panel h4 {
        color: #58a6ff;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
    }
    .info-panel p {
        color: #8b949e;
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.6;
    }

    /* ── Active violation banner ── */
    .active-violation-banner {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.15), rgba(248, 81, 73, 0.05));
        border: 1px solid rgba(248, 81, 73, 0.4);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        animation: pulse-border 2s ease-in-out infinite;
    }
    @keyframes pulse-border {
        0%, 100% { border-color: rgba(248, 81, 73, 0.4); }
        50% { border-color: rgba(248, 81, 73, 0.8); }
    }
    .active-violation-banner .banner-icon {
        font-size: 1.5rem;
    }
    .active-violation-banner .banner-text {
        color: #f85149;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ── Person compliance table ── */
    .compliance-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin: 0.5rem 0;
    }
    .compliance-table th {
        background: #161b22;
        color: #8b949e;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 0.7rem 1rem;
        text-align: left;
    }
    .compliance-table td {
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #e8eaed;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
    }
    .compliance-table tr:nth-child(even) td {
        background: rgba(255, 255, 255, 0.02);
    }

    /* ── Mode selector enhancements ── */
    div[data-testid="stRadio"] label {
        font-weight: 500 !important;
    }

    /* ── Hide default streamlit elements ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Webcam status panel ── */
    .webcam-status {
        background: linear-gradient(135deg, #161b22, #1a1f2e);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"


@st.cache_resource(show_spinner=False)
def load_detector():
    """Load the YOLO model once and cache it."""
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: `{MODEL_PATH}`")
        st.stop()
    return PPEDetector(str(MODEL_PATH), device="cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🦺 PPE Detection")
    st.markdown("---")

    mode = st.radio(
        "**Inference Mode**",
        ["📷 Image", "🎬 Video", "📹 Real-Time Camera"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90, value=0.25, step=0.05,
        help="Minimum confidence score for a detection to be kept."
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="IoU threshold for Non-Maximum Suppression."
    )

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.markdown("""
    <div class="info-panel">
        <h4>YOLO11m — Fine-tuned</h4>
        <p>
            <strong>Architecture:</strong> YOLO11 Medium<br>
            <strong>Dataset:</strong> Roboflow construction-rineu<br>
            <strong>Classes:</strong> helmet, no-helmet, no-vest, person, vest<br>
            <strong>Input Size:</strong> 640×640
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Compliance Rules")
    st.markdown("""
    <div class="info-panel">
        <p>
            ✅ <strong>Compliant:</strong> Helmet + Vest<br>
            ❌ <strong>Violation:</strong> Missing helmet or vest<br>
            ❓ <strong>Unknown:</strong> PPE items not detected
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>🦺 PPE Compliance Detection</h1>
    <p>Real-time construction-site safety monitoring powered by YOLO11m — detecting helmets, vests, and compliance violations.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def render_metrics(summary):
    """Render metric cards from a summary dict."""
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card total">
            <p class="metric-value">{summary['total_persons']}</p>
            <p class="metric-label">Total Persons</p>
        </div>
        <div class="metric-card compliant">
            <p class="metric-value">{summary['compliant']}</p>
            <p class="metric-label">Compliant</p>
        </div>
        <div class="metric-card violation">
            <p class="metric-value">{summary['violations']}</p>
            <p class="metric-label">Violations</p>
        </div>
        <div class="metric-card unknown">
            <p class="metric-value">{summary['unknown']}</p>
            <p class="metric-label">Unknown</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_person_table(person_results):
    """Render a styled person compliance table."""
    if not person_results:
        return

    rows = ""
    for p in person_results:
        badge_class = {
            "COMPLIANT": "badge-compliant",
            "VIOLATION": "badge-violation",
        }.get(p["compliance"], "badge-unknown")

        helmet_icon = {"helmet": "🟢", "no-helmet": "🔴"}.get(p["helmet_status"], "⚪")
        vest_icon = {"vest": "🟢", "no-vest": "🔴"}.get(p["vest_status"], "⚪")

        rows += f"""
        <tr>
            <td>Person {p['id']}</td>
            <td>{p['confidence']:.0%}</td>
            <td>{helmet_icon} {p['helmet_status'].title()}</td>
            <td>{vest_icon} {p['vest_status'].title()}</td>
            <td><span class="badge {badge_class}">{p['compliance']}</span></td>
        </tr>
        """

    st.markdown(f"""
    <table class="compliance-table">
        <thead>
            <tr>
                <th>Person</th>
                <th>Confidence</th>
                <th>Helmet</th>
                <th>Vest</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


def render_violation_events(events):
    """Render violation event cards."""
    if not events:
        st.markdown("""
        <div class="info-panel">
            <p>✅ No violation events detected in this video.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    for event in events:
        st.markdown(f"""
        <div class="violation-event">
            <div class="event-title">🚨 Event #{event['event_id']} — {event['violation']}</div>
            <div class="event-detail">
                ⏱ {event['start_time']} → {event['end_time']}
                &nbsp;|&nbsp; Confidence: {event['confidence']:.0%}
                &nbsp;|&nbsp; Frames: {event['start_frame']}–{event['end_frame']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading YOLO11m model..."):
    detector = load_detector()


# ──────────────────────────────────────────────────────────────────────────────
# Mode: Image
# ──────────────────────────────────────────────────────────────────────────────

if mode == "📷 Image":
    st.markdown('<div class="section-header">📷 Image Inference</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="image_uploader",
    )

    if uploaded is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_bgr is None:
            st.error("Failed to decode the uploaded image.")
        else:
            with st.spinner("🔍 Running PPE detection..."):
                t0 = time.time()
                annotated, person_results, summary = analyze_frame(
                    detector, original_bgr,
                    conf=conf_threshold, iou=iou_threshold
                )
                inference_time = time.time() - t0

            # Side-by-side display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(
                    cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
            with col2:
                st.markdown("**Detection Results**")
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

            # Inference time
            st.caption(f"⚡ Inference completed in **{inference_time:.2f}s**")

            # Metrics
            st.markdown('<div class="section-header">📊 Compliance Summary</div>', unsafe_allow_html=True)
            render_metrics(summary)

            # Person table
            st.markdown('<div class="section-header">👷 Per-Person Results</div>', unsafe_allow_html=True)
            if person_results:
                render_person_table(person_results)
            else:
                st.markdown("""
                <div class="info-panel">
                    <p>No persons detected in this image. Try adjusting the confidence threshold.</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-panel">
            <h4>📤 Upload an Image</h4>
            <p>
                Upload a construction site image to analyze PPE compliance.
                Supported formats: JPG, JPEG, PNG, WEBP, BMP.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Mode: Video
# ──────────────────────────────────────────────────────────────────────────────

elif mode == "🎬 Video":
    st.markdown('<div class="section-header">🎬 Video Inference</div>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Failed to open the uploaded video.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.markdown(f"""
            <div class="info-panel">
                <h4>📹 Video Info</h4>
                <p>
                    <strong>Resolution:</strong> {width}×{height} &nbsp;|&nbsp;
                    <strong>FPS:</strong> {fps:.1f} &nbsp;|&nbsp;
                    <strong>Frames:</strong> {total_frames} &nbsp;|&nbsp;
                    <strong>Duration:</strong> {total_frames / fps:.1f}s
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶️ Process Video", type="primary", use_container_width=True):
                # Output temp file
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                tracker = PersonTracker()
                event_log = ViolationEventLog()

                progress_bar = st.progress(0, text="Processing frames...")
                frame_display = st.empty()
                stats_display = st.empty()

                frame_number = 0
                all_person_results = []
                t_start = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_number += 1

                    annotated, person_results, summary = annotate_frame_with_tracking(
                        frame, detector, tracker, event_log,
                        frame_number, fps,
                        conf=conf_threshold, iou=iou_threshold,
                        imgsz=DEFAULT_IMGSZ
                    )

                    writer.write(annotated)
                    all_person_results.append(summary)

                    # Update UI every 5 frames to avoid Streamlit overhead
                    if frame_number % 5 == 0 or frame_number == total_frames:
                        progress = min(frame_number / max(total_frames, 1), 1.0)
                        elapsed = time.time() - t_start
                        proc_fps = frame_number / elapsed if elapsed > 0 else 0

                        progress_bar.progress(
                            progress,
                            text=f"Frame {frame_number}/{total_frames} — {proc_fps:.1f} FPS"
                        )

                        # Show current frame (downsampled for speed)
                        preview = cv2.resize(annotated, (min(width, 720), min(height, 480)))
                        frame_display.image(
                            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_number}",
                            use_container_width=True,
                        )

                # Finalize
                event_log.close_if_active(frame_number, fps)
                cap.release()
                writer.release()

                elapsed = time.time() - t_start
                progress_bar.progress(1.0, text=f"✅ Done — {frame_number} frames in {elapsed:.1f}s ({frame_number / elapsed:.1f} FPS)")

                # Aggregate metrics
                total_persons_all = sum(s["total_persons"] for s in all_person_results)
                total_compliant_all = sum(s["compliant"] for s in all_person_results)
                total_violations_all = sum(s["violations"] for s in all_person_results)
                total_unknown_all = sum(s["unknown"] for s in all_person_results)

                st.markdown('<div class="section-header">📊 Aggregate Statistics</div>', unsafe_allow_html=True)
                render_metrics({
                    "total_persons": total_persons_all,
                    "compliant": total_compliant_all,
                    "violations": total_violations_all,
                    "unknown": total_unknown_all,
                })

                # Violation events
                st.markdown('<div class="section-header">🚨 Violation Events</div>', unsafe_allow_html=True)
                render_violation_events(event_log.events)

                # Playback
                st.markdown('<div class="section-header">🎥 Processed Video</div>', unsafe_allow_html=True)

                # Re-encode to H.264 for browser playback using OpenCV
                h264_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                try:
                    # Read the mp4v file and re-encode
                    cap_out = cv2.VideoCapture(out_path)
                    h264_writer = cv2.VideoWriter(h264_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

                    if not h264_writer.isOpened():
                        # Fallback: try with H264 fourcc
                        h264_writer = cv2.VideoWriter(h264_path, cv2.VideoWriter_fourcc(*"H264"), fps, (width, height))

                    if h264_writer.isOpened():
                        while True:
                            ret_out, frame_out = cap_out.read()
                            if not ret_out:
                                break
                            h264_writer.write(frame_out)
                        h264_writer.release()
                        cap_out.release()
                        playback_path = h264_path
                    else:
                        cap_out.release()
                        playback_path = out_path
                except Exception:
                    playback_path = out_path

                with open(playback_path, "rb") as vf:
                    st.video(vf.read())

    else:
        st.markdown("""
        <div class="info-panel">
            <h4>📤 Upload a Video</h4>
            <p>
                Upload a construction site video to analyze PPE compliance frame-by-frame.
                The processed video will show detections, compliance status, and violation events.
                Supported formats: MP4, AVI, MOV, MKV.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Mode: Real-Time Camera
# ──────────────────────────────────────────────────────────────────────────────

elif mode == "📹 Real-Time Camera":
    st.markdown('<div class="section-header">📹 Real-Time Camera Inference</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
        <h4>🔴 Live PPE Monitoring</h4>
        <p>
            Grant camera access to start real-time PPE compliance detection.
            Works with your laptop webcam or phone camera.
            The video stream is processed frame-by-frame with the YOLO11m model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Store real-time stats in session state
    if "rt_summary" not in st.session_state:
        st.session_state.rt_summary = {
            "total_persons": 0, "compliant": 0, "violations": 0, "unknown": 0,
        }
    if "rt_person_results" not in st.session_state:
        st.session_state.rt_person_results = []

    class PPEVideoProcessor(VideoProcessorBase):
        """WebRTC video processor that runs PPE detection on each frame."""

        def __init__(self):
            self.tracker = PersonTracker()
            self.event_log = ViolationEventLog()
            self.frame_count = 0
            self.conf = conf_threshold
            self.iou = iou_threshold
            self.last_summary = {}
            self.last_person_results = []

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            annotated, person_results, summary = annotate_frame_with_tracking(
                img, detector, self.tracker, self.event_log,
                self.frame_count, 30.0,
                conf=self.conf, iou=self.iou,
                imgsz=DEFAULT_IMGSZ
            )

            self.last_summary = summary
            self.last_person_results = person_results

            # Update session state (for display outside the stream)
            st.session_state.rt_summary = summary
            st.session_state.rt_person_results = person_results

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # WebRTC streamer
    ctx = webrtc_streamer(
        key="ppe-realtime",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PPEVideoProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # Real-time status panel
    if ctx.state.playing:
        st.markdown("""
        <div class="active-violation-banner" style="border-color: rgba(63, 185, 80, 0.4); background: linear-gradient(135deg, rgba(63, 185, 80, 0.1), rgba(63, 185, 80, 0.03));">
            <div class="banner-icon">🟢</div>
            <div class="banner-text" style="color: #3fb950;">Camera Active — Processing frames in real time</div>
        </div>
        """, unsafe_allow_html=True)

        # Display latest stats
        summary_placeholder = st.empty()
        table_placeholder = st.empty()

        with summary_placeholder:
            render_metrics(st.session_state.rt_summary)
        with table_placeholder:
            if st.session_state.rt_person_results:
                render_person_table(st.session_state.rt_person_results)

    else:
        st.markdown("""
        <div class="info-panel">
            <p>👆 Click <strong>"START"</strong> above to begin the live camera feed. 
            Your browser will ask for camera permission.</p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.8rem; padding: 1rem 0;">
    <p>PPE Compliance Detection System — Powered by YOLO11m fine-tuned on Roboflow Construction Dataset</p>
    <p>5 Classes: Helmet · No-Helmet · Vest · No-Vest · Person</p>
</div>
""", unsafe_allow_html=True)
