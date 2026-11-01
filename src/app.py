"""
Road Pothole Detection — End-to-End Demo Dashboard
===================================================
Streamlit single-page app that lets you:
  1. Upload a video (or use the smoke-test sample)
  2. Run YOLO11s detection + RoadTracker tracking + severity classification
  3. View annotated video output inline
  4. Explore severity breakdowns and detection statistics
  5. Download CSV, JSON, and annotated video

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_settings
from src.video.runner import run_video

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Pothole Detector",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #141428 50%, #0a0a18 100%);
}

/* Header hero */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f97316, #ef4444, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.0rem;
    margin-bottom: 1.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(249,115,22,0.5); }
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f97316;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Severity badges */
.badge-low    { background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#92400e; color:#fde68a; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-high   { background:#7f1d1d; color:#fecaca; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-uncertain { background:#1e293b; color:#94a3b8; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 3px solid #f97316;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 15, 26, 0.95);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f97316, #ef4444);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.5rem;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* Progress bar color */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #f97316, #a855f7); }

/* DataFrame styling */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Alert box */
.info-box {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #c7d2fe;
    font-size: 0.88rem;
    margin: 0.5rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
SMOKE_VIDEO = (
    PROJECT_ROOT / "sample_videos" / "video_smoke_native_input.mp4"
    if (PROJECT_ROOT / "sample_videos" / "video_smoke_native_input.mp4").exists()
    else PROJECT_ROOT / "outputs" / "video_smoke_native_input.mp4"
)
SEVERITY_COLORS = {
    "low":       "#22c55e",
    "medium":    "#f59e0b",
    "high":      "#ef4444",
    "uncertain": "#64748b",
}
CLASS_NAMES = {0: "Longitudinal Crack", 1: "Transverse Crack", 2: "Alligator Crack", 3: "Pothole"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def severity_badge(label: str) -> str:
    cls = label.lower() if label.lower() in ("low", "medium", "high") else "uncertain"
    return f'<span class="badge-{cls}">{label or "—"}</span>'


def metric_card(value: str, label: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


@st.cache_resource
def get_settings():
    return load_settings(CONFIG_PATH)


def run_inference(
    video_path: Path,
    output_dir: Path,
    device: str,
    severity: bool,
    pothole_conf: float = 0.05,
    image_size: int = 800,
) -> dict:
    settings = get_settings()
    updated_class_conf = dict(settings.detector.class_confidence)
    updated_class_conf[3] = pothole_conf
    settings = replace(
        settings,
        detector=replace(
            settings.detector,
            device=device,
            image_size=image_size,
            class_confidence=updated_class_conf,
            candidate_confidence=min(settings.detector.candidate_confidence, pothole_conf * 0.6),
        ),
    )
    return run_video(
        video_path,
        settings,
        output_dir=output_dir,
        severity_enabled=severity,
    )


def build_severity_bar_chart(track_rows: list[dict]) -> None:
    """Render a horizontal bar chart for pothole severity distribution."""
    pothole_tracks = [r for r in track_rows if r["class_id"] == 3]
    if not pothole_tracks:
        st.info("No pothole tracks detected in this video.")
        return

    counts = {"Low": 0, "Medium": 0, "High": 0, "Uncertain": 0}
    for row in pothole_tracks:
        sev = (row.get("severity") or "").capitalize()
        if sev in counts:
            counts[sev] += 1
        else:
            counts["Uncertain"] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SEVERITY_COLORS[k.lower()] for k in labels]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none")

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left",
                color="white", fontsize=11, fontweight="bold",
            )

    ax.set_xlabel("Number of Pothole Tracks", color="#64748b", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#1e293b")
    ax.spines["left"].set_color("#1e293b")
    ax.set_xlim(0, max(values) + 1.5 if max(values) > 0 else 3)
    ax.grid(axis="x", color="#1e293b", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def build_defect_donut(track_rows: list[dict]) -> None:
    """Donut chart of all detected defect classes."""
    if not track_rows:
        return
    class_counts: dict[str, int] = {}
    for row in track_rows:
        name = CLASS_NAMES.get(row["class_id"], f"Class {row['class_id']}")
        class_counts[name] = class_counts.get(name, 0) + 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(class_counts.keys())
    values = list(class_counts.values())
    palette = ["#f97316", "#a855f7", "#3b82f6", "#ef4444"]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.0f%%",
        colors=palette[:len(labels)],
        startangle=90, wedgeprops=dict(width=0.55, edgecolor="#0f0f1a"),
        textprops=dict(color="#94a3b8", fontsize=8.5),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax.set_title("Defect Distribution", color="#e2e8f0", fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    use_sample = st.checkbox(
        "Use built-in smoke video",
        value=SMOKE_VIDEO.exists(),
        help="Use the pre-existing 10-frame smoke test video",
    )

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader(
            "Upload a road video",
            type=["mp4", "avi", "mov", "mkv"],
            help="MP4 recommended. Large files (>200MB) may take several minutes.",
        )

    st.markdown("---")
    device_choice = st.selectbox(
        "Compute device",
        options=["auto", "cpu", "0"],
        index=0,
        help="'auto' selects GPU if available, else CPU. '0' = first CUDA GPU.",
    )
    enable_severity = st.checkbox("Enable severity classification", value=True)

    pothole_conf = st.slider(
        "Pothole sensitivity (confidence threshold)",
        min_value=0.02,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Lower values detect simple/faint/distant potholes earlier. Default: 0.05",
    )
    resolution_choice = st.selectbox(
        "Inference resolution",
        options=[800, 640],
        index=0,
        help="800px captures distant potholes ahead on the road with high clarity. 640px is standard.",
    )
    st.markdown("---")

    st.markdown("""
    <div style='color:#64748b; font-size:0.78rem; line-height:1.6'>
     <b style='color:#94a3b8'>Model:</b> YOLO checkpoint from the configured model path<br>
    <b style='color:#94a3b8'>Severity:</b> ResNet18 v3 + BBox heuristic<br>
     <b style='color:#94a3b8'>Tracker:</b> RoadTracker<br>
    <b style='color:#94a3b8'>Classes:</b> 4 defect types<br>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🚧 Road Pothole Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered defect detection & severity classification '
    'using YOLO11s + ResNet18 · RoadTracker multi-object tracking</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="info-box">Upload a dashcam or inspection video. '
            'The system detects potholes and road cracks, tracks each defect across '
            'frames, classifies pothole severity as <b>Low / Medium / High</b>, '
            'and produces an annotated video with a full report.</div>',
            unsafe_allow_html=True)

# ── Run section ───────────────────────────────────────────────────────────────
run_ready = use_sample and SMOKE_VIDEO.exists() or uploaded_file is not None

if st.button("▶ Analyze Video", disabled=not run_ready, use_container_width=True):
    with st.spinner("Loading models and preparing video…"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="pothole_"))
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_sample and SMOKE_VIDEO.exists():
            video_path = SMOKE_VIDEO
        else:
            video_bytes = uploaded_file.read()
            video_path = tmp_dir / (uploaded_file.name or "input.mp4")
            video_path.write_bytes(video_bytes)

    progress_bar = st.progress(0, text="Starting inference…")
    status_text  = st.empty()

    start_time = time.time()
    try:
        status_text.markdown("⚙️ Running detection & tracking — this may take a moment…")
        progress_bar.progress(10, text="Running YOLO11s + RoadTracker…")

        report = run_inference(
            video_path,
            output_dir,
            device_choice,
            enable_severity,
            pothole_conf=pothole_conf,
            image_size=resolution_choice,
        )
        progress_bar.progress(90, text="Generating reports…")

        elapsed = time.time() - start_time
        progress_bar.progress(100, text="Done!")
        status_text.success(f"✅ Analysis complete in {elapsed:.1f}s")

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Inference failed: {exc}")
        st.stop()

    # ── Store in session state ───────────────────────────────────────────────
    st.session_state["report"] = report
    st.session_state["output_dir"] = output_dir
    st.session_state["video_path"] = video_path

# ── Results panel ─────────────────────────────────────────────────────────────
if "report" in st.session_state:
    report: dict = st.session_state["report"]
    output_dir: Path = st.session_state["output_dir"]
    video_path: Path = st.session_state["video_path"]

    run_info   = report.get("run", {})
    track_rows = report.get("tracks", [])
    config     = report.get("configuration", {})

    # ── KPI row ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card(str(run_info.get("frames", "—")), "Frames processed"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(str(run_info.get("tracks", "—")), "Unique defect tracks"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(str(run_info.get("unique_pothole_tracks", "—")), "Pothole tracks"), unsafe_allow_html=True)
    fps = run_info.get("fps", 0)
    duration = run_info.get("duration_seconds", 0) or 0
    with c4:
        st.markdown(metric_card(f"{duration:.1f}s", "Video duration"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card(f"{fps:.0f}", "FPS"), unsafe_allow_html=True)

    # ── Charts row ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Severity & Defect Analysis</div>', unsafe_allow_html=True)
    chart_col, donut_col = st.columns([3, 2])
    with chart_col:
        st.markdown("**Pothole Severity Distribution**")
        build_severity_bar_chart(track_rows)
    with donut_col:
        st.markdown("**Defect Class Breakdown**")
        build_defect_donut(track_rows)

    # ── Video output ─────────────────────────────────────────────────────────
    annotated_mp4 = output_dir / "annotated.mp4"
    if annotated_mp4.exists():
        st.markdown('<div class="section-header">🎬 Annotated Output Video</div>', unsafe_allow_html=True)
        col_vid, col_dl = st.columns([4, 1])
        with col_vid:
            video_bytes = annotated_mp4.read_bytes()
            st.video(video_bytes, format="video/mp4")
        with col_dl:
            st.download_button(
                "⬇ Download annotated video",
                data=video_bytes,
                file_name="annotated_pothole_detection.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    # ── Track table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗂️ Detection Track Report</div>', unsafe_allow_html=True)

    if track_rows:
        df = pd.DataFrame(track_rows)
        # Rename for display
        display_cols = {
            "track_id": "Track ID",
            "class_name": "Defect Type",
            "first_time_seconds": "First Seen (s)",
            "last_time_seconds": "Last Seen (s)",
            "observations": "Frames",
            "max_detection_confidence": "Confidence",
            "severity": "Severity",
            "severity_confidence": "Sev. Confidence",
        }
        display_df = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)

        # Round floats
        for col in ["First Seen (s)", "Last Seen (s)", "Confidence", "Sev. Confidence"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: round(float(x), 3) if x not in (None, "", "nan") else "—"
                )

        st.dataframe(
            display_df,
            use_container_width=True,
            height=320,
            hide_index=True,
        )

        # Downloads
        dl1, dl2, dl3 = st.columns(3)
        tracks_csv = output_dir / "tracks.csv"
        report_json = output_dir / "report.json"

        with dl1:
            if tracks_csv.exists():
                st.download_button(
                    "⬇ tracks.csv",
                    data=tracks_csv.read_bytes(),
                    file_name="tracks.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with dl2:
            if report_json.exists():
                st.download_button(
                    "⬇ report.json",
                    data=report_json.read_bytes(),
                    file_name="report.json",
                    mime="application/json",
                    use_container_width=True,
                )
        with dl3:
            config_str = json.dumps(config, indent=2)
            st.download_button(
                "⬇ config.json",
                data=config_str.encode(),
                file_name="run_config.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("No defect tracks were detected in this video.")

    # ── Per-track severity detail ─────────────────────────────────────────────
    pothole_tracks = [r for r in track_rows if r.get("class_id") == 3]
    if pothole_tracks and enable_severity:
        st.markdown('<div class="section-header">🔍 Pothole Severity Detail</div>', unsafe_allow_html=True)
        for row in pothole_tracks:
            sev = (row.get("severity") or "").lower()
            conf = row.get("severity_confidence") or 0.0
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 0.0
            conf_pct = f"{conf * 100:.1f}%"
            badge_html = severity_badge(sev or "uncertain")

            t1, t2, t3, t4 = st.columns([1, 2, 2, 2])
            with t1:
                st.markdown(f"**Track #{row['track_id']}**")
            with t2:
                st.markdown(f"Severity: {badge_html}", unsafe_allow_html=True)
            with t3:
                st.markdown(f"Confidence: **{conf_pct}**")
            with t4:
                obs = row.get("severity_observations", 0)
                st.markdown(f"Severity observations: **{obs}**")

    # ── Run config accordion ─────────────────────────────────────────────────
    with st.expander("⚙️ Run configuration"):
        st.json(config)

else:
    # ── Placeholder when no analysis run yet ─────────────────────────────────
    st.markdown('<div class="section-header">🎥 How it works</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    steps = [
        ("1️⃣ Upload", "Select a dashcam or road inspection video in the sidebar, or use the built-in smoke test."),
        ("2️⃣ Detect & Track", "YOLO11s detects 4 road defect classes. RoadTracker assigns persistent IDs within the video clip."),
        ("3️⃣ Severity", "Pothole tracks can receive ResNet18 severity estimates blended with a bounding-box area prior."),
    ]
    for col, (title, desc) in zip([col_a, col_b, col_c], steps):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="min-height:130px">
                <div style="font-size:1.4rem; margin-bottom:0.4rem">{title}</div>
                <div style="color:#94a3b8; font-size:0.88rem">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="margin-top:1rem">
        <div style="color:#64748b; font-size:0.85rem">
        <b style="color:#94a3b8">Detector:</b> YOLO checkpoint from the configured model path<br>
        <b style="color:#94a3b8">Severity:</b> ResNet18-v3 &middot; Focal Loss + MixUp &middot; TTA 5-crop &middot; balanced acc = 61.7%<br>
        <b style="color:#94a3b8">Severity:</b> Optional ResNet18 checkpoint with a bounding-box area prior<br>
        <b style="color:#94a3b8">Tracker:</b> RoadTracker &mdash; defects tracked within the video clip
        </div>
    </div>""", unsafe_allow_html=True)
