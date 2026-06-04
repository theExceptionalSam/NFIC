import io
import time
import warnings
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go
import timm
ML_AVAILABLE = True
except ImportError:
ML_AVAILABLE = False
 
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Naija Eats",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/best_fold0.pth"
IMG_SIZE        = 288
DEVICE          = torch.device("cpu")
TOP_K           = 5

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & base ─────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="block-container"] {
        background-color: #1A1208 !important;
        color: #FAF5EE !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }

    /* ── Sidebar ──────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #221B0F !important;
        border-right: 0.5px solid rgba(250,245,238,0.08) !important;
    }
    [data-testid="stSidebar"] * { color: #B8A898 !important; }

    /* ── Block padding ────────────────────────────── */
    [data-testid="block-container"] {
        padding: 0 2.5rem 3rem !important;
        max-width: 1100px !important;
        margin: 0 auto !important;
    }

    /* ── Headings ─────────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #FAF5EE !important;
        letter-spacing: -0.02em !important;
    }

    /* ── Metrics ──────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #2C2218 !important;
        border: 0.5px solid rgba(250,245,238,0.08) !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        color: #E8855A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #7A6A5A !important;
    }
    [data-testid="stMetricDelta"] { display: none !important; }

    /* ── File uploader ────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: #2C2218 !important;
        border: 1.5px dashed rgba(196,83,42,0.4) !important;
        border-radius: 18px !important;
        padding: 2rem !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #C4532A !important;
    }
    [data-testid="stFileUploader"] * { color: #B8A898 !important; }
    [data-testid="stFileUploader"] button {
        background: #C4532A !important;
        color: #FAF5EE !important;
        border: none !important;
        border-radius: 100px !important;
        padding: 0.4rem 1.2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ── Buttons ──────────────────────────────────── */
    [data-testid="stButton"] > button {
        background: transparent !important;
        border: 0.5px solid rgba(250,245,238,0.15) !important;
        color: #B8A898 !important;
        border-radius: 100px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        padding: 0.35rem 1rem !important;
        transition: all 0.2s !important;
    }
    [data-testid="stButton"] > button:hover {
        border-color: #C4532A !important;
        color: #E8855A !important;
        background: rgba(196,83,42,0.1) !important;
    }

    /* ── Toggle / Checkbox ────────────────────────── */
    [data-testid="stCheckbox"] label,
    [data-testid="stToggle"] label {
        color: #B8A898 !important;
        font-size: 0.85rem !important;
    }

    /* ── Selectbox ────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background: #2C2218 !important;
        border: 0.5px solid rgba(250,245,238,0.1) !important;
        border-radius: 8px !important;
        color: #FAF5EE !important;
    }

    /* ── Plotly chart container ───────────────────── */
    [data-testid="stPlotlyChart"] {
        background: transparent !important;
    }

    /* ── Divider ──────────────────────────────────── */
    hr {
        border-color: rgba(250,245,238,0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Image ────────────────────────────────────── */
    [data-testid="stImage"] img {
        border-radius: 14px !important;
        border: 0.5px solid rgba(250,245,238,0.08) !important;
    }

    /* ── Caption / small text ─────────────────────── */
    [data-testid="stCaptionContainer"] p,
    .stCaption, small {
        color: #7A6A5A !important;
        font-size: 0.72rem !important;
    }

    /* ── Spinner ──────────────────────────────────── */
    [data-testid="stSpinner"] p { color: #B8A898 !important; }

    /* ── Alerts / info ────────────────────────────── */
    [data-testid="stAlert"] {
        background: rgba(196,83,42,0.08) !important;
        border: 0.5px solid rgba(196,83,42,0.3) !important;
        border-radius: 12px !important;
        color: #FAF5EE !important;
    }

    /* ── Expander ─────────────────────────────────── */
    [data-testid="stExpander"] {
        background: #2C2218 !important;
        border: 0.5px solid rgba(250,245,238,0.08) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary { color: #B8A898 !important; }

    /* ── Columns gap fix ──────────────────────────── */
    [data-testid="column"] { gap: 0 !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class NigerianFoodClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout / 2),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
def _tfm(img_size: int, flip: bool = False, brightness: float = 0.0,
         crop_frac: float = 1.0) -> A.Compose:
    steps = []
    if crop_frac < 1.0:
        steps += [
            A.Resize(img_size, img_size),
            A.CenterCrop(int(img_size * crop_frac), int(img_size * crop_frac)),
        ]
    steps.append(A.Resize(img_size, img_size))
    if flip:
        steps.append(A.HorizontalFlip(p=1.0))
    if brightness != 0.0:
        steps.append(A.RandomBrightnessContrast(
            brightness_limit=(brightness, brightness), contrast_limit=0, p=1.0))
    steps += [
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ]
    return A.Compose(steps)


TTA_TRANSFORMS = [
    _tfm(IMG_SIZE),
    _tfm(IMG_SIZE, flip=True),
    _tfm(IMG_SIZE, brightness=0.1),
    _tfm(IMG_SIZE, crop_frac=0.9),
]


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = NigerianFoodClassifier(
        model_name  = ckpt["model_name"],
        num_classes = ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["class_names"]


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(img_pil: Image.Image, model: nn.Module, use_tta: bool = True):
    img_np = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    transforms = TTA_TRANSFORMS if use_tta else [TTA_TRANSFORMS[0]]
    all_probs  = []
    for tfm in transforms:
        x     = tfm(image=img_np)["image"].unsqueeze(0).to(DEVICE)
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────
def gradcam_overlay(img_pil: Image.Image, model: nn.Module, class_idx: int):
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return None

    last_conv = None
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        return None

    img_np  = np.array(img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
    img_f32 = img_np.astype(np.float32) / 255.0
    tfm     = TTA_TRANSFORMS[0]
    x       = tfm(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    cam     = GradCAMPlusPlus(model=model, target_layers=[last_conv])
    mask    = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
    overlay = show_cam_on_image(img_f32, mask, use_rgb=True)
    return Image.fromarray(overlay)


# ─────────────────────────────────────────────────────────────
# CONFIDENCE CHART  (dark theme)
# ─────────────────────────────────────────────────────────────
def confidence_chart(class_names, probs, top_k=5):
    top_idx = probs.argsort()[-top_k:][::-1]
    names   = [class_names[i] for i in top_idx][::-1]
    values  = [float(probs[i]) for i in top_idx][::-1]
    colors  = ["#E8855A" if i == len(values) - 1 else "rgba(196,83,42,0.45)"
               for i in range(len(values))]

    fig = go.Figure(go.Bar(
        x            = values,
        y            = names,
        orientation  = "h",
        marker_color = colors,
        marker_line_width = 0,
        text         = [f"{v:.1%}" for v in values],
        textposition = "outside",
        textfont     = dict(color="#B8A898", size=12, family="DM Sans"),
        hovertemplate = "<b>%{y}</b><br>Confidence: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        margin            = dict(l=0, r=60, t=4, b=4),
        xaxis             = dict(
            range=[0, 1.18],
            tickformat=".0%",
            tickfont=dict(color="#7A6A5A", size=11),
            gridcolor="rgba(250,245,238,0.05)",
            title="",
        ),
        yaxis             = dict(
            tickfont=dict(color="#B8A898", size=12, family="DM Sans"),
            title="",
        ),
        height            = 240,
        paper_bgcolor     = "rgba(0,0,0,0)",
        plot_bgcolor      = "rgba(0,0,0,0)",
        font              = dict(family="DM Sans"),
        hoverlabel        = dict(
            bgcolor="#2C2218",
            bordercolor="rgba(196,83,42,0.4)",
            font=dict(color="#FAF5EE", family="DM Sans"),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────
def nav_bar():
    st.markdown("""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding: 1.1rem 0 1.4rem;
        border-bottom: 0.5px solid rgba(250,245,238,0.08);
        margin-bottom: 2.5rem;
    ">
        <div style="
            display:flex; align-items:center; gap:10px;
            font-family:'Playfair Display',serif;
            font-size:1.1rem; color:#FAF5EE; letter-spacing:-0.01em;
        ">
            <div style="
                width:8px; height:8px; border-radius:50%;
                background:#C4532A;
                animation: pulse 2.4s ease-in-out infinite;
            "></div>
            Naija Eats
        </div>
        <div style="display:flex; gap:6px;">
            <span style="
                background:rgba(196,83,42,0.15);
                border:0.5px solid rgba(196,83,42,0.5);
                color:#E8855A; padding:4px 14px; border-radius:100px;
                font-size:0.72rem; font-weight:500;
                letter-spacing:0.04em; text-transform:uppercase;
            ">Classify</span>
            <span style="
                background:transparent;
                border:0.5px solid rgba(250,245,238,0.08);
                color:#7A6A5A; padding:4px 14px; border-radius:100px;
                font-size:0.72rem; letter-spacing:0.04em; text-transform:uppercase;
            ">EfficientNetV2-M</span>
        </div>
    </div>
    <style>
    @keyframes pulse {
        0%,100%{opacity:1;transform:scale(1)}
        50%{opacity:0.45;transform:scale(0.75)}
    }
    </style>
    """, unsafe_allow_html=True)


def hero_headline():
    st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <h1 style="
            font-family:'Playfair Display',Georgia,serif;
            font-size: clamp(2.4rem, 5vw, 3.8rem);
            font-weight:700; line-height:1.05;
            letter-spacing:-0.03em; color:#FAF5EE; margin:0 0 0.9rem;
        ">
            Identify any<br>
            <em style="color:#E8855A; font-style:italic;">Nigerian dish</em><br>
            instantly.
        </h1>
        <p style="
            font-size:0.95rem; line-height:1.75;
            color:#B8A898; max-width:460px; margin:0;
        ">
            Upload a photo of your meal — from jollof rice to egusi soup —
            and our fine-tuned model returns a classification with confidence
            scores in milliseconds.
        </p>
    </div>
    """, unsafe_allow_html=True)


def prediction_badge(food_name: str, confidence: float):
    if confidence >= 0.6:
        color, bg = "#E8855A", "rgba(196,83,42,0.12)"
        border     = "rgba(196,83,42,0.4)"
    elif confidence >= 0.35:
        color, bg = "#E8A020", "rgba(232,160,32,0.1)"
        border     = "rgba(232,160,32,0.35)"
    else:
        color, bg = "#E05C5C", "rgba(224,92,92,0.1)"
        border     = "rgba(224,92,92,0.35)"

    st.markdown(f"""
    <div style="
        background:{bg}; border:1px solid {border};
        border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1rem;
        position:relative; overflow:hidden;
    ">
        <div style="
            position:absolute; top:-30px; right:-30px;
            width:90px; height:90px; border-radius:50%;
            background:rgba(196,83,42,0.08);
        "></div>
        <p style="
            font-size:0.65rem; text-transform:uppercase;
            letter-spacing:0.1em; color:{color};
            margin:0 0 0.3rem;
        ">Classified as</p>
        <p style="
            font-family:'Playfair Display',serif;
            font-size:1.9rem; font-weight:700;
            letter-spacing:-0.02em; color:#FAF5EE;
            margin:0 0 0.7rem; line-height:1.1;
        ">{food_name}</p>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="
                flex:1; height:5px; border-radius:100px;
                background:rgba(250,245,238,0.08); overflow:hidden;
            ">
                <div style="
                    height:100%; width:{confidence*100:.1f}%;
                    background:{color}; border-radius:100px;
                "></div>
            </div>
            <span style="
                font-family:'Playfair Display',serif;
                font-size:1.25rem; color:{color};
                min-width:52px; text-align:right;
            ">{confidence:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def inference_meta_strip(elapsed_ms: float, use_tta: bool, num_classes: int):
    tta_label = "TTA ×4 on" if use_tta else "TTA off"
    st.markdown(f"""
    <div style="
        display:flex; gap:20px; flex-wrap:wrap;
        padding:0.8rem 0;
        border-top:0.5px solid rgba(250,245,238,0.08);
        margin-top:0.5rem;
    ">
        <span style="display:flex;align-items:center;gap:6px;font-size:0.72rem;color:#7A6A5A;">
            <span style="width:6px;height:6px;border-radius:50%;background:#4CAF50;display:inline-block"></span>
            Inference: <strong style="color:#B8A898;">{elapsed_ms:.0f} ms</strong>
        </span>
        <span style="display:flex;align-items:center;gap:6px;font-size:0.72rem;color:#7A6A5A;">
            <span style="width:6px;height:6px;border-radius:50%;background:#E8A020;display:inline-block"></span>
            <strong style="color:#B8A898;">{tta_label}</strong>
        </span>
        <span style="display:flex;align-items:center;gap:6px;font-size:0.72rem;color:#7A6A5A;">
            <span style="width:6px;height:6px;border-radius:50%;background:#C4532A;display:inline-block"></span>
            Classes: <strong style="color:#B8A898;">{num_classes}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f"""
    <p style="
        font-size:0.65rem; text-transform:uppercase;
        letter-spacing:0.1em; color:#7A6A5A;
        margin:0 0 0.6rem;
    ">{text}</p>
    """, unsafe_allow_html=True)


def footer():
    st.markdown("""
    <hr style="border-color:rgba(250,245,238,0.08);margin:3rem 0 1.5rem;">
    <div style="
        display:flex; justify-content:space-between; flex-wrap:wrap;
        gap:0.5rem; padding-bottom:2rem;
    ">
        <p style="font-size:0.7rem;color:#7A6A5A;margin:0;">
            EfficientNetV2-M &middot; 5-fold CV &middot; Macro F1 &asymp; 0.805
            &middot; Built with PyTorch + timm + Streamlit
        </p>
        <p style="font-size:0.7rem;color:#7A6A5A;margin:0;">
            288&times;288 input &middot; Fold 0 Val F1 0.8086
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def sidebar(class_names):
    with st.sidebar:
        st.markdown("""
        <p style="
            font-family:'Playfair Display',serif;
            font-size:1.2rem; color:#FAF5EE;
            margin:1rem 0 0.25rem;
        ">Settings</p>
        """, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        use_tta  = st.toggle("Test-Time Augmentation", value=True,
                             help="Averages 4 views — more accurate, slightly slower")
        show_cam = st.toggle("Grad-CAM Overlay", value=False,
                             help="Highlights which part of the image drove the prediction")
        top_k    = st.selectbox("Top-K predictions", [3, 5, 10], index=1)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.65rem;text-transform:uppercase;
                   letter-spacing:0.08em;color:#7A6A5A;margin-bottom:0.5rem;">
            Model info
        </p>""", unsafe_allow_html=True)

        info_rows = [
            ("Architecture", "EfficientNetV2-M"),
            ("Classes",      str(len(class_names))),
            ("Input size",   f"{IMG_SIZE}×{IMG_SIZE}"),
            ("Val F1",       "0.8086  (fold 0)"),
            ("CV strategy",  "5-fold"),
        ]
        for label, val in info_rows:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        margin-bottom:0.3rem;">
                <span style="font-size:0.75rem;color:#7A6A5A;">{label}</span>
                <span style="font-size:0.75rem;color:#B8A898;font-weight:500;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.65rem;text-transform:uppercase;
                   letter-spacing:0.08em;color:#7A6A5A;margin-bottom:0.5rem;">
            All classes
        </p>""", unsafe_allow_html=True)

        for name in sorted(class_names):
            st.markdown(f"""
            <div style="
                display:inline-block; background:rgba(196,83,42,0.08);
                border:0.5px solid rgba(196,83,42,0.2);
                border-radius:100px; padding:2px 10px;
                margin:2px 2px; font-size:0.7rem; color:#B8A898;
            ">{name}</div>
            """, unsafe_allow_html=True)

    return use_tta, show_cam, top_k


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    inject_styles()

    # ── Load model ───────────────────────────────────────────
    if not Path(CHECKPOINT_PATH).exists():
        st.markdown("""
        <div style="
            background:rgba(224,92,92,0.08);
            border:1px solid rgba(224,92,92,0.3);
            border-radius:14px; padding:1.5rem;
        ">
            <p style="font-family:'Playfair Display',serif;
                       font-size:1.1rem;color:#FAF5EE;margin:0 0 0.5rem;">
                Checkpoint not found
            </p>
            <p style="font-size:0.8rem;color:#B8A898;margin:0;">
                Place <code>best_fold0.pth</code> inside a
                <code>checkpoints/</code> folder next to <code>app.py</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    model, class_names = load_model(CHECKPOINT_PATH)

    # ── Sidebar ──────────────────────────────────────────────
    use_tta, show_cam, top_k = sidebar(class_names)

    # ── Nav + Hero ───────────────────────────────────────────
    nav_bar()

    col_hero, col_upload = st.columns([1.1, 1], gap="large")

    with col_hero:
        hero_headline()

        # Stat row
        m1, m2, m3 = st.columns(3)
        m1.metric("Food classes", len(class_names))
        m2.metric("Macro F1",     "80.8%")
        m3.metric("Inference",    "<200 ms")

    with col_upload:
        st.markdown("""
        <div style="
            background:#2C2218; border:1.5px dashed rgba(196,83,42,0.35);
            border-radius:18px; padding:1.4rem 1.4rem 0.8rem;
            margin-top:0.5rem;
        ">
            <p style="
                font-family:'Playfair Display',serif;
                font-size:1.15rem; color:#FAF5EE; margin:0 0 0.3rem;
            ">Drop your dish here</p>
            <p style="font-size:0.78rem;color:#7A6A5A;margin:0 0 0.8rem;">
                JPG · PNG · WEBP · up to 10 MB
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload food image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

    st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)

    # ── No image state ────────────────────────────────────────
    if uploaded is None:
        st.markdown("""
        <div style="
            background:rgba(196,83,42,0.06);
            border:0.5px solid rgba(196,83,42,0.2);
            border-radius:14px; padding:1.2rem 1.5rem;
        ">
            <p style="font-size:0.85rem;color:#B8A898;margin:0;">
                👆 Upload any Nigerian food photo — jollof rice, egusi soup,
                suya, puff puff, moi moi, and more are all supported.
            </p>
        </div>
        """, unsafe_allow_html=True)
        footer()
        return

    # ── Divider ───────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Layout: image | results ───────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    img_pil = Image.open(uploaded).convert("RGB")

    with col_img:
        section_label("Uploaded image")
        st.image(img_pil, use_container_width=True)
        st.caption(f"Size: {img_pil.width}×{img_pil.height} px")

    with col_res:
        section_label("Prediction")

        with st.spinner("Classifying…"):
            t0      = time.time()
            probs   = predict(img_pil, model, use_tta=use_tta)
            elapsed = (time.time() - t0) * 1000

        top_idx   = int(probs.argmax())
        top_class = class_names[top_idx]
        top_conf  = float(probs[top_idx])

        prediction_badge(top_class, top_conf)

        section_label(f"Top {top_k} predictions")
        st.plotly_chart(
            confidence_chart(class_names, probs, top_k),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        inference_meta_strip(elapsed, use_tta, len(class_names))

    # ── Grad-CAM ──────────────────────────────────────────────
    if show_cam:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <p style="
            font-family:'Playfair Display',serif;
            font-size:1.3rem;color:#FAF5EE;margin:0 0 0.3rem;
        ">Grad-CAM — what the model is looking at</p>
        <p style="font-size:0.8rem;color:#7A6A5A;margin-bottom:1.2rem;">
            Warm regions drove the classification; cool regions were ignored.
        </p>
        """, unsafe_allow_html=True)

        with st.spinner("Generating attention map…"):
            overlay = gradcam_overlay(img_pil, model, top_idx)

        if overlay is not None:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.image(img_pil.resize((IMG_SIZE, IMG_SIZE)), caption="Original",
                         use_container_width=True)
            with c2:
                st.image(overlay, caption="Grad-CAM++ overlay",
                         use_container_width=True)
            with c3:
                st.markdown("""
                <div style="
                    background:#2C2218; border:0.5px solid rgba(250,245,238,0.08);
                    border-radius:12px; padding:1.1rem 1.2rem;
                ">
                    <p style="font-size:0.75rem;color:#B8A898;margin:0 0 0.6rem;font-weight:500;">
                        Reading the map
                    </p>
                    <p style="font-size:0.78rem;color:#7A6A5A;line-height:1.6;margin:0;">
                        🔴 <strong style="color:#E8855A;">Red / warm</strong> —
                        high activation, the model focused here.<br><br>
                        🔵 <strong style="color:#6EA8D8;">Blue / cool</strong> —
                        ignored region.<br><br>
                        A well-calibrated model lights up the
                        <em>dish itself</em>, not the plate or background.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background:rgba(232,160,32,0.08);
                border:0.5px solid rgba(232,160,32,0.3);
                border-radius:12px; padding:1rem 1.2rem;
            ">
                <p style="font-size:0.82rem;color:#B8A898;margin:0;">
                    <code>pytorch-grad-cam</code> not installed.
                    Add it to <code>requirements.txt</code> and redeploy.
                </p>
            </div>
            """, unsafe_allow_html=True)

    footer()


if __name__ == "__main__":
    main()
