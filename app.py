"""
Naija Eats AI — Nigerian Food Classifier
Streamlit app wired to the NigerianFoodClassifier notebook model.

Install:
    pip install streamlit pillow torch torchvision timm albumentations grad-cam

Run:
    streamlit run naija_eats_app.py

Checkpoint:
    Place best_fold0.pth (or any best_fold*.pth) in the same directory.
    The checkpoint must contain the keys saved during training:
        model_state_dict, class_names, num_classes, img_size, model_name
"""

# ── Standard lib ───────────────────────────────────────────────────────────────
import io, gc, time, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ── Third-party ────────────────────────────────────────────────────────────────
import streamlit as st
from PIL import Image
import numpy as np

# ── Optional ML ───────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Fallback class list (alphabetical) if no checkpoint is loaded.
# The notebook derives these from sorted(df['class_name'].unique()) — 18 classes.
FALLBACK_CLASSES: List[str] = [
    "Afang Soup", "Asun", "Banga Rice", "Banga Soup", "Chin Chin",
    "Eba", "Edikaikong", "Egusi Soup", "Ewedu", "Fried Rice",
    "Fufu", "Jollof Rice", "Kilishi", "Moi Moi", "Ogbono Soup",
    "Ofe Onugbu", "Pepper Soup", "Pounded Yam",
]

IMG_SIZE     = 224          # CFG.img_size
MODEL_NAME   = "tf_efficientnetv2_m"   # CFG.model_name
DROPOUT      = 0.3          # CFG.dropout_rate

# ── Auto-discover checkpoint (best_fold0.pth … best_fold4.pth) ────────────────
def _find_checkpoint() -> Optional[Path]:
    for i in range(5):
        p = Path(f"best_fold{i}.pth")
        if p.exists():
            return p
    return None

CKPT_PATH = _find_checkpoint()

DEMO_DISHES = [
    ("🍚", "Jollof Rice"),
    ("🫕", "Puff Puff"),
    ("🍖", "Pepper Soup"),
    ("🧁", "Chin Chin"),
]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must match the notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════

if ML_AVAILABLE:
    class NigerianFoodClassifier(nn.Module):
        """
        Mirrors the notebook architecture:
            tf_efficientnetv2_m backbone (num_classes=0, global_pool='avg')
            → BN → Dropout(p/2) → Linear(feat→feat//2)
            → BN → SiLU → Dropout(p) → Linear(feat//2→num_classes)
        """
        def __init__(self, model_name: str, num_classes: int,
                     pretrained: bool = False, dropout: float = 0.3):
            super().__init__()
            self.backbone = timm.create_model(
                model_name,
                pretrained  = pretrained,
                num_classes = 0,
                global_pool = "avg",
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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached so it only runs once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[Optional[object], List[str], int, bool]:
    """
    Returns (model, class_names, img_size, weights_loaded).
    Falls back gracefully if ML libs or checkpoint are missing.
    """
    if not ML_AVAILABLE:
        return None, FALLBACK_CLASSES, IMG_SIZE, False

    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        # Build empty model so architecture is at least validated
        model = NigerianFoodClassifier(MODEL_NAME, len(FALLBACK_CLASSES),
                                       pretrained=False, dropout=DROPOUT)
        model.eval()
        return model, FALLBACK_CLASSES, IMG_SIZE, False

    ckpt        = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt.get("class_names", FALLBACK_CLASSES)
    num_classes = ckpt.get("num_classes", len(class_names))
    img_size    = ckpt.get("img_size",    IMG_SIZE)
    model_name  = ckpt.get("model_name",  MODEL_NAME)

    model = NigerianFoodClassifier(model_name, num_classes,
                                   pretrained=False, dropout=DROPOUT)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, img_size, True


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS  (mirrors the notebook's get_val_transforms / get_tta_transforms)
# ══════════════════════════════════════════════════════════════════════════════

def _val_tfm(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])

def _tta_tfms(img_size: int) -> List[A.Compose]:
    """4 deterministic TTA views — exact copy of notebook's get_tta_transforms."""
    return [
        # 1. Original (clean val transform)
        _val_tfm(img_size),
        # 2. Horizontal flip
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]),
        # 3. Slight brightness boost
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1),
                                       contrast_limit=0, p=1.0),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]),
        # 4. Center crop 90 %
        A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(int(img_size * 0.9), int(img_size * 0.9)),
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM++  (mirrors run_gradcam in the notebook)
# ══════════════════════════════════════════════════════════════════════════════

def _last_conv(model: "NigerianFoodClassifier") -> Optional[nn.Module]:
    """
    Return the last nn.Conv2d in model.backbone — exact strategy from notebook:
        last_conv = None
        for _, m in model.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
    """
    last = None
    for _, m in model.backbone.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def run_gradcam(
    model: "NigerianFoodClassifier",
    img_np: np.ndarray,       # uint8 RGB HxWx3
    class_idx: int,
    img_size: int,
) -> Optional[Image.Image]:
    """
    GradCAM++ overlay using ClassifierOutputTarget — same as the notebook.
    Returns a PIL image (RGB) with the heatmap blended onto the original.
    """
    if not GRADCAM_AVAILABLE:
        return None

    last = _last_conv(model)
    if last is None:
        return None

    val_tfm   = _val_tfm(img_size)
    tensor    = val_tfm(image=img_np)["image"].unsqueeze(0)  # (1,3,H,W)
    orig_resized = np.array(
        Image.fromarray(img_np).resize((img_size, img_size))
    ).astype(np.float32) / 255.0

    target = [ClassifierOutputTarget(class_idx)]

    try:
        with GradCAMPlusPlus(model=model, target_layers=[last]) as cam:
            grayscale = cam(input_tensor=tensor, targets=target)[0]  # (H, W)
        overlay = show_cam_on_image(orig_resized, grayscale, use_rgb=True)
        return Image.fromarray(overlay)
    except Exception as exc:
        st.caption(f"Grad-CAM error: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def classify(
    pil_img:   Image.Image,
    use_tta:   bool,
    top_k:     int,
    use_gcam:  bool,
) -> Tuple[List[Dict], float, Optional[Image.Image]]:
    """
    Returns:
        results    — [{'class': str, 'conf': float}, ...]  top-k descending
        elapsed_ms — wall-clock inference time (0.0 in demo mode)
        cam_img    — PIL overlay or None
    """
    model, class_names, img_size, weights_loaded = load_model()

    img_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    # ── Demo mode (no model / no weights) ─────────────────────────────────────
    if model is None or not ML_AVAILABLE:
        picks = random.sample(range(len(class_names)), min(top_k, len(class_names)))
        raw   = sorted([(i, random.uniform(0.02, 0.95)) for i in picks],
                       key=lambda x: -x[1])
        s     = sum(v for _, v in raw)
        results = [{"class": class_names[i], "conf": round(v / s, 4)} for i, v in raw]
        return results, 0.0, None

    # ── Real inference ─────────────────────────────────────────────────────────
    device = next(model.parameters()).device
    transforms = _tta_tfms(img_size) if use_tta else [_val_tfm(img_size)]

    t0 = time.perf_counter()
    all_probs = []
    with torch.no_grad():
        for tfm in transforms:
            x     = tfm(image=img_np)["image"].unsqueeze(0).to(device)
            probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
            all_probs.append(probs)
    avg_probs  = np.mean(all_probs, axis=0)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    top_idx = avg_probs.argsort()[-top_k:][::-1]
    results = [{"class": class_names[i], "conf": round(float(avg_probs[i]), 4)}
               for i in top_idx]

    # ── Grad-CAM++ ────────────────────────────────────────────────────────────
    cam_img = None
    if use_gcam:
        if not weights_loaded:
            st.caption(
                f"⚠️  No checkpoint found (`best_fold0.pth` … `best_fold4.pth`). "
                "Grad-CAM needs real weights — place your `.pth` file alongside the app."
            )
        elif not GRADCAM_AVAILABLE:
            st.caption("⚠️  `grad-cam` not installed. Run `pip install grad-cam`.")
        else:
            cam_img = run_gradcam(model, img_np, int(top_idx[0]), img_size)

    return results, elapsed_ms, cam_img


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Naija Eats · AI",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #1a1008;
    color: #f0e6d3;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* NAV */
.navbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:.75rem 1.5rem;
    border-bottom:1px solid rgba(240,230,211,.12);
    margin-bottom:2rem;
}
.nav-logo { font-weight:600; font-size:1rem; color:#f0e6d3; display:flex; align-items:center; gap:6px; }
.nav-logo span { color:#e8673a; }
.nav-pills { display:flex; gap:8px; }
.nav-pill {
    padding:6px 16px; border:1px solid rgba(240,230,211,.3); border-radius:999px;
    font-size:.82rem; font-weight:500; color:#f0e6d3; background:transparent;
    cursor:pointer; text-decoration:none;
}
.nav-pill:hover { background:rgba(232,103,58,.15); border-color:#e8673a; }

/* HERO */
.hero-headline {
    font-family:'Playfair Display',serif;
    font-size:clamp(2.8rem,6vw,5rem); font-weight:700; line-height:1.05;
    color:#f0e6d3; margin-bottom:1rem;
}
.hero-headline em { font-style:italic; color:#e8673a; }
.hero-sub { font-size:1rem; color:rgba(240,230,211,.65); max-width:480px; line-height:1.65; margin-bottom:1.8rem; }

/* STATS */
.stats-row { display:flex; gap:2.5rem; margin-top:1.5rem; }
.stat-item { display:flex; flex-direction:column; }
.stat-value { font-family:'Playfair Display',serif; font-size:1.9rem; font-weight:700; color:#e8673a; line-height:1; }
.stat-label { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:rgba(240,230,211,.5); margin-top:3px; }

/* UPLOAD CARD */
.upload-card {
    background:rgba(232,103,58,.08); border:1.5px dashed rgba(232,103,58,.4);
    border-radius:16px; padding:2.5rem 1.5rem; text-align:center;
}
.upload-icon { font-size:2.5rem; margin-bottom:.75rem; }
.upload-title { font-family:'Playfair Display',serif; font-size:1.25rem; color:#f0e6d3; margin-bottom:.4rem; }
.upload-hint { font-size:.8rem; color:rgba(240,230,211,.45); margin-bottom:1rem; }
.fmt-badge {
    display:inline-block; padding:3px 10px;
    border:1px solid rgba(240,230,211,.2); border-radius:999px;
    font-size:.72rem; color:rgba(240,230,211,.55); margin:2px;
}

/* MODEL BADGE */
.model-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(240,230,211,.06); border:1px solid rgba(240,230,211,.12);
    border-radius:999px; padding:4px 12px; font-size:.75rem; color:rgba(240,230,211,.6);
    margin-bottom:1.2rem;
}
.dot-green  { width:7px; height:7px; border-radius:50%; background:#4ade80; }
.dot-yellow { width:7px; height:7px; border-radius:50%; background:#facc15; }

/* RESULT CARD */
.result-card {
    background:rgba(240,230,211,.05); border:1px solid rgba(240,230,211,.1);
    border-radius:16px; padding:1.5rem; margin-top:1.5rem;
}
.result-label { font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:rgba(240,230,211,.45); margin-bottom:.3rem; }
.result-dish  { font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:#f0e6d3; line-height:1.1; margin-bottom:.25rem; }
.result-conf  { font-size:1.2rem; color:#e8673a; font-weight:600; }
.conf-bar-bg  { background:rgba(240,230,211,.1); border-radius:999px; height:6px; margin:.75rem 0; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#e8673a,#f5a623); }
.top-k-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:6px 0; border-bottom:1px solid rgba(240,230,211,.06); font-size:.85rem;
}
.top-k-row:last-child { border-bottom:none; }
.top-k-name { color:rgba(240,230,211,.8); }
.top-k-pct  { color:#e8673a; font-weight:600; font-size:.82rem; }

/* GRADCAM SECTION */
.cam-label {
    font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
    color:rgba(240,230,211,.4); margin-bottom:.5rem; margin-top:.4rem;
}
.cam-empty {
    height:100%; min-height:200px; display:flex; align-items:center;
    justify-content:center; border:1px dashed rgba(240,230,211,.12);
    border-radius:12px; padding:2rem; text-align:center;
    color:rgba(240,230,211,.3); font-size:.85rem; margin-top:.4rem;
}

/* DEMO CHIPS */
.demo-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:.5rem; }
.demo-label { font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:rgba(240,230,211,.4); white-space:nowrap; }
.demo-chip {
    display:inline-flex; align-items:center; gap:6px;
    padding:8px 14px; border:1px solid rgba(240,230,211,.15); border-radius:999px;
    font-size:.82rem; color:#f0e6d3; background:rgba(240,230,211,.04);
    white-space:nowrap;
}

/* DISHES GRID */
.dishes-section { margin-top:3rem; padding-top:2rem; border-top:1px solid rgba(240,230,211,.08); }
.dishes-title { font-size:.72rem; letter-spacing:.15em; text-transform:uppercase; color:rgba(240,230,211,.4); margin-bottom:1rem; }
.dish-tags { display:flex; flex-wrap:wrap; gap:8px; }
.dish-tag { padding:6px 14px; border:1px solid rgba(240,230,211,.15); border-radius:6px; font-size:.82rem; color:rgba(240,230,211,.7); }

/* FOOTER */
.footer {
    margin-top:3rem; padding-top:1.5rem; border-top:1px solid rgba(240,230,211,.08);
    font-size:.75rem; color:rgba(240,230,211,.3);
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem;
}
.footer a { color:rgba(240,230,211,.45); text-decoration:none; }
.footer a:hover { color:#e8673a; }

/* Streamlit overrides */
div[data-testid="stFileUploader"] > label { display:none; }
div[data-testid="stFileUploader"] section { background:transparent !important; border:none !important; padding:0 !important; }
div[data-testid="stFileUploader"] section > div { display:none; }
button[kind="primary"], div[data-testid="stButton"] button {
    background:#e8673a !important; border:none !important; color:#fff !important;
    border-radius:8px !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key in ("results", "inference_ms", "uploaded_img", "cam_img"):
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL (eager — so status badge is accurate)
# ══════════════════════════════════════════════════════════════════════════════

_, class_names, img_size, weights_loaded = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# NAV
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="navbar">
  <div class="nav-logo">🍲 Naija Eats · <span>AI</span></div>
  <div class="nav-pills">
    <a class="nav-pill" href="#">Classify</a>
    <a class="nav-pill" href="#">Model Info</a>
    <a class="nav-pill" href="#">All Classes ↗</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONTROLS
# ══════════════════════════════════════════════════════════════════════════════

col_tta, col_gcam, col_topk = st.columns([2, 2, 1])
with col_tta:
    use_tta  = st.toggle("Test-Time Augmentation", value=False,
                         help="Averages 4 augmented views: original, h-flip, brightness, center-crop")
with col_gcam:
    use_gcam = st.toggle("Grad-CAM++ Overlay", value=True,
                         help="Highlights the image regions that drove the prediction (GradCAM++)")
with col_topk:
    top_k    = st.selectbox("Top-K", [5, 3, 10], index=0, label_visibility="collapsed")

# Model status badge
dot_cls   = "dot-green" if weights_loaded else "dot-yellow"
ckpt_info = f"Checkpoint: {CKPT_PATH.name}" if CKPT_PATH else "No checkpoint found"
st.markdown(f"""
<div style="text-align:right; margin-top:-1rem; margin-bottom:1rem;">
  <span class="model-badge">
    <span class="{dot_cls}"></span>
    tf_efficientnetv2_m · {img_size}×{img_size} · {ckpt_info}
  </span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO + UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

hero_col, upload_col = st.columns([1.15, 0.85], gap="large")

with hero_col:
    st.markdown(f"""
<h1 class="hero-headline">
  Identify<br>any<br><em>Nigerian<br>dish</em><br>instantly.
</h1>
<p class="hero-sub">
  From jollof rice to egusi soup — our fine-tuned
  EfficientNetV2-M model returns a classification in milliseconds.
</p>
<div class="stats-row">
  <div class="stat-item">
    <span class="stat-value">{len(class_names)}</span>
    <span class="stat-label">Food Classes</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">80.8%</span>
    <span class="stat-label">Macro F1</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">&lt;200ms</span>
    <span class="stat-label">Inference</span>
  </div>
</div>
""", unsafe_allow_html=True)

with upload_col:
    st.markdown("""
<div class="upload-card">
  <div class="upload-icon">🍲</div>
  <div class="upload-title">Drop your dish here</div>
  <div class="upload-hint">Drag &amp; drop or browse — JPG, PNG, WEBP up to 10 MB</div>
  <div>
    <span class="fmt-badge">JPG</span>
    <span class="fmt-badge">PNG</span>
    <span class="fmt-badge">WEBP</span>
  </div>
</div>
""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.session_state.uploaded_img = pil_img
        # Reset stale results when a new image is uploaded
        st.session_state.results    = None
        st.session_state.cam_img    = None
        st.session_state.inference_ms = None

    if st.session_state.uploaded_img is not None:
        st.image(st.session_state.uploaded_img, use_container_width=True,
                 caption="Uploaded image", output_format="PNG")

        if st.button("🔍  Classify Dish", use_container_width=True):
            with st.spinner("Running inference…"):
                results, ms, cam_img = classify(
                    st.session_state.uploaded_img,
                    use_tta=use_tta,
                    top_k=top_k,
                    use_gcam=use_gcam,
                )
            st.session_state.results      = results
            st.session_state.inference_ms = ms
            st.session_state.cam_img      = cam_img

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS + GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.results:
    results   = st.session_state.results
    ms        = st.session_state.inference_ms
    cam_img   = st.session_state.cam_img
    top_dish  = results[0]["class"]
    top_conf  = results[0]["conf"]
    pct       = int(top_conf * 100)
    bar_w     = max(4, pct)

    top_k_rows = "".join(
        f'<div class="top-k-row">'
        f'<span class="top-k-name">{r["class"]}</span>'
        f'<span class="top-k-pct">{int(r["conf"]*100)}%</span>'
        f'</div>'
        for r in results
    )

    res_col, cam_col = st.columns([1, 1], gap="large")

    with res_col:
        mode_tag = f"{ms} ms · {'TTA ×4' if use_tta else 'single pass'}" if ms else "demo mode"
        st.markdown(f"""
<div class="result-card">
  <div class="result-label">Top Prediction · {mode_tag}</div>
  <div class="result-dish">{top_dish}</div>
  <div class="result-conf">{pct}% confidence</div>
  <div class="conf-bar-bg">
    <div class="conf-bar-fill" style="width:{bar_w}%"></div>
  </div>
  <div style="margin-top:.8rem; font-size:.72rem; letter-spacing:.1em;
       text-transform:uppercase; color:rgba(240,230,211,.4); margin-bottom:.5rem;">
    Top-{len(results)} Predictions
  </div>
  {top_k_rows}
</div>
""", unsafe_allow_html=True)

    with cam_col:
        if cam_img is not None:
            st.markdown('<div class="cam-label">Grad-CAM++ · What the model sees</div>',
                        unsafe_allow_html=True)
            st.image(cam_img, use_container_width=True, output_format="PNG")
            st.markdown(
                "<p style='font-size:.75rem; color:rgba(240,230,211,.35); margin-top:4px;'>"
                "🔴 High activation &nbsp;·&nbsp; 🔵 Low activation &nbsp;·&nbsp; "
                "GradCAM++ on last Conv2d in backbone</p>",
                unsafe_allow_html=True,
            )
        elif use_gcam:
            st.markdown(
                '<div class="cam-empty">'
                '🗺️ Grad-CAM unavailable<br>'
                '<span style="font-size:.75rem;">Load a checkpoint to enable overlays</span>'
                '</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# DEMO CHIPS
# ══════════════════════════════════════════════════════════════════════════════

chips = "".join(f'<span class="demo-chip">{ico} {name}</span>'
                for ico, name in DEMO_DISHES)
st.markdown(
    f'<div class="demo-row"><span class="demo-label">Try a demo →</span>{chips}</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SUPPORTED DISHES
# ══════════════════════════════════════════════════════════════════════════════

tags = "".join(f'<span class="dish-tag">{c}</span>' for c in class_names)
st.markdown(f"""
<div class="dishes-section">
  <div class="dishes-title">Supported Dishes</div>
  <div class="dish-tags">{tags}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  <span>tf_efficientnetv2_m · 5-fold CV · Macro F1 = 0.805 · PyTorch + timm + albumentations + Streamlit</span>
  <span>
    <a href="#">Training details ↗</a> &nbsp;
    <a href="#">Grad-CAM++ docs ↗</a> &nbsp;
    <a href="#">Improve model ↗</a>
  </span>
</div>
""", unsafe_allow_html=True)
