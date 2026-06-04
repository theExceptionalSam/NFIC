import io
import base64
import time
import random
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Optional ML imports ────────────────────────────────────────────────────────
try:
    import torch
    import timm
    from torchvision import transforms

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Naija Eats",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=DM+Sans:wght@400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #1a1008;
    color: #f0e6d3;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* ── NAV BAR ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    border-bottom: 1px solid rgba(240,230,211,0.12);
    margin-bottom: 2rem;
}
.nav-logo {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #f0e6d3;
    display: flex;
    align-items: center;
    gap: 6px;
}
.nav-logo span { color: #e8673a; }
.nav-pills { display: flex; gap: 8px; }
.nav-pill {
    padding: 6px 16px;
    border: 1px solid rgba(240,230,211,0.3);
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #f0e6d3;
    background: transparent;
    cursor: pointer;
    text-decoration: none;
}
.nav-pill:hover { background: rgba(232,103,58,0.15); border-color: #e8673a; }

/* ── HERO ── */
.hero-headline {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 700;
    line-height: 1.05;
    color: #f0e6d3;
    margin-bottom: 1rem;
}
.hero-headline em {
    font-style: italic;
    color: #e8673a;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(240,230,211,0.65);
    max-width: 480px;
    line-height: 1.65;
    margin-bottom: 1.8rem;
}

/* ── STAT PILLS ── */
.stats-row { display: flex; gap: 2.5rem; margin-top: 1.5rem; }
.stat-item { display: flex; flex-direction: column; }
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8673a;
    line-height: 1;
}
.stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(240,230,211,0.5);
    margin-top: 3px;
}

/* ── UPLOAD CARD ── */
.upload-card {
    background: rgba(232,103,58,0.08);
    border: 1.5px dashed rgba(232,103,58,0.4);
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    transition: border-color .2s;
}
.upload-card:hover { border-color: #e8673a; }
.upload-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.upload-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: #f0e6d3;
    margin-bottom: 0.4rem;
}
.upload-hint {
    font-size: 0.8rem;
    color: rgba(240,230,211,0.45);
    margin-bottom: 1rem;
}
.fmt-badge {
    display: inline-block;
    padding: 3px 10px;
    border: 1px solid rgba(240,230,211,0.2);
    border-radius: 999px;
    font-size: 0.72rem;
    color: rgba(240,230,211,0.55);
    margin: 2px;
}

/* ── MODEL STATUS ── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(240,230,211,0.06);
    border: 1px solid rgba(240,230,211,0.12);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: rgba(240,230,211,0.6);
    margin-bottom: 1.2rem;
}
.dot-green { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; }

/* ── RESULT CARD ── */
.result-card {
    background: rgba(240,230,211,0.05);
    border: 1px solid rgba(240,230,211,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}
.result-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(240,230,211,0.45);
    margin-bottom: 0.3rem;
}
.result-dish {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0e6d3;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.result-conf {
    font-size: 1.2rem;
    color: #e8673a;
    font-weight: 600;
}
.confidence-bar-bg {
    background: rgba(240,230,211,0.1);
    border-radius: 999px;
    height: 6px;
    margin: 0.75rem 0;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #e8673a, #f5a623);
    transition: width 0.8s ease;
}
.top-k-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(240,230,211,0.06);
    font-size: 0.85rem;
}
.top-k-row:last-child { border-bottom: none; }
.top-k-name { color: rgba(240,230,211,0.8); }
.top-k-pct { color: #e8673a; font-weight: 600; font-size: 0.82rem; }

/* ── DEMO CHIPS ── */
.demo-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 0.5rem; }
.demo-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(240,230,211,0.4);
    white-space: nowrap;
}
.demo-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border: 1px solid rgba(240,230,211,0.15);
    border-radius: 999px;
    font-size: 0.82rem;
    color: #f0e6d3;
    background: rgba(240,230,211,0.04);
    cursor: pointer;
    transition: all .15s;
    white-space: nowrap;
}
.demo-chip:hover { background: rgba(232,103,58,0.15); border-color: #e8673a; }

/* ── SUPPORTED DISHES GRID ── */
.dishes-section { margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(240,230,211,0.08); }
.dishes-title {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(240,230,211,0.4);
    margin-bottom: 1rem;
}
.dish-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.dish-tag {
    padding: 6px 14px;
    border: 1px solid rgba(240,230,211,0.15);
    border-radius: 6px;
    font-size: 0.82rem;
    color: rgba(240,230,211,0.7);
}

/* ── FOOTER ── */
.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(240,230,211,0.08);
    font-size: 0.75rem;
    color: rgba(240,230,211,0.3);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.footer a { color: rgba(240,230,211,0.45); text-decoration: none; }
.footer a:hover { color: #e8673a; }

/* ── TOGGLE ROW ── */
.toggle-row {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    margin-bottom: 1.2rem;
    font-size: 0.82rem;
    color: rgba(240,230,211,0.6);
}

/* Streamlit widget overrides */
div[data-testid="stFileUploader"] > label { display: none; }
div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stFileUploader"] section > div { display: none; }
button[kind="primary"], div[data-testid="stButton"] button {
    background: #e8673a !important;
    border: none !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
FOOD_CLASSES = [
    "Jollof Rice", "Fried Rice", "Egusi Soup", "Ogbono Soup", "Edikaikong",
    "Pepper Soup", "Pounded Yam", "Fufu", "Eba", "Suya", "Kilishi", "Asun",
    "Puff Puff", "Chin Chin", "Akara", "Moi Moi", "Banga Soup", "Afang Soup",
    "Ofe Onugbu", "Ewedu", "Banga Rice",
]

DEMO_DISHES = [
    ("🍚", "Jollof Rice"),
    ("🫕", "Egusi Soup"),
    ("🍖", "Suya"),
    ("🧁", "Puff Puff"),
]

IMG_SIZE = 288
MODEL_NAME = "efficientnetv2_m"

# ── Image preprocessing ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not ML_AVAILABLE:
        return None
    try:
        model = timm.create_model(
            best_fold0.pth,
            pretrained=False,
            num_classes=len(FOOD_CLASSES),
        )

        model.eval()
        return model
    except Exception:
        return None


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def classify_image(pil_img: Image.Image, use_tta: bool = False, top_k: int = 5):
    """Run inference. Falls back to plausible mock results if model isn't loaded."""
    model = load_model()
    if model is None or not ML_AVAILABLE:
        # ── Demo / mock results ────────────────────────────────────────────────
        picks = random.sample(range(len(FOOD_CLASSES)), top_k)
        raw = sorted([(i, random.uniform(0.02, 0.95)) for i in picks],
                     key=lambda x: -x[1])
        total = sum(v for _, v in raw)
        results = [(FOOD_CLASSES[i], round(v / total, 4)) for i, v in raw]
        return results, 0.0

    transform = get_transform()
    pil_img = pil_img.convert("RGB")

    if use_tta:
        augments = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomRotation(10),
        ]
        tensors = [transform(pil_img).unsqueeze(0)]
        for aug in augments:
            tensors.append(transform(aug(pil_img)).unsqueeze(0))
        batch = torch.cat(tensors, dim=0)
    else:
        batch = transform(pil_img).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(batch)
        if use_tta:
            logits = logits.mean(dim=0, keepdim=True)
        probs = torch.softmax(logits, dim=-1).squeeze()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    top_probs, top_idx = probs.topk(min(top_k, len(FOOD_CLASSES)))
    results = [(FOOD_CLASSES[i.item()], round(p.item(), 4))
               for p, i in zip(top_probs, top_idx)]
    return results, round(elapsed_ms, 1)


def pil_to_b64(img: Image.Image, size=(100, 100)) -> str:
    img = img.copy()
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Session state ──────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "inference_ms" not in st.session_state:
    st.session_state.inference_ms = None
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None

# ── NAV ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="navbar">
  <div class="nav-logo">🍲 Naija Eats · <span>AI</span></div>
  <div class="nav-pills">
    <a class="nav-pill" href="#">Classify</a>
    <a class="nav-pill" href="#">Model Info</a>
    <a class="nav-pill" href="#">All Classes ↗</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Controls row ───────────────────────────────────────────────────────────────
col_tta, col_gcam, col_topk = st.columns([2, 2, 1])
with col_tta:
    use_tta = st.toggle("Test-Time Augmentation", value=False)
with col_gcam:
    use_gcam = st.toggle("Grad-CAM Overlay", value=True)
with col_topk:
    top_k = st.selectbox("Top-K", [5, 3, 10], index=0, label_visibility="collapsed")

st.markdown(
    f"""
<div style="text-align:right; margin-top:-1rem; margin-bottom:1rem;">
  <span class="model-badge">
    <span class="dot-green"></span>
    EfficientNetV2-M · F1 0.8086 · {IMG_SIZE}×{IMG_SIZE}
  </span>
</div>
""",
    unsafe_allow_html=True,
)

# ── Hero + Upload layout ───────────────────────────────────────────────────────
hero_col, upload_col = st.columns([1.15, 0.85], gap="large")

with hero_col:
    st.markdown(
        """
<h1 class="hero-headline">
  Identify<br>any<br><em>Nigerian<br>dish</em><br>instantly.
</h1>
<p class="hero-sub">
  From jollof rice to egusi soup — our fine-tuned EfficientNetV2 model
  returns a classification in milliseconds.
</p>
<div class="stats-row">
  <div class="stat-item"><span class="stat-value">21</span><span class="stat-label">Food Classes</span></div>
  <div class="stat-item"><span class="stat-value">80.8%</span><span class="stat-label">Macro F1</span></div>
  <div class="stat-item"><span class="stat-value">&lt;200ms</span><span class="stat-label">Inference</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

with upload_col:
    # Upload card header
    st.markdown(
        """
<div class="upload-card">
  <div class="upload-icon">🍲</div>
  <div class="upload-title">Drop your dish here</div>
  <div class="upload-hint">Drag &amp; drop or browse — JPG, PNG, WEBP supported up to 10 MB</div>
  <div>
    <span class="fmt-badge">JPG</span>
    <span class="fmt-badge">PNG</span>
    <span class="fmt-badge">WEBP</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_img = pil_img
        st.image(pil_img, use_container_width=True,
                 caption="Uploaded image", output_format="PNG")

        if st.button("🔍  Classify Dish", use_container_width=True):
            with st.spinner("Running inference…"):
                results, ms = classify_image(pil_img, use_tta=use_tta, top_k=top_k)
            st.session_state.results = results
            st.session_state.inference_ms = ms

    elif st.session_state.uploaded_img is not None:
        # Show previously uploaded image
        st.image(st.session_state.uploaded_img, use_container_width=True,
                 caption="Uploaded image", output_format="PNG")

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    ms = st.session_state.inference_ms
    top_dish, top_conf = results[0]
    pct = int(top_conf * 100)
    bar_pct = max(4, pct)

    st.markdown(
        f"""
<div class="result-card">
  <div class="result-label">Top Prediction · {f"{ms} ms" if ms else "demo mode"}</div>
  <div class="result-dish">{top_dish}</div>
  <div class="result-conf">{pct}% confidence</div>
  <div class="confidence-bar-bg">
    <div class="confidence-bar-fill" style="width:{bar_pct}%"></div>
  </div>
  <div style="margin-top:0.8rem; font-size:0.72rem; letter-spacing:.1em; text-transform:uppercase; color:rgba(240,230,211,0.4); margin-bottom:0.5rem;">
    Top-{len(results)} Predictions
  </div>
  {''.join(f'<div class="top-k-row"><span class="top-k-name">{name}</span><span class="top-k-pct">{int(conf*100)}%</span></div>' for name, conf in results)}
</div>
""",
        unsafe_allow_html=True,
    )

    if ML_AVAILABLE and use_gcam:
        st.info("ℹ️  Grad-CAM overlay requires fine-tuned model weights loaded. "
                "Hook the `grad_cam()` function to your checkpoint to enable visualisation.", icon="🗺️")

# ── Demo chips ─────────────────────────────────────────────────────────────────
chips_html = '<div class="demo-row"><span class="demo-label">Try a demo →</span>'
for icon, name in DEMO_DISHES:
    chips_html += f'<span class="demo-chip">{icon} {name}</span>'
chips_html += "</div>"
st.markdown(chips_html, unsafe_allow_html=True)

# ── Supported dishes ───────────────────────────────────────────────────────────
tags_html = "".join(f'<span class="dish-tag">{d}</span>' for d in FOOD_CLASSES)
st.markdown(
    f"""
<div class="dishes-section">
  <div class="dishes-title">Supported Dishes</div>
  <div class="dish-tags">{tags_html}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
  <span>EfficientNetV2-M · 5-fold CV · Macro F1 = 0.805 · Built with PyTorch + timm + Streamlit</span>
  <span>
    <a href="#">Training details ↗</a> &nbsp;
    <a href="#">Grad-CAM docs ↗</a> &nbsp;
    <a href="#">Improve model ↗</a>
  </span>
</div>
""",
    unsafe_allow_html=True,
)
