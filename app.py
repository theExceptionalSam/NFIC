import io
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS — pulled directly from notebook CFG + discover_dataset
# ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "tf_efficientnetv2_m"
NUM_CLASSES = 21
IMG_SIZE    = 224
CHECKPOINT  = "best_fold0.pth"

# ImageNet normalisation (standard for timm models)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Exact class order from notebook (sorted alphabetically by discover_dataset)
CLASS_NAMES = [
    "Abacha and Ugba(african salad)",   # 0
    "Akara and Eko",                    # 1
    "Akara and Eko-Akamu",              # 2
    "Amala and Ewedu-Gbegiri",          # 3
    "Amala and Gbegiri- Ewedu",         # 4
    "Asaro",                            # 5
    "Boli(bole)",                       # 6
    "Chin Chin",                        # 7
    "Egusi Soup",                       # 8
    "Ewa-Agoyin",                       # 9
    "Fried Plantains (dodo)",           # 10
    "Jollof Rice",                      # 11
    "Meat-pie",                         # 12
    "Moin-Moin",                        # 13
    "Nkwobi",                           # 14
    "Okro Soup",                        # 15
    "Pepper Soup",                      # 16
    "Pepper-Soup",                      # 17
    "Puff-Puff",                        # 18
    "Suya",                             # 19
    "Vegetable Soup",                   # 20
]

FOOD_EMOJIS = {
    "Abacha and Ugba(african salad)": "🥗",
    "Akara and Eko":                  "🫓",
    "Akara and Eko-Akamu":            "🫓",
    "Amala and Ewedu-Gbegiri":        "🍲",
    "Amala and Gbegiri- Ewedu":       "🍲",
    "Asaro":                          "🍠",
    "Boli(bole)":                     "🍌",
    "Chin Chin":                      "🍪",
    "Egusi Soup":                     "🥣",
    "Ewa-Agoyin":                     "🫘",
    "Fried Plantains (dodo)":         "🍟",
    "Jollof Rice":                    "🍚",
    "Meat-pie":                       "🥧",
    "Moin-Moin":                      "🫔",
    "Nkwobi":                         "🍖",
    "Okro Soup":                      "🥘",
    "Pepper Soup":                    "🌶️",
    "Pepper-Soup":                    "🌶️",
    "Puff-Puff":                      "🔮",
    "Suya":                           "🍢",
    "Vegetable Soup":                 "🥬",
}

# Per-class F1 from OOF (for info display in sidebar)
CLASS_F1 = {
    "Abacha and Ugba(african salad)": 0.974, "Akara and Eko": 0.012,
    "Akara and Eko-Akamu": 0.463, "Amala and Ewedu-Gbegiri": 0.295,
    "Amala and Gbegiri- Ewedu": 0.350, "Asaro": 0.987, "Boli(bole)": 0.989,
    "Chin Chin": 0.983, "Egusi Soup": 0.968, "Ewa-Agoyin": 0.986,
    "Fried Plantains (dodo)": 0.979, "Jollof Rice": 0.983, "Meat-pie": 1.000,
    "Moin-Moin": 0.969, "Nkwobi": 0.976, "Okro Soup": 0.952,
    "Pepper Soup": 0.536, "Pepper-Soup": 0.292, "Puff-Puff": 0.997,
    "Suya": 0.993, "Vegetable Soup": 0.976,
}


# ─────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE — mirrors NigerianFoodClassifier from notebook
# ─────────────────────────────────────────────────────────────────
class NigerianFoodClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int,
                 pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool='avg',
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


# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍜 Nigerian Food Classifier",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #111827 45%, #1a0533 100%);
    border: 1px solid #2d1f4e;
    border-radius: 20px;
    padding: 2.8rem 2.5rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(139,92,246,0.12) 0%, transparent 65%),
                radial-gradient(ellipse at 30% 50%, rgba(245,158,11,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero h1 { color: #f5d87e; font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero .sub { color: #9ca3af; font-size: 0.95rem; margin-top: .5rem; }
.hero .badges { margin-top: 1rem; }
.badge {
    display: inline-block;
    background: rgba(139,92,246,0.2);
    border: 1px solid rgba(139,92,246,0.4);
    color: #c4b5fd;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    padding: .25rem .75rem;
    margin: .2rem;
}

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, #111827, #1e1033);
    border: 1px solid #2d1f4e;
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.pred-card .emoji  { font-size: 4rem; line-height: 1; }
.pred-card .name   { color: #f5d87e; font-size: 1.7rem; font-weight: 700; margin-top: .5rem; }
.pred-card .conf   { color: #9ca3af; font-size: .9rem; margin-top: .25rem; }
.pred-card .f1tag  {
    display: inline-block;
    margin-top: .7rem;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.35);
    color: #6ee7b7;
    border-radius: 999px;
    font-size: .75rem;
    padding: .2rem .7rem;
}

/* Section header */
.sh {
    font-size: .95rem; font-weight: 700;
    color: #a78bfa;
    border-left: 3px solid #f5d87e;
    padding-left: .65rem;
    margin: 1.4rem 0 .7rem;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* Bar row */
.bar-row {
    display: flex; align-items: center; gap: .6rem;
    margin-bottom: .45rem;
}
.bar-label { color: #d1d5db; font-size: .85rem; min-width: 220px; }
.bar-outer {
    flex: 1; height: 10px; background: #1f2937;
    border-radius: 999px; overflow: hidden;
}
.bar-inner {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #8b5cf6, #f59e0b);
}
.bar-pct { color: #9ca3af; font-size: .8rem; min-width: 45px; text-align: right; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1a 0%, #0f1929 100%) !important;
    border-right: 1px solid #1f2d44;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] h2 { color: #f5d87e !important; font-size: 1rem !important; }

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #0d1525;
    border: 2px dashed #2d3f5e;
    border-radius: 14px;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg,#8b5cf6,#f59e0b) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#4c1d95,#1e3a5f);
    color: #f5d87e !important; border: none;
    border-radius: 10px; font-weight: 700;
    padding: .55rem 1.5rem; width: 100%;
    letter-spacing: .02em;
}
.stButton > button:hover { opacity: .85; }

/* Expander */
details { background: #0d1525 !important; border: 1px solid #1f2d44 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


# ─────────────────────────────────────────────────────────────────
#  MODEL LOAD
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mn  = ckpt.get("model_name", MODEL_NAME)
    nc  = ckpt.get("num_classes", NUM_CLASSES)
    model = NigerianFoodClassifier(mn, nc, pretrained=False, dropout=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("val_f1", None), ckpt.get("epoch", None)


# ─────────────────────────────────────────────────────────────────
#  GRAD-CAM  (hooks on backbone's last conv block)
# ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.grads = None
        self.acts  = None
        # EfficientNetV2: last block of model.backbone.blocks
        target = list(model.backbone.blocks)[-1][-1]
        # use the last conv available in that sub-block
        for name in ("conv_pwl", "conv_dw", "conv_exp"):
            if hasattr(target, name):
                layer = getattr(target, name)
                break
        else:
            layer = target
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out): self.acts  = out.detach()
    def _bwd(self, _m, _i, out): self.grads = out[0].detach()

    def generate(self, tensor: torch.Tensor, cls_idx: int) -> np.ndarray:
        self.model.zero_grad()
        t = tensor.clone().requires_grad_(True)
        logits = self.model(t)
        logits[0, cls_idx].backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.acts).sum(dim=1).squeeze()
        cam = F.relu(cam)
        if cam.max() > 0: cam = cam / cam.max()
        return cam.cpu().numpy()


def blend_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    resized  = cv2.resize(cam, (image.width, image.height))
    heatmap  = (cm.jet(resized)[:, :, :3] * 255).astype(np.uint8)
    orig     = np.array(image.convert("RGB"))
    return (alpha * heatmap + (1 - alpha) * orig).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    ckpt_path = st.text_input("📁 Checkpoint path", value=CHECKPOINT,
                               help="Path to best_fold0.pth relative to app.py")

    st.markdown("### 🔥 Grad-CAM")
    show_gcam   = st.toggle("Enable Grad-CAM", value=True)
    cam_alpha   = st.slider("Heatmap blend", 0.2, 0.8, 0.48, 0.02)
    cam_target  = st.selectbox("Visualise class", ["Top prediction"] + CLASS_NAMES)

    st.markdown("### 📊 Results")
    top_k          = st.slider("Top-K predictions", 3, 10, 5)
    show_all_probs = st.toggle("Show full probability table", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.78rem;color:#6b7280;line-height:1.7'>
    <b style='color:#a78bfa'>Model</b> tf_efficientnetv2_m<br>
    <b style='color:#a78bfa'>Classes</b> 21 Nigerian foods<br>
    <b style='color:#a78bfa'>Input</b> 224 × 224 px<br>
    <b style='color:#a78bfa'>OOF Accuracy</b> 87.68%<br>
    <b style='color:#a78bfa'>Macro F1</b> 0.7933
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🏷️ All 21 classes"):
        for name in CLASS_NAMES:
            f1  = CLASS_F1.get(name, 0)
            bar = int(f1 * 10)
            st.markdown(
                f"{FOOD_EMOJIS.get(name,'🍽️')} **{name}**  \n"
                f"<small style='color:#6b7280'>OOF F1: {f1:.3f} {'█'*bar}{'░'*(10-bar)}</small>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🍜 Nigerian Food Classifier</h1>
  <p class="sub">
    Upload a photo of any Nigerian dish or snack — the AI identifies it instantly<br>
    with visual Grad-CAM explainability so you can see <em>why</em> it decided.
  </p>
  <div class="badges">
    <span class="badge">EfficientNetV2-M</span>
    <span class="badge">21 Food Classes</span>
    <span class="badge">87.7% Accuracy</span>
    <span class="badge">Grad-CAM</span>
    <span class="badge">PyTorch + timm</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────
model_ok = False
with st.spinner("⏳ Loading model weights…"):
    try:
        model, saved_f1, saved_ep = load_model(ckpt_path)
        model_ok = True
        f1_str   = f"F1 {saved_f1:.4f}" if saved_f1 else ""
        ep_str   = f" · Epoch {saved_ep}" if saved_ep else ""
        st.success(f"✅ `{ckpt_path}` loaded  {f1_str}{ep_str}")
    except FileNotFoundError:
        st.error(f"❌ `{ckpt_path}` not found. Place `best_fold0.pth` beside `app.py`.")
    except Exception as e:
        st.error(f"❌ Load error: {e}")


# ─────────────────────────────────────────────────────────────────
#  FILE UPLOADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sh">📤 Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop or click — JPG / PNG / WEBP",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────────────────────────
#  INFERENCE & DISPLAY
# ─────────────────────────────────────────────────────────────────
if uploaded and model_ok:
    image  = Image.open(uploaded).convert("RGB")
    tensor = preprocess(image)

    with st.spinner("🔍 Analysing image…"):
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()

    top_idx   = probs.argsort()[::-1][:top_k]
    top_probs = probs[top_idx]
    top_names = [CLASS_NAMES[i] for i in top_idx]

    best_name  = top_names[0]
    best_conf  = top_probs[0]
    best_emoji = FOOD_EMOJIS.get(best_name, "🍽️")
    best_f1    = CLASS_F1.get(best_name, 0)

    # ── Two-column layout ────────────────────────────────────
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="sh">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col_r:
        st.markdown('<div class="sh">🎯 Top Prediction</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-card">
          <div class="emoji">{best_emoji}</div>
          <div class="name">{best_name}</div>
          <div class="conf">Confidence: <b style="color:#f5d87e">{best_conf*100:.1f}%</b></div>
          <div class="f1tag">OOF F1 Score: {best_f1:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="sh">📊 Top {top_k} Results</div>', unsafe_allow_html=True)
        for rank, (name, conf) in enumerate(zip(top_names, top_probs)):
            emoji    = FOOD_EMOJIS.get(name, "🍽️")
            pct      = conf * 100
            bar_w    = int(pct)
            r_colour = "#f5d87e" if rank == 0 else "#6b7280"
            st.markdown(f"""
            <div class="bar-row">
              <div class="bar-label">
                <span style="color:{r_colour};font-weight:700">#{rank+1}</span>
                {emoji} {name}
              </div>
              <div class="bar-outer">
                <div class="bar-inner" style="width:{bar_w}%"></div>
              </div>
              <div class="bar-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Grad-CAM ─────────────────────────────────────────────
    if show_gcam:
        st.markdown('<div class="sh">🔥 Grad-CAM Explainability</div>', unsafe_allow_html=True)
        try:
            gcam = GradCAM(model)
            cls_idx = (CLASS_NAMES.index(cam_target)
                       if cam_target != "Top prediction"
                       else int(top_idx[0]))
            cls_label = CLASS_NAMES[cls_idx]

            cam_map = gcam.generate(preprocess(image), cls_idx)
            overlay = blend_cam(image, cam_map, alpha=cam_alpha)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original**")
                st.image(image, use_container_width=True)
            with c2:
                st.markdown("**Activation heatmap**")
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor("#0d1525")
                ax.set_facecolor("#0d1525")
                resized_cam = cv2.resize(cam_map, (image.width, image.height))
                im  = ax.imshow(resized_cam, cmap="jet")
                ax.axis("off")
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color="#9ca3af")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#9ca3af")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight",
                            facecolor="#0d1525", dpi=120)
                plt.close(fig)
                st.image(buf.getvalue(), use_container_width=True)
            with c3:
                st.markdown(f"**Overlay** · *{cls_label}*")
                st.image(overlay, use_container_width=True)

            st.caption(
                f"🔍 Highlighted regions drove the model toward **{cls_label}** "
                f"(confidence {probs[cls_idx]*100:.1f}%). "
                "Red = high activation · Blue = low activation."
            )
        except Exception as e:
            st.warning(f"⚠️ Grad-CAM could not run: {e}")

    # ── Full probability table ────────────────────────────────
    if show_all_probs:
        st.markdown('<div class="sh">📋 All 21 Class Probabilities</div>', unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame({
            "Class": [f"{FOOD_EMOJIS.get(c,'🍽️')} {c}" for c in CLASS_NAMES],
            "Probability (%)": np.round(probs * 100, 3),
            "OOF F1": [CLASS_F1.get(c, 0) for c in CLASS_NAMES],
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, height=420)

elif not model_ok:
    st.info("💡 Fix the checkpoint path in the sidebar, then upload an image.")

else:
    # ── Placeholder landing ───────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 1rem;color:#4b5563">
      <div style="font-size:5rem">📸</div>
      <p style="font-size:1.15rem;color:#6b7280;margin-top:1rem">
        Upload a photo of Nigerian food or snack to get started
      </p>
      <p style="font-size:.85rem;color:#374151">
        Jollof Rice · Suya · Puff-Puff · Egusi Soup · Chin Chin · and 16 more
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">🏷️ All Supported Classes</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        f1  = CLASS_F1.get(name, 0)
        cols[i % 3].markdown(
            f"{FOOD_EMOJIS.get(name,'🍽️')} **{name}**  \n"
            f"<small style='color:#4b5563'>F1: {f1:.3f}</small>",
            unsafe_allow_html=True,
        )
