import os
import sys
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
from torchvision import transforms
from scipy import ndimage

# --- Path Setup ---
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
UNET_CKPT      = os.path.join(CHECKPOINT_DIR, 'best_unet.pth')
SA_UNET_CKPT   = os.path.join(CHECKPOINT_DIR, 'best_sa_unet.pth')
CLASS_CKPT     = os.path.join(CHECKPOINT_DIR, 'best_classifier.pth')

from src.models.unet import UNet
from src.models.self_attention_unet import SelfAttentionUNet
from src.models.classifier import LesionClassifier

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate models
unet_model       = UNet(n_channels=3, n_classes=1).to(device)
sa_unet_model    = SelfAttentionUNet(n_channels=3, n_classes=1).to(device)
classifier_model = LesionClassifier(num_classes=3).to(device)

def load_weights(model, path, label="model"):
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            print(f"[OK]   Loaded {label}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load {label}: {e}")
            return False
    print(f"[WARN] Not found: {path}")
    return False

unet_loaded       = load_weights(unet_model,       UNET_CKPT,   "U-Net")
sa_unet_loaded    = load_weights(sa_unet_model,    SA_UNET_CKPT,"SA-UNet")
classifier_loaded = load_weights(classifier_model, CLASS_CKPT,  "Classifier")

# --- Transforms ---
transform_seg = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
transform_class = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

classes = ["Common Nevus", "Atypical Nevus", "Melanoma"]

# ─────────────────────────────────────────────────────────
#  DERMOSCOPY VALIDATOR
#  Returns (is_skin: bool, confidence: float 0-1)
#
#  A dermoscopic skin-lesion image typically has:
#   1. A significant portion of skin-tone pixels (peach/tan/brown range)
#   2. A focal dark region (the lesion) in or near the image centre
#   3. Low sky-blue / green-grass pixel fraction (rules out nature photos)
#   4. Not dominated by grey/silver (rules out phones, metal objects)
# ─────────────────────────────────────────────────────────
def is_dermoscopic_image(img_np):
    """
    Multi-factor dermoscopy validator.
    Returns (is_valid: bool, score: float, reason: str)
    """
    img = img_np.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # ── Factor 1: Skin-tone pixel fraction ──────────────────────────────────
    # Skin tones: R > G > B  AND  R in [0.35, 0.98]  AND  (R-B) > 0.08
    skin_mask = (
        (r > g) & (g >= b) &
        (r > 0.30) & (r < 0.98) &
        ((r - b) > 0.05) &
        (r < 0.97) & (g < 0.90) & (b < 0.85)
    )
    skin_frac = float(np.mean(skin_mask))

    # ── Factor 2: Dark-lesion region ────────────────────────────────────────
    # A lesion: dark (all channels < 0.45), and forms a connected blob
    dark_mask  = (r < 0.45) & (g < 0.45) & (b < 0.45)
    dark_frac  = float(np.mean(dark_mask))
    # Check if dark blob is concentrated (lesion) vs scattered (shadows in photos)
    dark_uint8 = dark_mask.astype(np.uint8)
    labeled, num_blobs = ndimage.label(dark_uint8)
    largest_blob_size = 0
    if num_blobs > 0:
        blob_sizes = ndimage.sum(dark_uint8, labeled, range(1, num_blobs + 1))
        largest_blob_size = float(np.max(blob_sizes)) / (h * w)

    # ── Factor 3: Reject bright sky-blue (outdoor photos) ───────────────────
    sky_mask  = (b > 0.55) & (b > r + 0.10) & (b > g - 0.05)
    sky_frac  = float(np.mean(sky_mask))

    # ── Factor 4: Reject green grass / foliage ──────────────────────────────
    grass_mask = (g > r + 0.05) & (g > b + 0.05) & (g > 0.30)
    grass_frac = float(np.mean(grass_mask))

    # ── Factor 5: Reject grey/silver (electronics, phones) ──────────────────
    grey_mask  = (np.abs(r - g) < 0.06) & (np.abs(g - b) < 0.06) & (r > 0.25)
    grey_frac  = float(np.mean(grey_mask))

    # ── Factor 6: Check saturation (dermoscopy = moderate saturation) ────────
    max_rgb    = np.maximum(np.maximum(r, g), b)
    min_rgb    = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
    mean_sat   = float(np.mean(saturation))

    # ── Decision rules ────────────────────────────────────────────────────────
    # Hard rejections
    if sky_frac > 0.25:
        return False, 0.0, "Outdoor/sky image detected"
    if grass_frac > 0.30:
        return False, 0.0, "Nature/grass image detected"
    if grey_frac > 0.55:
        return False, 0.0, "Electronic/grey object detected"
    if mean_sat > 0.75:
        return False, 0.0, "Highly saturated non-skin image"

    # Scoring: needs enough skin + focal dark area
    score = 0.0
    score += min(skin_frac / 0.25, 1.0) * 0.50       # up to 0.50 for skin pixels
    score += min(largest_blob_size / 0.08, 1.0) * 0.30 # up to 0.30 for focal lesion blob
    score += min(dark_frac / 0.15, 1.0) * 0.10        # up to 0.10 for dark area
    score -= sky_frac * 0.30
    score -= grass_frac * 0.30
    score -= grey_frac * 0.20

    # Require both meaningful skin AND a focal dark lesion blob
    threshold = 0.35
    is_valid = (score >= threshold) and (skin_frac > 0.08) and (dark_frac > 0.02)

    reason = (
        f"skin={skin_frac:.2f}, dark_blob={largest_blob_size:.3f}, "
        f"sky={sky_frac:.2f}, grass={grass_frac:.2f}, grey={grey_frac:.2f}, score={score:.2f}"
    )
    return is_valid, score, reason


# --- Classifier prediction is now handled directly by the neural network ---
def tensor_to_base64_mask(tensor_mask):
    mask_np = tensor_mask.squeeze().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    img = Image.fromarray(mask_np, mode='L')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def blank_mask_b64(size=256):
    """Return a solid black mask."""
    img = Image.new('L', (size, size), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    global unet_loaded, sa_unet_loaded, classifier_loaded

    if not unet_loaded:       unet_loaded       = load_weights(unet_model,       UNET_CKPT,   "U-Net")
    if not sa_unet_loaded:    sa_unet_loaded    = load_weights(sa_unet_model,    SA_UNET_CKPT,"SA-UNet")
    if not classifier_loaded: classifier_loaded = load_weights(classifier_model, CLASS_CKPT,  "Classifier")

    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)

    # ── Step 1: Validate – is this a dermoscopic skin image? ──────────────
    is_skin, skin_score, reason = is_dermoscopic_image(image_np)
    print(f"[VALIDATE] is_skin={is_skin}, score={skin_score:.2f}, {reason}")

    if not is_skin:
        blank = blank_mask_b64()
        return JSONResponse(content={
            "unet_mask":             f"data:image/png;base64,{blank}",
            "sa_unet_mask":          f"data:image/png;base64,{blank}",
            "predicted_class_probs": {c: 0.0 for c in classes},
            "is_skin_image":         False,
            "skin_score":            round(skin_score, 3),
            "models_loaded": {
                "unet":       unet_loaded,
                "sa_unet":    sa_unet_loaded,
                "classifier": classifier_loaded
            }
        })

    # ── Step 2: Segmentation ──────────────────────────────────────────────
    img_seg = transform_seg(image).unsqueeze(0).to(device)
    with torch.no_grad():
        unet_pred    = unet_model(img_seg)
        sa_unet_pred = sa_unet_model(img_seg)

    unet_mask_b64    = tensor_to_base64_mask(unet_pred > 0.5)
    sa_unet_mask_b64 = tensor_to_base64_mask(sa_unet_pred > 0.5)

    # ── Step 3: Neural Network Classification ─────────────────────────────
    # Classification Inference
    img_class = transform_class(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        class_out = classifier_model(img_class)
        # Apply temperature scaling to make predictions more confident
        temperature = 0.5
        probabilities = torch.nn.functional.softmax(class_out / temperature, dim=1)[0]
        
    class_probs = {classes[i]: round(float(prob.item()) * 100, 2) for i, prob in enumerate(probabilities)}
    print(f"[CLASS] NN probs: {class_probs}")

    return JSONResponse(content={
        "unet_mask":             f"data:image/png;base64,{unet_mask_b64}",
        "sa_unet_mask":          f"data:image/png;base64,{sa_unet_mask_b64}",
        "predicted_class_probs": class_probs,
        "is_skin_image":         True,
        "skin_score":            round(skin_score, 3),
        "models_loaded": {
            "unet":       unet_loaded,
            "sa_unet":    sa_unet_loaded,
            "classifier": classifier_loaded
        }
    })
