"""
Synthetic Dermoscopy Dataset Generator
Creates clinically-motivated synthetic images for 3 classes:
  0 = Common Nevus
  1 = Atypical Nevus
  2 = Melanoma
"""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import os, random

np.random.seed(42)
random.seed(42)

IMG_SIZE = 224
CLASSES  = ["Common_Nevus", "Atypical_Nevus", "Melanoma"]
N_PER_CLASS_TRAIN = 400
N_PER_CLASS_VAL   = 80

# ─── Skin background ────────────────────────────────────────────────────────
def gen_skin_bg(size):
    """Generate a skin-tone background with slight vignette."""
    base_r = np.random.randint(200, 235)
    base_g = np.random.randint(165, 195)
    base_b = np.random.randint(145, 175)
    img = np.ones((size, size, 3), dtype=np.float32)
    img[:,:,0] = base_r
    img[:,:,1] = base_g
    img[:,:,2] = base_b
    # Subtle skin texture
    noise = np.random.normal(0, 6, (size, size, 3))
    img = np.clip(img + noise, 0, 255)
    return img.astype(np.uint8)

# ─── Shape generation ────────────────────────────────────────────────────────
def gen_ellipse_mask(size, cx, cy, rx, ry, angle_deg=0):
    """Create an ellipse mask."""
    Y, X = np.ogrid[:size, :size]
    angle = np.radians(angle_deg)
    xr = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
    yr = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
    mask = (xr / rx)**2 + (yr / ry)**2 <= 1.0
    return mask

def gen_irregular_mask(size, cx, cy, base_r, n_bumps=8, irregularity=0.4):
    """Create an irregular blob by perturbing a circle with sinusoidal bumps."""
    angles = np.linspace(0, 2*np.pi, 360)
    bump_freqs  = np.random.randint(2, n_bumps+1, size=5)
    bump_amps   = np.random.uniform(0, irregularity * base_r, size=5)
    bump_phases = np.random.uniform(0, 2*np.pi, size=5)
    r_vals = base_r * np.ones(360)
    for f, a, p in zip(bump_freqs, bump_amps, bump_phases):
        r_vals += a * np.sin(f * angles + p)
    r_vals = np.clip(r_vals, base_r * 0.4, base_r * 1.6)
    Y, X = np.ogrid[:size, :size]
    mask = np.zeros((size, size), bool)
    for i, ang in enumerate(angles):
        px = int(cx + r_vals[i] * np.cos(ang))
        py = int(cy + r_vals[i] * np.sin(ang))
        if 0 <= px < size and 0 <= py < size:
            # fill inside by drawing polygon
            pass
    # Use a different approach: draw all points inside the polar contour
    pts_x = (cx + r_vals * np.cos(angles)).astype(int)
    pts_y = (cy + r_vals * np.sin(angles)).astype(int)
    pts = np.stack([pts_x, pts_y], axis=1)
    pil_mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(pil_mask)
    draw.polygon([tuple(p) for p in pts], fill=255)
    mask = np.array(pil_mask) > 127
    return mask

# ─── Texture / colour fills ─────────────────────────────────────────────────
def apply_nevus_colour(img, mask, dark_level):
    """Uniform tan/brown fill for common nevus."""
    # Typical nevus: medium brown, uniform
    r = int(np.random.uniform(90, 140))
    g = int(np.random.uniform(60, 100))
    b = int(np.random.uniform(40, 80))
    noise = np.random.normal(0, 8, img.shape)
    lesion_img = img.copy().astype(np.float32)
    lesion_img[mask, 0] = np.clip(r + noise[mask, 0], 20, 200)
    lesion_img[mask, 1] = np.clip(g + noise[mask, 1], 10, 150)
    lesion_img[mask, 2] = np.clip(b + noise[mask, 2], 10, 120)
    return lesion_img.astype(np.uint8)

def apply_atypical_colour(img, mask, dark_level):
    """Multi-shade brown fill for atypical nevus."""
    lesion_img = img.copy().astype(np.float32)
    # Create 2-3 colour zones inside the lesion
    n_zones = np.random.randint(2, 4)
    zone_map = np.zeros_like(mask, dtype=int)
    pts = np.argwhere(mask)
    if len(pts) > 0:
        from scipy.ndimage import label
        # Random zone seed assignment
        rand_field = np.random.random(img.shape[:2]) * mask
        zone_map = (rand_field * n_zones).astype(int)
    colours = [
        (np.random.randint(70,130), np.random.randint(45, 90), np.random.randint(30, 70)),
        (np.random.randint(110,160),np.random.randint(70,110), np.random.randint(40, 80)),
        (np.random.randint(40, 90), np.random.randint(25, 60), np.random.randint(15, 50)),
    ]
    for z, (r, g, b) in enumerate(colours[:n_zones]):
        zmask = mask & (zone_map == z)
        if zmask.sum() == 0: continue
        noise = np.random.normal(0, 10, img.shape)
        lesion_img[zmask, 0] = np.clip(r + noise[zmask, 0], 20, 220)
        lesion_img[zmask, 1] = np.clip(g + noise[zmask, 1], 10, 160)
        lesion_img[zmask, 2] = np.clip(b + noise[zmask, 2], 10, 130)
    return lesion_img.astype(np.uint8)

def apply_melanoma_colour(img, mask, dark_level):
    """
    Dark, multicolour fill for melanoma with:
    - Very dark brown / black core
    - Grey-blue veil patches
    - White regression patches (sometimes)
    - Red/pink areas
    """
    lesion_img = img.copy().astype(np.float32)
    size = img.shape[0]
    # Create sub-regions
    rand_field = np.random.random(img.shape[:2])
    # Core: dark
    core = mask & (rand_field < 0.45)
    # Blue-white veil
    veil = mask & (rand_field >= 0.45) & (rand_field < 0.65)
    # Regression/white
    regress = mask & (rand_field >= 0.65) & (rand_field < 0.75)
    # Tan rim
    tan = mask & (rand_field >= 0.75)

    noise = np.random.normal(0, 12, img.shape)
    cr, cg, cb = (np.random.randint(20,60), np.random.randint(10,40), np.random.randint(10,35))
    if core.sum() > 0:
        lesion_img[core, 0] = np.clip(cr + noise[core,0], 5, 100)
        lesion_img[core, 1] = np.clip(cg + noise[core,1], 5, 70)
        lesion_img[core, 2] = np.clip(cb + noise[core,2], 5, 70)
    if veil.sum() > 0:
        lesion_img[veil, 0] = np.clip(100 + noise[veil,0], 60, 160)
        lesion_img[veil, 1] = np.clip(100 + noise[veil,1], 60, 160)
        lesion_img[veil, 2] = np.clip(140 + noise[veil,2], 90, 200)
    if regress.sum() > 0:
        lesion_img[regress, 0] = np.clip(180 + noise[regress,0], 130, 230)
        lesion_img[regress, 1] = np.clip(170 + noise[regress,1], 120, 220)
        lesion_img[regress, 2] = np.clip(160 + noise[regress,2], 110, 210)
    if tan.sum() > 0:
        lesion_img[tan, 0] = np.clip(100 + noise[tan,0], 55, 160)
        lesion_img[tan, 1] = np.clip(65  + noise[tan,1], 30, 110)
        lesion_img[tan, 2] = np.clip(45  + noise[tan,2], 20, 90)
    return lesion_img.astype(np.uint8)

# ─── Image generators per class ──────────────────────────────────────────────
def gen_common_nevus():
    size = IMG_SIZE
    img  = gen_skin_bg(size)
    cx   = size // 2 + np.random.randint(-15, 15)
    cy   = size // 2 + np.random.randint(-15, 15)
    rx   = np.random.randint(35, 60)
    ry   = np.random.randint(30, 55)
    ang  = np.random.randint(0, 180)
    mask = gen_ellipse_mask(size, cx, cy, rx, ry, ang)
    img  = apply_nevus_colour(img, mask, dark_level=0.5)
    # Smooth edges
    pil  = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=1.5))
    return np.array(pil)

def gen_atypical_nevus():
    size = IMG_SIZE
    img  = gen_skin_bg(size)
    cx   = size // 2 + np.random.randint(-15, 15)
    cy   = size // 2 + np.random.randint(-15, 15)
    base_r = np.random.randint(38, 65)
    mask = gen_irregular_mask(size, cx, cy, base_r, n_bumps=6, irregularity=0.25)
    img  = apply_atypical_colour(img, mask, dark_level=0.6)
    pil  = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=1.2))
    return np.array(pil)

def gen_melanoma():
    size = IMG_SIZE
    img  = gen_skin_bg(size)
    cx   = size // 2 + np.random.randint(-20, 20)
    cy   = size // 2 + np.random.randint(-20, 20)
    base_r = np.random.randint(45, 80)
    mask = gen_irregular_mask(size, cx, cy, base_r, n_bumps=10, irregularity=0.50)
    img  = apply_melanoma_colour(img, mask, dark_level=0.9)
    pil  = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=1.0))
    return np.array(pil)

GENERATORS = [gen_common_nevus, gen_atypical_nevus, gen_melanoma]

def generate_split(out_dir, n_per_class, seed_offset=0):
    os.makedirs(out_dir, exist_ok=True)
    csv_lines = ["filepath,label"]
    for cls_idx, (cls_name, gen_fn) in enumerate(zip(CLASSES, GENERATORS)):
        cls_dir = os.path.join(out_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            np.random.seed(seed_offset + cls_idx * 10000 + i)
            random.seed(seed_offset + cls_idx * 10000 + i)
            arr = gen_fn()
            fp  = os.path.join(cls_dir, f"{cls_name}_{i:04d}.png")
            Image.fromarray(arr).save(fp)
            csv_lines.append(f"{fp},{cls_idx}")
    csv_path = os.path.join(out_dir, "labels.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"[GEN] Saved {len(csv_lines)-1} images to {out_dir}")
    return csv_path

if __name__ == "__main__":
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "datasets", "synthetic")
    print(f"[GEN] Generating synthetic dermoscopy dataset in: {data_root}")
    generate_split(os.path.join(data_root, "train"), N_PER_CLASS_TRAIN, seed_offset=0)
    generate_split(os.path.join(data_root, "val"),   N_PER_CLASS_VAL,   seed_offset=999999)
    print("[GEN] Done.")
