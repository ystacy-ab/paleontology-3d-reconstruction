import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams
)
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor

# ── налаштування ──────────────────────────────
GT_PATH   = "masks/image3_mask.png"
OUT_PATH  = "masks_code/pytorch3d_best.png"
OBJ_PATH  = "3d-model.obj"
IMG_SIZE  = 512   # більше = точніший IoU
WORKERS   = 4
# ─────────────────────────────────────────────

ANGLES = [
    (90,   0, 1.5),
    (90,  90, 1.5),
    (90, 180, 1.5),
    (90, 270, 1.5),
    (85,   0, 1.5),
    (85,  90, 1.5),
    (85, 180, 1.5),
    (85, 270, 1.5),
    (75,   0, 1.5),
    (75,  90, 1.5),
    (75, 180, 1.5),
    (75, 270, 1.5),
]

def iou(mask_a, mask_b):
    a = (mask_a > 127).astype(np.uint8)
    b = (mask_b > 127).astype(np.uint8)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0

# ── завантаження моделі (один раз, до пулу) ──
print("Завантажую модель...")
device = torch.device("cpu")
mesh = load_objs_as_meshes([OBJ_PATH], device=device)

verts = mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh = mesh.offset_verts(-center)
mesh = mesh.scale_verts(1.0 / scale.item())

# початковий поворот як у render_silhouette.py
def rotate_mesh(m, ax_deg, ay_deg, az_deg):
    ax = torch.tensor(ax_deg * np.pi / 180.0)
    ay = torch.tensor(ay_deg * np.pi / 180.0)
    az = torch.tensor(az_deg * np.pi / 180.0)
    Rx = torch.tensor([[1,0,0],[0,float(torch.cos(ax)),float(-torch.sin(ax))],[0,float(torch.sin(ax)),float(torch.cos(ax))]], dtype=torch.float32)
    Ry = torch.tensor([[float(torch.cos(ay)),0,float(torch.sin(ay))],[0,1,0],[float(-torch.sin(ay)),0,float(torch.cos(ay))]], dtype=torch.float32)
    Rz = torch.tensor([[float(torch.cos(az)),float(-torch.sin(az)),0],[float(torch.sin(az)),float(torch.cos(az)),0],[0,0,1]], dtype=torch.float32)
    R_total = Rz @ Ry @ Rx
    new_verts = (R_total @ m.verts_packed().T).T
    return m.update_padded(new_verts.unsqueeze(0))

mesh = rotate_mesh(mesh, 90.0, 0.0, 180.0)

# ── еталонна маска ────────────────────────────
gt_raw = cv2.imread(GT_PATH, cv2.IMREAD_GRAYSCALE)
if gt_raw is None:
    raise FileNotFoundError(f"Не знайдено еталонну маску: {GT_PATH}")
gt = cv2.resize(gt_raw, (IMG_SIZE, IMG_SIZE))
_, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
print(f"Еталон завантажено: {GT_PATH}")

# ── рендер одного кута (запускається у потоці) ─
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    faces_per_pixel=50,
    cull_backfaces=False
)

def render_angle(args):
    elev_t, azim_t, dist_t = args
    R_t, T_t = look_at_view_transform(dist=dist_t, elev=elev_t, azim=azim_t)
    cameras_t = FoVPerspectiveCameras(device=device, R=R_t, T=T_t)

    # кожен потік створює свій рендерер (не thread-safe спільно)
    renderer_local = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras_t, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    sil = renderer_local(mesh)[0, ..., 3].detach().cpu().numpy()
    sil_uint8 = (sil * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sil_filled = cv2.morphologyEx(sil_uint8, cv2.MORPH_CLOSE, kernel)

    score = iou(gt, sil_filled)
    return elev_t, azim_t, score, sil_filled

# ── паралельний рендер ─────────────────────────
print(f"\nРендерю {len(ANGLES)} кутів паралельно ({WORKERS} потоки)...")

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    results = list(executor.map(render_angle, ANGLES))

# ── знаходимо найкращий ───────────────────────
results.sort(key=lambda x: -x[2])   # сортуємо за IoU спадно

print(f"\n{'Кут (elev, azim)':<22} {'IoU':>6}")
print("-" * 30)
for elev_t, azim_t, score, _ in results:
    marker = " ← найкращий" if (elev_t, azim_t) == (results[0][0], results[0][1]) else ""
    print(f"e={elev_t:3d}  az={azim_t:3d}         {score:.4f}{marker}")

best_elev, best_azim, best_iou, best_mask = results[0]

# ── зберігаємо найкращу маску ─────────────────
os.makedirs("masks_code", exist_ok=True)
cv2.imwrite(OUT_PATH, best_mask)
print(f"\nНайкраща маска збережена: {OUT_PATH}")
print(f"PyTorch3D IoU = {best_iou:.4f}  (e={best_elev}, az={best_azim})")
print("\nТепер запусти: python compare_iou.py")