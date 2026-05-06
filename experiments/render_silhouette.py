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
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

device = torch.device("cpu")
IMG_SIZE = 64

obj_filename = "3d-model.obj"
mesh = load_objs_as_meshes([obj_filename], device=device)
print(f"Модель завантажена: {mesh.verts_packed().shape[0]} вершин")

verts = mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh = mesh.offset_verts(-center)
mesh = mesh.scale_verts(1.0 / scale.item())

print(f"   Модель нормалізована.")
print(f"   Центр оригінальний: {center}")
print(f"   Масштаб: {scale.item():.3f}")

def rotate_mesh(mesh, angle_x=0.0, angle_y=0.0, angle_z=0.0):
    ax = torch.tensor(angle_x * np.pi / 180.0)
    ay = torch.tensor(angle_y * np.pi / 180.0)
    az = torch.tensor(angle_z * np.pi / 180.0)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, float(torch.cos(ax)), float(-torch.sin(ax))],
        [0, float(torch.sin(ax)),  float(torch.cos(ax))]
    ], dtype=torch.float32)
    Ry = torch.tensor([
        [float(torch.cos(ay)), 0, float(torch.sin(ay))],
        [0, 1, 0],
        [float(-torch.sin(ay)), 0, float(torch.cos(ay))]
    ], dtype=torch.float32)
    Rz = torch.tensor([
        [float(torch.cos(az)), float(-torch.sin(az)), 0],
        [float(torch.sin(az)),  float(torch.cos(az)), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    R_total = Rz @ Ry @ Rx
    verts = mesh.verts_packed()
    new_verts = (R_total @ verts.T).T
    return mesh.update_padded(new_verts.unsqueeze(0))

mesh = rotate_mesh(mesh, angle_x=90.0, angle_y=0.0, angle_z=180.0)

verts_after = mesh.verts_packed()
print(f"   Модель повернута.")
print(f"   Центр після повороту: {verts_after.mean(0)}")
print(f"   Min/Max по Y: {verts_after[:, 1].min():.3f} / {verts_after[:, 1].max():.3f}")


blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma

raster_coarse = RasterizationSettings(
    image_size=IMG_SIZE,
    blur_radius=blur_radius,
    faces_per_pixel=10,
    cull_backfaces=False
)

raster_fine = RasterizationSettings(
    image_size=IMG_SIZE,
    blur_radius=blur_radius,
    faces_per_pixel=50,
    cull_backfaces=False
)

R_init, T_init = look_at_view_transform(dist=1.5, elev=90, azim=0)
cameras = FoVPerspectiveCameras(device=device, R=R_init, T=T_init)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_fine),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

print("\nПроекції з різних кутів (паралельно)...")

angles = [
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

def render_angle(args):
    """Рендерить один кут — викликається паралельно"""
    elev_t, azim_t, dist_t = args
    R_t, T_t = look_at_view_transform(dist=dist_t, elev=elev_t, azim=azim_t)
    cameras_t = FoVPerspectiveCameras(device=device, R=R_t, T=T_t)
    renderer_local = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras_t, raster_settings=raster_coarse),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    sil = renderer_local(mesh)[0, ..., 3].detach().cpu().numpy()
    sil_uint8 = (sil * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(sil_uint8, cv2.MORPH_CLOSE, kernel)

with ThreadPoolExecutor(max_workers=4) as executor:
    rendered_angles = list(executor.map(render_angle, angles))

plt.figure(figsize=(20, 12))
for i, (sil_filled, (elev_t, azim_t, dist_t)) in enumerate(zip(rendered_angles, angles)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(sil_filled, cmap='gray')
    plt.title(f"e={elev_t} az={azim_t}", fontsize=7, pad=2)
    plt.axis('off')

plt.suptitle("Проекції 3D моделі евриптерида з різних кутів камери", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

mask = "masks/image3_mask.png"
mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
mask_img = cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE))
real_silhouette = torch.tensor(mask_img, device=device).float() / 255.0
print(f"   Маска завантажена: {real_silhouette.shape}")
print(f"   Ненульових пікселів: {(real_silhouette > 0.5).sum().item()}")

plt.figure(figsize=(6, 6))
plt.imshow(real_silhouette.cpu().numpy(), cmap='gray')
plt.title("Маска силуету")
plt.axis('off')
plt.show()

def silhouette_loss(pred_silhouette, target_silhouette, deform_verts=None):
    pred_alpha = torch.clamp(pred_silhouette[0, ..., 3], 0, 1)
    intersection = (pred_alpha * target_silhouette).sum()
    union = pred_alpha.sum() + target_silhouette.sum() - intersection
    iou_loss = 1 - intersection / (union + 1e-8)
    if deform_verts is not None:
        reg_loss = deform_verts.norm(dim=1).mean() * 0.01
        return iou_loss + reg_loss
    return iou_loss

print("\nНайкращий стартовий кут (паралельно)...")

def eval_angle(args):
    elev_s, azim_s = args
    R_s, T_s = look_at_view_transform(dist=1.5, elev=elev_s, azim=azim_s)
    cam_s = FoVPerspectiveCameras(device=device, R=R_s, T=T_s)
    renderer_s = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam_s, raster_settings=raster_coarse),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    with torch.no_grad():
        sil_s = renderer_s(mesh)
        l = silhouette_loss(sil_s, real_silhouette, None).item()
    return l, elev_s, azim_s

search_angles = [
    (90, 0), (90, 90), (90, 180), (90, 270),
    (85, 0), (85, 90), (85, 180), (85, 270),
    (75, 0), (75, 90), (75, 180), (75, 270),
]

with ThreadPoolExecutor(max_workers=4) as executor:
    angle_results = list(executor.map(eval_angle, search_angles))

best_loss, best_elev, best_azim = min(angle_results, key=lambda x: x[0])
print(f"Найкращий кут: elev={best_elev}, azim={best_azim}, Loss={best_loss:.4f}")

ELEV_START = best_elev
AZIM_START = best_azim
DIST_START = 1.5

# ------------------------------------------------------------
# OPTIMIZATION
# ------------------------------------------------------------
R_opt, T_opt = look_at_view_transform(dist=DIST_START, elev=ELEV_START, azim=AZIM_START)
cameras_opt = FoVPerspectiveCameras(device=device, R=R_opt, T=T_opt)

deform_verts = torch.zeros(mesh.verts_packed().shape, device=device, requires_grad=True)

loss_history = []
t_start = time.time()

print(f"\nФаза 1 — груба оптимізація (швидка, faces_per_pixel=10)...")

renderer_coarse = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras_opt, raster_settings=raster_coarse),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

optimizer1 = torch.optim.Adam([deform_verts], lr=0.005)

PHASE1_MAX = 150
EARLY_STOP_WINDOW = 30
EARLY_STOP_DELTA = 1e-4

for i in tqdm(range(PHASE1_MAX), desc="Фаза 1"):
    optimizer1.zero_grad()
    deformed_mesh = mesh.offset_verts(deform_verts)
    silhouette = renderer_coarse(deformed_mesh)
    loss = silhouette_loss(silhouette, real_silhouette, deform_verts)
    loss.backward()
    optimizer1.step()
    loss_history.append(loss.item())

    if i % 30 == 0:
        print(f"  Ітерація {i:3d} | Loss: {loss.item():.4f}")

    if i >= EARLY_STOP_WINDOW:
        recent_improvement = loss_history[-EARLY_STOP_WINDOW] - loss_history[-1]
        if recent_improvement < EARLY_STOP_DELTA:
            print(f"  Рання зупинка фази 1 на ітерації {i} "
                  f"(покращення {recent_improvement:.5f} < {EARLY_STOP_DELTA})")
            break

phase1_end = len(loss_history)
print(f"Фаза 1 завершена: {phase1_end} ітерацій, Loss={loss_history[-1]:.4f}")

print(f"\nФаза 2 — точна оптимізація (якісна, faces_per_pixel=50)...")

renderer_fine = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras_opt, raster_settings=raster_fine),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

optimizer2 = torch.optim.Adam([deform_verts], lr=0.001)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.5)

PHASE2_MAX = 150

for i in tqdm(range(PHASE2_MAX), desc="Фаза 2"):
    optimizer2.zero_grad()
    deformed_mesh = mesh.offset_verts(deform_verts)
    silhouette = renderer_fine(deformed_mesh)
    loss = silhouette_loss(silhouette, real_silhouette, deform_verts)
    loss.backward()
    optimizer2.step()
    scheduler2.step()
    loss_history.append(loss.item())

    if i % 30 == 0:
        print(f"  Ітерація {i:3d} | Loss: {loss.item():.4f}")

    if i >= EARLY_STOP_WINDOW:
        recent_improvement = loss_history[-EARLY_STOP_WINDOW] - loss_history[-1]
        if recent_improvement < EARLY_STOP_DELTA:
            print(f"  Рання зупинка фази 2 на ітерації {i} "
                  f"(покращення {recent_improvement:.5f} < {EARLY_STOP_DELTA})")
            break

t_elapsed = time.time() - t_start
total_iters = len(loss_history)
print(f"\nОптимізація завершена за {t_elapsed:.1f} с ({total_iters} ітерацій)")
print(f"Фінальний Loss: {loss_history[-1]:.4f}")
pytorch3d_iou = 1 - loss_history[-1]

plt.figure(figsize=(10, 4))
plt.plot(range(phase1_end), loss_history[:phase1_end],
         color='steelblue', linewidth=2, label='Фаза 1 (груба, швидка)')
plt.plot(range(phase1_end, total_iters), loss_history[phase1_end:],
         color='darkorange', linewidth=2, label='Фаза 2 (точна, якісна)')
plt.axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.7, label='Перехід між фазами')
plt.axhline(y=loss_history[-1], color='red', linestyle='--',
            label=f"Фінальний Loss = {loss_history[-1]:.4f}")
plt.title("Loss по ітераціях — двофазна оптимізація")
plt.xlabel("Ітерація")
plt.ylabel("IoU Loss (менше = краще)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------
final_mesh = mesh.offset_verts(deform_verts)
final_silhouette = renderer_fine(final_mesh)
final_img_raw = final_silhouette[0, ..., 3].detach().cpu().numpy()

final_uint8 = (final_img_raw * 255).astype(np.uint8)
kernel = np.ones((3, 3), np.uint8)
final_filled = cv2.morphologyEx(final_uint8, cv2.MORPH_CLOSE, kernel)
final_img = final_filled.astype(np.float32) / 255.0

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(real_silhouette.cpu().numpy(), cmap='gray')
plt.title("Реальний відбиток\n(маска)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(final_img, cmap='gray')
plt.title("Оптимізована 3D модель\n(PyTorch3D проекція)")
plt.axis('off')

plt.subplot(1, 3, 3)
difference = np.abs(final_img - real_silhouette.cpu().numpy())
plt.imshow(difference, cmap='hot')
plt.title(f"Різниця\n(Loss = {loss_history[-1]:.4f})")
plt.axis('off')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# COMPARISON WITH COMBINE.PY
# ------------------------------------------------------------
combine_iou = 1 - best_loss

methods = ['combine.py\n(хмара точок +\nNelder-Mead)', 'PyTorch3D\n(меш + Adam +\nдеформація)']
iou_values = [combine_iou, pytorch3d_iou]
colors = ['#e07b54', '#5b8db8']

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, iou_values, color=colors, width=0.4, edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, iou_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'IoU = {val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(0, 1.1)
plt.ylabel("IoU (більше = краще)", fontsize=12)
plt.title("Порівняння підходів: IoU збігу з реальним відбитком", fontsize=13)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n  combine.py  (Nelder-Mead, хмара точок): IoU = {combine_iou:.4f}")
print(f"  PyTorch3D   (Adam, деформація меша):     IoU = {pytorch3d_iou:.4f}")
print(f"  Покращення PyTorch3D над combine.py:     +{(pytorch3d_iou - combine_iou):.4f}")

print("\nПІДСУМОК")
print(f"  Стартовий Loss:    {loss_history[0]:.4f}")
print(f"  Фінальний Loss:    {loss_history[-1]:.4f}")
print(f"  Всього ітерацій:   {total_iters} (замість 1000)")
print(f"  Час оптимізації:   {t_elapsed:.1f} с")
print(f"  Фінальний IoU:     {pytorch3d_iou:.4f} ({pytorch3d_iou*100:.1f}%)")