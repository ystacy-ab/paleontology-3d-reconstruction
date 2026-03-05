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

device = torch.device("cpu")
IMG_SIZE = 64

print("Завантажую модель...")
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
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    faces_per_pixel=50,
    cull_backfaces=False
)

R_init, T_init = look_at_view_transform(dist=1.5, elev=90, azim=0)
cameras = FoVPerspectiveCameras(device=device, R=R_init, T=T_init)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

print(" Рендерер створено.")
print("\n Проекції з різних кутів...")

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

plt.figure(figsize=(20, 12))
for i, (elev_t, azim_t, dist_t) in enumerate(angles):
    R_t, T_t = look_at_view_transform(dist=dist_t, elev=elev_t, azim=azim_t)
    cameras_t = FoVPerspectiveCameras(device=device, R=R_t, T=T_t)
    renderer.rasterizer.cameras = cameras_t

    sil = renderer(mesh)[0, ..., 3].detach().cpu().numpy()

    sil_uint8 = (sil * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sil_filled = cv2.morphologyEx(sil_uint8, cv2.MORPH_CLOSE, kernel)

    plt.subplot(3, 4, i + 1)
    plt.imshow(sil_filled, cmap='gray')
    plt.title(f"e={elev_t} az={azim_t}", fontsize=7, pad=2)
    plt.axis('off')

plt.suptitle("Проекції 3D моделі евриптерида з різних кутів камери", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

mask = "image3_rembg_mask.png"

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
