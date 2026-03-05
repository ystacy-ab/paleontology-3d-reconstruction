import numpy as np
import cv2
import trimesh
from scipy.optimize import minimize

def project_3d_to_2d(params, points, img_shape):
    rx, ry, rz, scale, tx, ty = params
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    rotated = points @ Rx.T @ Ry.T @ Rz.T
    projected = (rotated[:, :2] * scale) + [tx, ty]
    
    mask_gen = np.zeros(img_shape, dtype=np.uint8)
    h, w = img_shape
    valid = (projected[:, 0] >= 0) & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h)
    pts = projected[valid].astype(int)
    mask_gen[pts[:, 1], pts[:, 0]] = 255
            
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_gen = cv2.morphologyEx(mask_gen, cv2.MORPH_CLOSE, kernel)
    mask_gen = cv2.dilate(mask_gen, kernel, iterations=1)
    return mask_gen

def objective(params, points, target_mask):
    gen_mask = project_3d_to_2d(params, points, target_mask.shape)
    intersection = np.count_nonzero(np.logical_and(target_mask, gen_mask))
    union = np.count_nonzero(np.logical_or(target_mask, gen_mask))
    return 1 - (intersection / (union + 1e-6))

mask = cv2.imread('image3_mask.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

loaded = trimesh.load('eurypterus.glb')
mesh = trimesh.util.concatenate([g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]) if isinstance(loaded, trimesh.Scene) else loaded

points = mesh.sample(5000)
points -= mesh.center_mass
points /= np.max(np.linalg.norm(points, axis=1))

coords = np.argwhere(mask == 255)
center_y, center_x = coords.mean(axis=0)
mask_size = (np.max(coords[:, 1]) - np.min(coords[:, 1]) + np.max(coords[:, 0]) - np.min(coords[:, 0])) / 2

best_global_iou = -1
best_global_params = None

grid_angles = [0, np.pi/2, np.pi, 3*np.pi/2]

for rx in grid_angles:
    for ry in [0, np.pi/2]:
        for rz in np.linspace(0, 2*np.pi, 8):
            start_params = [rx, ry, rz, mask_size, center_x, center_y]
            iou = 1 - objective(start_params, points, mask)
            if iou > best_global_iou:
                best_global_iou = iou
                best_global_params = start_params

res = minimize(objective, best_global_params, args=(points, mask), method='Nelder-Mead', tol=1e-4)

final_mask = project_3d_to_2d(res.x, points, mask.shape)
overlap = cv2.merge([np.zeros_like(mask), final_mask, mask])
cv2.imwrite('overlap_result.png', overlap)

print(f"Final IoU: {1 - res.fun}")
print(f"Final Params: {res.x}")