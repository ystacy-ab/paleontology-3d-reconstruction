import trimesh
import numpy as np

scene = trimesh.load('eurypterus.glb')

if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate(
        [geometry for geometry in scene.geometry.values()]
    )
else:
    mesh = scene

points_3d = mesh.sample(5000)

points_3d -= mesh.center_mass
scale = np.max(np.linalg.norm(points_3d, axis=1))
points_3d /= scale

print(f"Отримано {points_3d.shape[0]} точок")
np.save('eurypterid_points.npy', points_3d)