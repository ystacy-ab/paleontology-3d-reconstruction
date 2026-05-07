import os
import time
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from threading import Thread
import multiprocessing as mp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from rembg import remove, new_session

from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
)


class ImageProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def generate_mask(self) -> np.ndarray:
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open: {self.image_path}")
        session = new_session("isnet-general-use")
        rgba = remove(img, session=session)
        alpha = rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        return mask


def _angle_worker(args: tuple) -> tuple[float, float, float]:
    elev, azim, obj_path, mask_array, img_size = args

    device = torch.device("cpu")
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster = RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=10,
        cull_backfaces=False,
    )
    mesh = load_objs_as_meshes([obj_path], device=device)
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = (verts - center).abs().max()
    mesh = mesh.offset_verts(-center).scale_verts(1.0 / scale.item())

    resized = cv2.resize(mask_array, (img_size, img_size))
    target = torch.tensor(resized, dtype=torch.float32) / 255.0

    R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )
    with torch.no_grad():
        sil = renderer(mesh)[0, ..., 3]

    pred = sil.clamp(0.0, 1.0)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou_loss = (1.0 - intersection / (union + 1e-8)).item()
    return iou_loss, elev, azim


SEARCH_ANGLES = [
    (90, 0),   (90, 45),  (90, 90),  (90, 135),
    (90, 180), (90, 225), (90, 270), (90, 315),
    (75, 0),   (75, 90),  (75, 180), (75, 270),
    (60, 0),   (60, 90),  (60, 180), (60, 270),
    (85, 45),  (85, 135), (85, 225), (85, 315),
]

EARLY_STOP_WINDOW = 30
EARLY_STOP_DELTA  = 1e-4


class Fossil3DReconstructor:

    def __init__(
        self,
        obj_path: str,
        mask: np.ndarray,
        img_size: int = 256,
        device: str = "cpu",
    ):
        self.obj_path = obj_path
        self.mask_raw = mask
        self.img_size = img_size
        self.device = torch.device(device)

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        blur_radius = np.log(1.0 / 1e-4 - 1.0) * self.blend_params.sigma

        self.raster_coarse = RasterizationSettings(
            image_size=img_size,
            blur_radius=blur_radius,
            faces_per_pixel=10,
            cull_backfaces=False,
        )
        self.raster_fine = RasterizationSettings(
            image_size=img_size,
            blur_radius=blur_radius,
            faces_per_pixel=50,
            cull_backfaces=False,
        )

        self.target_silhouette = self._prepare_mask(mask)
        self.mesh = self._load_model(obj_path)

    def _prepare_mask(self, mask: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(mask, (self.img_size, self.img_size))
        return torch.tensor(resized, dtype=torch.float32, device=self.device) / 255.0

    def _load_model(self, obj_path: str) -> Meshes:
        mesh = load_objs_as_meshes([obj_path], device=self.device)
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = (verts - center).abs().max()
        return mesh.offset_verts(-center).scale_verts(1.0 / scale.item())

    def _make_renderer(self, elev: float, azim: float,
                       dist: float = 1.5, coarse: bool = False) -> MeshRenderer:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        raster = self.raster_coarse if coarse else self.raster_fine
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )

    def _iou_loss(self, pred: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        intersection = (pred * self.target_silhouette).sum()
        union = pred.sum() + self.target_silhouette.sum() - intersection
        return 1.0 - intersection / (union + 1e-8)

    def _smoothness_loss(self, deform_verts: torch.Tensor) -> torch.Tensor:
        edges = self.mesh.edges_packed()
        v0 = deform_verts[edges[:, 0]]
        v1 = deform_verts[edges[:, 1]]
        return ((v0 - v1) ** 2).sum(dim=1).mean()

    def _total_loss(self, pred: torch.Tensor,
                    deform_verts: torch.Tensor) -> torch.Tensor:
        iou_loss    = self._iou_loss(pred)
        reg_loss    = deform_verts.norm(dim=1).mean() * 0.01
        smooth_loss = self._smoothness_loss(deform_verts) * 0.001
        return iou_loss + reg_loss + smooth_loss

    def find_best_initial_angle(
        self,
        num_workers: int = 4,
        progress_callback=None,
    ) -> tuple[float, float, float, float]:
        worker_args = [
            (elev, azim, self.obj_path, self.mask_raw, self.img_size)
            for elev, azim in SEARCH_ANGLES
        ]
        t0 = time.time()
        results = []
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_angle_worker, worker_args)):
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(SEARCH_ANGLES))
        elapsed = time.time() - t0
        best_loss, best_elev, best_azim = min(results, key=lambda x: x[0])
        return best_elev, best_azim, best_loss, elapsed

    def optimize_deformation(
        self,
        elev: float,
        azim: float,
        iterations: int = 500,
        progress_callback=None,
    ) -> tuple[Meshes, torch.Tensor, list[float], float]:
        t0 = time.time()
        phase1_max = iterations // 2
        phase2_max = iterations - phase1_max

        deform_verts = torch.zeros(
            self.mesh.verts_packed().shape,
            device=self.device,
            requires_grad=True,
        )
        loss_history: list[float] = []

        renderer1 = self._make_renderer(elev, azim, coarse=True)
        optimizer1 = torch.optim.Adam([deform_verts], lr=0.005)

        for i in tqdm(range(phase1_max), desc="Phase 1 (coarse)"):
            optimizer1.zero_grad()
            deformed = self.mesh.offset_verts(deform_verts)
            sil = renderer1(deformed)[0, ..., 3]
            loss = self._total_loss(sil, deform_verts)
            loss.backward()
            optimizer1.step()
            loss_history.append(loss.item())
            if progress_callback:
                progress_callback(i + 1, iterations)
            if i >= EARLY_STOP_WINDOW:
                improvement = loss_history[-EARLY_STOP_WINDOW] - loss_history[-1]
                if improvement < EARLY_STOP_DELTA:
                    break

        phase1_end = len(loss_history)

        renderer2 = self._make_renderer(elev, azim, coarse=False)
        optimizer2 = torch.optim.Adam([deform_verts], lr=0.001)
        scheduler2 = torch.optim.lr_scheduler.StepLR(
            optimizer2, step_size=50, gamma=0.5
        )

        for i in tqdm(range(phase2_max), desc="Phase 2 (fine)"):
            optimizer2.zero_grad()
            deformed = self.mesh.offset_verts(deform_verts)
            sil = renderer2(deformed)[0, ..., 3]
            loss = self._total_loss(sil, deform_verts)
            loss.backward()
            optimizer2.step()
            scheduler2.step()
            loss_history.append(loss.item())
            if progress_callback:
                progress_callback(phase1_end + i + 1, iterations)
            if i >= EARLY_STOP_WINDOW:
                improvement = loss_history[-EARLY_STOP_WINDOW] - loss_history[-1]
                if improvement < EARLY_STOP_DELTA:
                    break

        elapsed = time.time() - t0
        final_mesh = self.mesh.offset_verts(deform_verts.detach())
        with torch.no_grad():
            final_sil = renderer2(final_mesh)[0, ..., 3].detach()

        return final_mesh, final_sil, loss_history, elapsed, phase1_end

    def save_mesh(self, mesh: Meshes, path: str) -> None:
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        save_obj(path, verts, faces)

    def visualize_result(
        self,
        final_sil: torch.Tensor,
        loss_history: list[float],
        phase1_end: int,
        best_elev: float,
        best_azim: float,
        search_time: float,
        opt_time: float,
    ) -> None:
        target = self.target_silhouette.cpu().numpy()
        pred = final_sil.cpu().numpy()
        diff = np.abs(pred - target)
        final_iou = 1.0 - loss_history[-1]
        total_time = search_time + opt_time

        fig = plt.figure(figsize=(18, 9))
        fig.suptitle(
            f"3D Fossil Reconstruction  |  IoU = {final_iou:.4f}  "
            f"|  elev={best_elev}, azim={best_azim}",
            fontsize=14, fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(target, cmap="gray")
        ax0.set_title("Fossil mask")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(pred, cmap="gray")
        ax1.set_title("Model projection")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(diff, cmap="hot")
        ax2.set_title(f"Difference (IoU={final_iou:.3f})")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 3])
        overlay = np.stack([target, pred, np.zeros_like(target)], axis=-1)
        ax3.imshow(overlay)
        ax3.set_title("Overlay (R=mask, G=projection)")
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[1, :2])
        ax4.plot(range(phase1_end), loss_history[:phase1_end],
                 color="#2196F3", linewidth=1.5, label="Phase 1 (coarse)")
        ax4.plot(range(phase1_end, len(loss_history)), loss_history[phase1_end:],
                 color="#FF9800", linewidth=1.5, label="Phase 2 (fine)")
        ax4.axvline(x=phase1_end, color="gray", linestyle=":", alpha=0.7)
        ax4.set_title("IoU Loss - two-phase optimization")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Loss")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis("off")
        time_data = [
            ["Angle search (parallel)", f"{search_time:.2f} s"],
            ["Deformation optimization", f"{opt_time:.2f} s"],
            ["Total", f"{total_time:.2f} s"],
            ["", ""],
            ["Best elev", f"{best_elev}"],
            ["Best azim", f"{best_azim}"],
            ["Final IoU", f"{final_iou:.4f}"],
            ["Total iterations", f"{len(loss_history)}"],
            ["Phase 1 iterations", f"{phase1_end}"],
            ["Phase 2 iterations", f"{len(loss_history) - phase1_end}"],
        ]
        table = ax5.table(
            cellText=time_data,
            colLabels=["Parameter", "Value"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.6)
        ax5.set_title("Results summary", pad=10)

        plt.savefig("reconstruction_result.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


class FossilApp:
    OBJ_PATH = "3d-model.obj"

    def __init__(self):
        self.image_path: str | None = None
        self.root = tk.Tk()
        self.root.title("3D Fossil Reconstruction")
        self.root.geometry("520x420")
        self.root.resizable(False, False)

        self.status_var = tk.StringVar(value="Ready")
        self.threads_var = tk.IntVar(value=min(4, mp.cpu_count()))
        self.img_size_var = tk.IntVar(value=64)
        self.search_progress = tk.DoubleVar(value=0)
        self.opt_progress = tk.DoubleVar(value=0)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        frame_file = tk.LabelFrame(self.root, text="Fossil Image", **pad)
        frame_file.pack(fill="x", **pad)
        self.file_label = tk.Label(frame_file, text="No file selected",
                                   fg="gray", wraplength=400, anchor="w")
        self.file_label.pack(side="left", expand=True, fill="x")
        tk.Button(frame_file, text="Browse...", command=self._choose_image,
                  width=8).pack(side="right")

        frame_params = tk.LabelFrame(self.root, text="Parameters", **pad)
        frame_params.pack(fill="x", **pad)

        row1 = tk.Frame(frame_params)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Threads (angle search):", width=32,
                 anchor="w").pack(side="left")
        tk.Spinbox(row1, from_=1, to=mp.cpu_count(), textvariable=self.threads_var,
                   width=5).pack(side="left")
        tk.Label(row1, text=f"(max {mp.cpu_count()})", fg="gray").pack(side="left", padx=4)

        row2 = tk.Frame(frame_params)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Image resolution:", width=32,
                 anchor="w").pack(side="left")
        for val in [64, 128, 256]:
            tk.Radiobutton(row2, text=str(val), variable=self.img_size_var,
                           value=val).pack(side="left", padx=4)
        tk.Label(row2, text="px", fg="gray").pack(side="left")

        frame_prog = tk.LabelFrame(self.root, text="Progress", **pad)
        frame_prog.pack(fill="x", **pad)
        tk.Label(frame_prog, text="Angle search:", anchor="w").pack(fill="x")
        ttk.Progressbar(frame_prog, variable=self.search_progress,
                        maximum=100).pack(fill="x", padx=4, pady=2)
        tk.Label(frame_prog, text="Optimization:", anchor="w").pack(fill="x")
        ttk.Progressbar(frame_prog, variable=self.opt_progress,
                        maximum=100).pack(fill="x", padx=4, pady=2)

        tk.Label(self.root, textvariable=self.status_var,
                 fg="#1565C0", anchor="w").pack(fill="x", **pad)
        tk.Button(
            self.root, text="Run Reconstruction",
            command=self._start,
            bg="#43A047", fg="white",
            font=("Helvetica", 11, "bold"),
            height=2,
        ).pack(fill="x", padx=12, pady=8)

    def _choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.image_path = path
            self.file_label.config(text=os.path.basename(path), fg="black")

    def _set_status(self, text: str):
        self.root.after(0, self.status_var.set, text)

    def _update_search_bar(self, done: int, total: int):
        self.root.after(0, self.search_progress.set, done / total * 100)

    def _update_opt_bar(self, done: int, total: int):
        self.root.after(0, self.opt_progress.set, done / total * 100)

    def _start(self):
        if not self.image_path:
            self._set_status("Please select an image first!")
            return
        self.search_progress.set(0)
        self.opt_progress.set(0)
        self._set_status("Processing image...")
        Thread(target=self._pipeline, daemon=True).start()

    def _pipeline(self):
        try:
            self._set_status("Generating mask (rembg)...")
            mask = ImageProcessor(self.image_path).generate_mask()
            self._set_status("Mask created. Loading model...")

            img_size = self.img_size_var.get()
            reconstructor = Fossil3DReconstructor(
                obj_path=self.OBJ_PATH,
                mask=mask,
                img_size=img_size,
            )

            n_workers = self.threads_var.get()
            self._set_status(
                f"Searching angle ({len(SEARCH_ANGLES)} candidates, {n_workers} threads)..."
            )
            best_elev, best_azim, best_loss, search_time = (
                reconstructor.find_best_initial_angle(
                    num_workers=n_workers,
                    progress_callback=self._update_search_bar,
                )
            )
            self._update_search_bar(len(SEARCH_ANGLES), len(SEARCH_ANGLES))
            self._set_status(
                f"Angle found in {search_time:.1f} s "
                f"(elev={best_elev}, azim={best_azim}). Optimizing..."
            )

            iters = 500
            final_mesh, final_sil, loss_history, opt_time, phase1_end = (
                reconstructor.optimize_deformation(
                    best_elev, best_azim,
                    iterations=iters,
                    progress_callback=self._update_opt_bar,
                )
            )
            self._update_opt_bar(iters, iters)

            out_path = "final_deformed_model.obj"
            reconstructor.save_mesh(final_mesh, out_path)

            final_iou = 1.0 - loss_history[-1]
            total = search_time + opt_time
            self._set_status(
                f"Done! {out_path}  |  IoU={final_iou:.4f}  |  "
                f"Search: {search_time:.1f} s  "
                f"Opt: {opt_time:.1f} s  "
                f"Total: {total:.1f} s"
            )
            reconstructor.visualize_result(
                final_sil, loss_history, phase1_end,
                best_elev, best_azim,
                search_time, opt_time,
            )
            self.root.after(0, self._open_result_image)

        except Exception as exc:
            self._set_status(f"Error: {exc}")
            import traceback; traceback.print_exc()

    def _open_result_image(self):
        import subprocess
        import platform
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", "reconstruction_result.png"])
            elif platform.system() == "Windows":
                os.startfile("reconstruction_result.png")
            else:
                subprocess.Popen(["xdg-open", "reconstruction_result.png"])
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    if not os.path.exists(FossilApp.OBJ_PATH):
        print(f"Warning: {FossilApp.OBJ_PATH} not found.")
    FossilApp().run()