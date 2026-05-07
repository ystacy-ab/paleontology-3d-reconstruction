"""
fossil_reconstruction.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Пайплайн 3D-реконструкції скам'янілості:
  1. Зображення відбитку → ЧБ маска (rembg)
  2. Паралельний пошук найкращого кута (multiprocessing)
  3. Двофазна оптимізація деформації вершин (PyTorch autograd)
  4. Збереження .obj + візуалізація положення у відбитку + час

ВИПРАВЛЕННЯ "ШИПІВ" (розірвана модель):
  Проблема: lr=0.005 + reg*0.01 + smooth*0.001 → вершини розлітались.
  Рішення:
    • lr фази 1: 0.005 → 0.002  (менш агресивний крок)
    • reg_loss:  *0.01 → *0.1   (у 10× сильніше утримує форму)
    • smooth_loss: *0.001 → *0.5 (у 500× — головний захист від шипів)
    • deform clamp: ±0.3 на кожну ітерацію (жорстка межа зміщення)
"""

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


# ══════════════════════════════════════════════════════════
# МОДУЛЬ 1 — ОБРОБКА ЗОБРАЖЕННЯ
# ══════════════════════════════════════════════════════════

class ImageProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def generate_mask(self) -> np.ndarray:
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Не вдалося відкрити: {self.image_path}")
        session = new_session("isnet-general-use")
        rgba = remove(img, session=session)
        alpha = rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        return mask


# ══════════════════════════════════════════════════════════
# МОДУЛЬ 2 — ВОРКЕР ДЛЯ ПАРАЛЕЛЬНОГО ПОШУКУ КУТА
# ══════════════════════════════════════════════════════════

def _angle_worker(args: tuple) -> tuple[float, float, float]:
    """Запускається у дочірньому процесі. Повертає (iou_loss, elev, azim)."""
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


# ══════════════════════════════════════════════════════════
# МОДУЛЬ 3 — 3D РЕКОНСТРУКТОР
# ══════════════════════════════════════════════════════════

SEARCH_ANGLES = [
    (90, 0),   (90, 45),  (90, 90),  (90, 135),
    (90, 180), (90, 225), (90, 270), (90, 315),
    (75, 0),   (75, 90),  (75, 180), (75, 270),
    (60, 0),   (60, 90),  (60, 180), (60, 270),
    (85, 45),  (85, 135), (85, 225), (85, 315),
]

EARLY_STOP_WINDOW = 30
EARLY_STOP_DELTA  = 1e-4

# ── Гіперпараметри регуляризації ──────────────────────────────────────
# Ці значення — ключовий фікс розірваної моделі.
# smooth_loss * 0.5 не дає сусіднім вершинам розійтись (шипи).
# reg_loss * 0.1 не дає вершинам відлетіти далеко від оригіналу.
REG_WEIGHT    = 0.30
SMOOTH_WEIGHT = 1.00
DEFORM_CLAMP  = 0.10

LR_SCALE  = 0.01    # lr для підбору масштабу
LR_PHASE1 = 0.001
LR_PHASE2 = 0.0005


class Fossil3DReconstructor:

    def __init__(
        self,
        obj_path: str,
        mask: np.ndarray,
        img_size: int = 64,
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
                       coarse: bool = False) -> MeshRenderer:
        R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
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
        """
        Штрафує за різкі перепади між сусідніми вершинами.
        Це головний захист від «шипів» на моделі.
        """
        edges = self.mesh.edges_packed()
        v0 = deform_verts[edges[:, 0]]
        v1 = deform_verts[edges[:, 1]]
        return ((v0 - v1) ** 2).sum(dim=1).mean()

    def _total_loss(self, pred: torch.Tensor,
                    deform_verts: torch.Tensor) -> torch.Tensor:
        return (
            self._iou_loss(pred)
            + deform_verts.norm(dim=1).mean()     * REG_WEIGHT
            + self._smoothness_loss(deform_verts) * SMOOTH_WEIGHT
        )

    # ── ПАРАЛЕЛЬНИЙ ПОШУК КУТА ─────────────────────────────────────────

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

    # ── ДВОФАЗНА ОПТИМІЗАЦІЯ ───────────────────────────────────────────

    def optimize_deformation(
        self,
        elev: float,
        azim: float,
        iterations: int = 500,
        progress_callback=None,
    ) -> tuple[Meshes, torch.Tensor, list[float], float, int]:
        """
        3-фазна оптимізація:

        Фаза 0 (50 іт.) — підбір масштабу:
          Оптимізує scalar log_scale без деформації вершин.
          Після цього модель і маска мають однаковий розмір.

        Фаза 1 — груба деформація (coarse renderer, lr=LR_PHASE1).
        Фаза 2 — точна деформація (fine renderer,  lr=LR_PHASE2).

        deform_verts clamp ±DEFORM_CLAMP кожну ітерацію.
        """
        t0 = time.time()
        n_scale  = 50
        n_deform = iterations - n_scale
        phase1_max = n_deform // 2
        phase2_max = n_deform - phase1_max

        loss_history: list[float] = []
        renderer_c = self._make_renderer(elev, azim, coarse=True)

        # ── ФАЗА 0: підбір масштабу ────────────────────────────────────
        print("Фаза 0 — підбір масштабу...")
        log_scale = torch.tensor(0.0, device=self.device, requires_grad=True)
        opt_s = torch.optim.Adam([log_scale], lr=LR_SCALE)

        for i in tqdm(range(n_scale), desc="Фаза 0 (масштаб)"):
            opt_s.zero_grad(set_to_none=True)
            # Множимо вершини на exp(log_scale) — градієнт зберігається
            scale = torch.exp(log_scale)
            scaled_verts = self.mesh.verts_packed() * scale
            scaled_mesh = self.mesh.update_padded(scaled_verts.unsqueeze(0))
            sil = renderer_c(scaled_mesh)[0, ..., 3]
            loss = self._iou_loss(sil)
            loss.backward()
            opt_s.step()
            loss_history.append(loss.item())
            if progress_callback and i % 10 == 0:
                progress_callback(i + 1, iterations)

        best_scale = torch.exp(log_scale.detach()).item()
        print(f"  Масштаб: {best_scale:.4f}, Loss={loss_history[-1]:.4f}")
        phase0_end = len(loss_history)

        with torch.no_grad():
            scaled_base = self.mesh.scale_verts(best_scale)

        # ── ФАЗИ 1 та 2: деформація ────────────────────────────────────
        deform_verts = torch.zeros(
            scaled_base.verts_packed().shape,
            device=self.device,
            requires_grad=True,
        )

        def run_deform_phase(renderer, optimizer, scheduler,
                             max_iters, label, offset) -> int:
            for i in tqdm(range(max_iters), desc=label):
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    deform_verts.clamp_(-DEFORM_CLAMP, DEFORM_CLAMP)
                deformed = scaled_base.offset_verts(deform_verts)
                sil = renderer(deformed)[0, ..., 3]
                loss = self._total_loss(sil, deform_verts)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                loss_history.append(loss.item())
                if progress_callback and (i % 5 == 0 or i == max_iters - 1):
                    progress_callback(n_scale + offset + i + 1, iterations)
                if i >= EARLY_STOP_WINDOW:
                    imp = (loss_history[-EARLY_STOP_WINDOW - 1]
                           - loss_history[-1])
                    if imp < EARLY_STOP_DELTA:
                        tqdm.write(f"  {label}: early stop іт.{i+1}")
                        if progress_callback:
                            progress_callback(iterations, iterations)
                        return i + 1
            return max_iters

        r1 = self._make_renderer(elev, azim, coarse=True)
        o1 = torch.optim.Adam([deform_verts], lr=LR_PHASE1)
        p1_done = run_deform_phase(r1, o1, None, phase1_max, "Фаза 1 (груба)", 0)

        r2 = self._make_renderer(elev, azim, coarse=False)
        o2 = torch.optim.Adam([deform_verts], lr=LR_PHASE2)
        s2 = torch.optim.lr_scheduler.StepLR(o2, step_size=50, gamma=0.5)
        run_deform_phase(r2, o2, s2, phase2_max, "Фаза 2 (точна)", p1_done)

        elapsed = time.time() - t0
        final_mesh = scaled_base.offset_verts(deform_verts.detach())
        with torch.no_grad():
            final_sil = r2(final_mesh)[0, ..., 3].detach()

        print(f"Готово: {len(loss_history)} іт., scale={best_scale:.3f}, "
              f"{elapsed:.1f} с, IoU={1.0 - loss_history[-1]:.4f}")
        return final_mesh, final_sil, loss_history, elapsed, phase0_end

    # ── ЗБЕРЕЖЕННЯ ─────────────────────────────────────────────────────

    def save_mesh(self, mesh: Meshes, path: str) -> None:
        save_obj(path, mesh.verts_packed(), mesh.faces_packed())

    # ── ВІЗУАЛІЗАЦІЯ ───────────────────────────────────────────────────

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
        pred   = final_sil.cpu().numpy()
        diff   = np.abs(pred - target)
        final_iou  = 1.0 - loss_history[-1]
        total_time = search_time + opt_time

        fig = plt.figure(figsize=(18, 9))
        fig.suptitle(
            f"3D Реконструкція  |  IoU={final_iou:.4f}  "
            f"|  elev={best_elev}°  azim={best_azim}°",
            fontsize=14, fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        for ax, data, title, cmap in zip(
            [fig.add_subplot(gs[0, i]) for i in range(4)],
            [target, pred, diff,
             np.stack([target, pred, np.zeros_like(target)], axis=-1)],
            ["Маска відбитку", "Проекція моделі",
             f"Різниця (IoU={final_iou:.3f})",
             "Overlay (R=маска  G=проекція)"],
            ["gray", "gray", "hot", None],
        ):
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        ax4 = fig.add_subplot(gs[1, :2])
        p0 = phase1_end  # межа фаза0/фаза1
        p1 = p0 + (len(loss_history) - p0) // 2  # приблизна межа фаза1/фаза2
        ax4.plot(range(p0), loss_history[:p0],
                 color="#43A047", lw=1.5, label="Фаза 0 (масштаб)")
        ax4.plot(range(p0, p1), loss_history[p0:p1],
                 color="#1565C0", lw=1.5, label="Фаза 1 (груба деформація)")
        ax4.plot(range(p1, len(loss_history)), loss_history[p1:],
                 color="#E65100", lw=1.5, label="Фаза 2 (точна деформація)")
        ax4.axvline(x=p0, color="gray", ls=":", alpha=0.6)
        ax4.axvline(x=p1, color="gray", ls=":", alpha=0.6)
        ax4.axhline(y=loss_history[-1], color="red", ls="--", alpha=0.5,
                    label=f"Фінал={loss_history[-1]:.4f}")
        ax4.set_title("IoU Loss — 3-фазна оптимізація")
        ax4.set_xlabel("Ітерація")
        ax4.set_ylabel("Loss")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis("off")
        table = ax5.table(
            cellText=[
                ["Пошук кута (паралельно)", f"{search_time:.2f} с"],
                ["Оптимізація деформації",  f"{opt_time:.2f} с"],
                ["Разом",                   f"{total_time:.2f} с"],
                ["", ""],
                ["elev / azim",  f"{best_elev}° / {best_azim}°"],
                ["Фінальний IoU", f"{final_iou:.4f}"],
                ["Всього ітерацій", f"{len(loss_history)}"],
                ["Фаза 1 / Фаза 2",
                 f"{phase1_end} / {len(loss_history) - phase1_end}"],
                ["REG / SMOOTH", f"{REG_WEIGHT} / {SMOOTH_WEIGHT}"],
                ["Deform clamp", f"±{DEFORM_CLAMP}"],
            ],
            colLabels=["Параметр", "Значення"],
            loc="center", cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax5.set_title("Зведення результатів", pad=10)

        plt.savefig("reconstruction_result.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ══════════════════════════════════════════════════════════
# МОДУЛЬ 4 — GUI
# ══════════════════════════════════════════════════════════

class FossilApp:
    OBJ_PATH = "3d-model.obj"

    def __init__(self):
        self.image_path: str | None = None
        self.root = tk.Tk()
        self.root.title("3D Реконструкція скам'янілості")
        self.root.geometry("520x420")
        self.root.resizable(False, False)

        self.status_var      = tk.StringVar(value="Готовий до роботи")
        self.threads_var     = tk.IntVar(value=min(4, mp.cpu_count()))
        self.iterations_var  = tk.IntVar(value=500)
        self.search_progress = tk.DoubleVar(value=0)
        self.opt_progress    = tk.DoubleVar(value=0)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        frame_file = tk.LabelFrame(self.root, text="Зображення відбитку", **pad)
        frame_file.pack(fill="x", **pad)
        self.file_label = tk.Label(frame_file, text="Файл не вибрано",
                                   fg="gray", wraplength=400, anchor="w")
        self.file_label.pack(side="left", expand=True, fill="x")
        tk.Button(frame_file, text="Огляд…", command=self._choose_image,
                  width=8).pack(side="right")

        frame_params = tk.LabelFrame(self.root, text="Параметри", **pad)
        frame_params.pack(fill="x", **pad)

        row1 = tk.Frame(frame_params)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Кількість потоків (пошук кута):", width=32,
                 anchor="w").pack(side="left")
        tk.Spinbox(row1, from_=1, to=mp.cpu_count(), textvariable=self.threads_var,
                   width=5).pack(side="left")
        tk.Label(row1, text=f"(макс. {mp.cpu_count()})",
                 fg="gray").pack(side="left", padx=4)

        row2 = tk.Frame(frame_params)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Ітерації оптимізації:", width=32,
                 anchor="w").pack(side="left")
        tk.Spinbox(row2, from_=100, to=2000, increment=100,
                   textvariable=self.iterations_var, width=6).pack(side="left")

        frame_prog = tk.LabelFrame(self.root, text="Прогрес", **pad)
        frame_prog.pack(fill="x", **pad)
        tk.Label(frame_prog, text="Пошук кута:", anchor="w").pack(fill="x")
        ttk.Progressbar(frame_prog, variable=self.search_progress,
                        maximum=100).pack(fill="x", padx=4, pady=2)
        tk.Label(frame_prog, text="Оптимізація:", anchor="w").pack(fill="x")
        ttk.Progressbar(frame_prog, variable=self.opt_progress,
                        maximum=100).pack(fill="x", padx=4, pady=2)

        tk.Label(self.root, textvariable=self.status_var,
                 fg="#1565C0", anchor="w").pack(fill="x", **pad)
        tk.Button(
            self.root, text="▶  Запустити реконструкцію",
            command=self._start,
            bg="#43A047", fg="white",
            font=("Helvetica", 11, "bold"),
            height=2,
        ).pack(fill="x", padx=12, pady=8)

    def _choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Зображення", "*.jpg *.jpeg *.png")]
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
            self._set_status("⚠  Спочатку виберіть зображення!")
            return
        self.search_progress.set(0)
        self.opt_progress.set(0)
        self._set_status("⏳  Обробка зображення…")
        Thread(target=self._pipeline, daemon=True).start()

    def _pipeline(self):
        try:
            self._set_status("⏳  Генерація маски (rembg)…")
            mask = ImageProcessor(self.image_path).generate_mask()
            self._set_status("✔  Маску створено. Завантаження моделі…")

            reconstructor = Fossil3DReconstructor(
                obj_path=self.OBJ_PATH,
                mask=mask,
                img_size=64,
            )

            n_workers = self.threads_var.get()
            self._set_status(
                f"🔍  Пошук кута ({len(SEARCH_ANGLES)} варіантів, {n_workers} потоків)…"
            )
            best_elev, best_azim, best_loss, search_time = (
                reconstructor.find_best_initial_angle(
                    num_workers=n_workers,
                    progress_callback=self._update_search_bar,
                )
            )
            self._update_search_bar(len(SEARCH_ANGLES), len(SEARCH_ANGLES))
            self._set_status(
                f"✔  Кут знайдено за {search_time:.1f} с "
                f"(elev={best_elev}°, azim={best_azim}°). Оптимізація…"
            )

            iters = self.iterations_var.get()
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
                f"✅  Готово! IoU={final_iou:.4f}  |  "
                f"Пошук: {search_time:.1f} с  "
                f"Опт.: {opt_time:.1f} с ({len(loss_history)} іт.)  "
                f"Разом: {total:.1f} с"
            )

            reconstructor.visualize_result(
                final_sil, loss_history, phase1_end,
                best_elev, best_azim, search_time, opt_time,
            )
            self.root.after(0, self._open_result_image)

        except Exception as exc:
            self._set_status(f"❌  Помилка: {exc}")
            import traceback; traceback.print_exc()

    def _open_result_image(self):
        import subprocess, platform
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


# ══════════════════════════════════════════════════════════
# ТОЧКА ВХОДУ
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.freeze_support()
    if not os.path.exists(FossilApp.OBJ_PATH):
        print(f"⚠  {FossilApp.OBJ_PATH} не знайдено.")
    FossilApp().run()