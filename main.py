"""
fossil_reconstruction.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Пайплайн 3D-реконструкції скам'янілості:
  1. Зображення відбитку → ЧБ маска (rembg)
  2. Паралельний пошук найкращого кута (multiprocessing)
  3. Оптимізація деформації вершин (PyTorch autograd)
  4. Збереження .obj + візуалізація положення у відбитку + час

ПАРАЛЕЛІЗАЦІЯ:
  • multiprocessing.Pool (spawn) — кожен процес ізольований,
    без проблем з PyTorch/GIL.
  • Кількість процесів задається у GUI (1–16).
  • Пошук кута: N процесів ділять між собою список кутів.
  • Оптимізація: torch.no_grad() там де градієнти не потрібні.
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from threading import Thread
import multiprocessing as mp
from functools import partial

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
    """Перетворює зображення відбитку у бінарну маску через rembg."""

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
# (top-level функція — обов'язково для multiprocessing)
# ══════════════════════════════════════════════════════════

def _angle_worker(args: tuple) -> tuple[float, float, float]:
    """
    Запускається у дочірньому процесі.
    Повертає (iou_loss, elev, azim).
    """
    elev, azim, obj_path, mask_array, img_size = args

    device = torch.device("cpu")
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster = RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=10,         # coarse — для швидкості
        cull_backfaces=False,
    )

    # Завантаження моделі у кожному процесі (ізольовано)
    mesh = load_objs_as_meshes([obj_path], device=device)
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = (verts - center).abs().max()
    mesh = mesh.offset_verts(-center).scale_verts(1.0 / scale.item())

    # Підготовка маски
    resized = cv2.resize(mask_array, (img_size, img_size))
    target = torch.tensor(resized, dtype=torch.float32) / 255.0

    # Рендеринг силуету
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

# Повний список кутів для пошуку
SEARCH_ANGLES = [
    (90, 0),   (90, 45),  (90, 90),  (90, 135),
    (90, 180), (90, 225), (90, 270), (90, 315),
    (75, 0),   (75, 90),  (75, 180), (75, 270),
    (60, 0),   (60, 90),  (60, 180), (60, 270),
    (85, 45),  (85, 135), (85, 225), (85, 315),
]


class Fossil3DReconstructor:
    """
    Завантажує 3D-модель, паралельно шукає найкращий кут,
    оптимізує деформацію під маску відбитку.
    """

    def __init__(
        self,
        obj_path: str,
        mask: np.ndarray,
        img_size: int = 256,
        device: str = "cpu",
    ):
        self.obj_path = obj_path
        self.mask_raw = mask            # зберігаємо для воркерів (numpy, серіалізується)
        self.img_size = img_size
        self.device = torch.device(device)

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * self.blend_params.sigma,
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

    def _make_renderer(self, elev: float, azim: float, dist: float = 1.5) -> MeshRenderer:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )

    def _iou_loss(self, pred: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        intersection = (pred * self.target_silhouette).sum()
        union = pred.sum() + self.target_silhouette.sum() - intersection
        return 1.0 - intersection / (union + 1e-8)

    # ── ПАРАЛЕЛЬНИЙ ПОШУК КУТА ─────────────────────────────────────────

    def find_best_initial_angle(
        self,
        num_workers: int = 4,
        progress_callback=None,
    ) -> tuple[float, float, float, float]:
        """
        Паралельний пошук кута через multiprocessing.Pool.
        Кожен процес незалежно рендерить силует і повертає IoU loss.

        Returns: (best_elev, best_azim, best_loss, elapsed_sec)
        """
        # Підготовка аргументів: numpy-масив серіалізується між процесами
        worker_args = [
            (elev, azim, self.obj_path, self.mask_raw, self.img_size)
            for elev, azim in SEARCH_ANGLES
        ]

        t0 = time.time()
        results = []

        # spawn — єдиний безпечний метод для PyTorch + multiprocessing
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_angle_worker, worker_args)):
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(SEARCH_ANGLES))

        elapsed = time.time() - t0
        best_loss, best_elev, best_azim = min(results, key=lambda x: x[0])
        return best_elev, best_azim, best_loss, elapsed

    # ── ОПТИМІЗАЦІЯ ДЕФОРМАЦІЇ ─────────────────────────────────────────

    def optimize_deformation(
        self,
        elev: float,
        azim: float,
        iterations: int = 500,
        progress_callback=None,
    ) -> tuple[Meshes, torch.Tensor, list[float], float]:
        """
        Оптимізує деформацію вершин під маску (однопотоково,
        PyTorch autograd — тут паралелізація не дає переваги).
        """
        t0 = time.time()
        renderer = self._make_renderer(elev, azim)   # один рендерер на весь цикл

        deform_verts = torch.zeros(
            self.mesh.verts_packed().shape,
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([deform_verts], lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        loss_history: list[float] = []
        for i in tqdm(range(iterations), desc="Оптимізація"):
            optimizer.zero_grad()
            deformed = self.mesh.offset_verts(deform_verts)
            sil = renderer(deformed)[0, ..., 3]

            iou_loss = self._iou_loss(sil)
            reg_loss = deform_verts.norm(dim=1).mean() * 0.01
            loss = iou_loss + reg_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())

            if progress_callback:
                progress_callback(i + 1, iterations)

        elapsed = time.time() - t0
        final_mesh = self.mesh.offset_verts(deform_verts.detach())

        with torch.no_grad():
            final_sil = renderer(final_mesh)[0, ..., 3].detach()

        return final_mesh, final_sil, loss_history, elapsed

    # ── ЗБЕРЕЖЕННЯ ─────────────────────────────────────────────────────

    def save_mesh(self, mesh: Meshes, path: str) -> None:
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        save_obj(path, verts, faces)

    # ── ВІЗУАЛІЗАЦІЯ ───────────────────────────────────────────────────

    def visualize_result(
        self,
        final_sil: torch.Tensor,
        loss_history: list[float],
        best_elev: float,
        best_azim: float,
        search_time: float,
        opt_time: float,
    ) -> None:
        """
        Відображає:
          • Маска відбитку
          • Оптимізована проекція
          • Різниця (overlay)
          • Положення моделі у відбитку (overlay)
          • Графік IoU loss
          • Зведення часу
        """
        target = self.target_silhouette.cpu().numpy()
        pred = final_sil.cpu().numpy()
        diff = np.abs(pred - target)
        final_iou = 1.0 - loss_history[-1]
        total_time = search_time + opt_time

        fig = plt.figure(figsize=(18, 9))
        fig.suptitle(
            f"3D Реконструкція скам'янілості  |  IoU = {final_iou:.4f}  "
            f"|  Кут: elev={best_elev}°, azim={best_azim}°",
            fontsize=14, fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        # — Маска відбитку
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(target, cmap="gray")
        ax0.set_title("Маска відбитку")
        ax0.axis("off")

        # — Проекція моделі
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(pred, cmap="gray")
        ax1.set_title("Проекція моделі")
        ax1.axis("off")

        # — Різниця
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(diff, cmap="hot")
        ax2.set_title(f"Різниця (IoU={final_iou:.3f})")
        ax2.axis("off")

        # — Overlay: положення моделі у відбитку
        ax3 = fig.add_subplot(gs[0, 3])
        overlay = np.stack([target, pred, np.zeros_like(target)], axis=-1)
        ax3.imshow(overlay)
        ax3.set_title("Положення моделі\nу відбитку (R=маска, G=проекція)")
        ax3.axis("off")

        # — Графік втрат
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.plot(loss_history, color="#2196F3", linewidth=1.5)
        ax4.set_title("Графік IoU Loss (оптимізація)")
        ax4.set_xlabel("Ітерація")
        ax4.set_ylabel("Loss")
        ax4.grid(True, alpha=0.3)

        # — Зведення часу
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis("off")
        time_data = [
            ["Пошук кута (паралельно)", f"{search_time:.2f} с"],
            ["Оптимізація деформації", f"{opt_time:.2f} с"],
            ["Разом", f"{total_time:.2f} с"],
            ["", ""],
            ["Найкращий elev", f"{best_elev}°"],
            ["Найкращий azim", f"{best_azim}°"],
            ["Фінальний IoU", f"{final_iou:.4f}"],
            ["Кількість ітерацій", f"{len(loss_history)}"],
        ]
        table = ax5.table(
            cellText=time_data,
            colLabels=["Параметр", "Значення"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.6)
        ax5.set_title("Зведення результатів", pad=10)

        plt.savefig("reconstruction_result.png", dpi=150, bbox_inches="tight")
        plt.show()


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

        self.status_var = tk.StringVar(value="Готовий до роботи")
        self.threads_var = tk.IntVar(value=min(4, mp.cpu_count()))
        self.iterations_var = tk.IntVar(value=500)

        # прогрес-бари
        self.search_progress = tk.DoubleVar(value=0)
        self.opt_progress = tk.DoubleVar(value=0)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        # ── Файл ──────────────────────────────────────────
        frame_file = tk.LabelFrame(self.root, text="Зображення відбитку", **pad)
        frame_file.pack(fill="x", **pad)

        self.file_label = tk.Label(frame_file, text="Файл не вибрано",
                                   fg="gray", wraplength=400, anchor="w")
        self.file_label.pack(side="left", expand=True, fill="x")
        tk.Button(frame_file, text="Огляд…", command=self._choose_image,
                  width=8).pack(side="right")

        # ── Параметри ─────────────────────────────────────
        frame_params = tk.LabelFrame(self.root, text="Параметри", **pad)
        frame_params.pack(fill="x", **pad)

        row1 = tk.Frame(frame_params)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Кількість потоків (пошук кута):", width=32,
                 anchor="w").pack(side="left")
        tk.Spinbox(row1, from_=1, to=mp.cpu_count(), textvariable=self.threads_var,
                   width=5).pack(side="left")
        tk.Label(row1, text=f"(макс. {mp.cpu_count()})", fg="gray").pack(side="left", padx=4)

        row2 = tk.Frame(frame_params)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Ітерації оптимізації:", width=32,
                 anchor="w").pack(side="left")
        tk.Spinbox(row2, from_=100, to=2000, increment=100,
                   textvariable=self.iterations_var, width=6).pack(side="left")

        # ── Прогрес ───────────────────────────────────────
        frame_prog = tk.LabelFrame(self.root, text="Прогрес", **pad)
        frame_prog.pack(fill="x", **pad)

        tk.Label(frame_prog, text="Пошук кута:", anchor="w").pack(fill="x")
        ttk.Progressbar(frame_prog, variable=self.search_progress,
                        maximum=100).pack(fill="x", padx=4, pady=2)

        tk.Label(frame_prog, text="Оптимізація:", anchor="w").pack(fill="x")
        ttk.Progressbar(frame_prog, variable=self.opt_progress,
                        maximum=100).pack(fill="x", padx=4, pady=2)

        # ── Статус + кнопка ───────────────────────────────
        tk.Label(self.root, textvariable=self.status_var,
                 fg="#1565C0", anchor="w").pack(fill="x", **pad)

        tk.Button(
            self.root, text="▶  Запустити реконструкцію",
            command=self._start,
            bg="#43A047", fg="white",
            font=("Helvetica", 11, "bold"),
            height=2,
        ).pack(fill="x", padx=12, pady=8)

    # ── Допоміжні ─────────────────────────────────────────────────────

    def _choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Зображення", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.image_path = path
            self.file_label.config(
                text=os.path.basename(path), fg="black"
            )

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

    # ── Основний пайплайн ─────────────────────────────────────────────

    def _pipeline(self):
        try:
            # 1. Маска
            self._set_status("⏳  Генерація маски (rembg)…")
            mask = ImageProcessor(self.image_path).generate_mask()
            self._set_status("✔  Маску створено. Завантаження моделі…")

            reconstructor = Fossil3DReconstructor(
                obj_path=self.OBJ_PATH,
                mask=mask,
            )

            # 2. Паралельний пошук кута
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

            # 3. Оптимізація деформації
            iters = self.iterations_var.get()
            final_mesh, final_sil, loss_history, opt_time = (
                reconstructor.optimize_deformation(
                    best_elev, best_azim,
                    iterations=iters,
                    progress_callback=self._update_opt_bar,
                )
            )
            self._update_opt_bar(iters, iters)

            # 4. Збереження
            out_path = "final_deformed_model.obj"
            reconstructor.save_mesh(final_mesh, out_path)

            total = search_time + opt_time
            self._set_status(
                f"✅  Готово! {out_path}  |  "
                f"Пошук: {search_time:.1f} с  "
                f"Оптимізація: {opt_time:.1f} с  "
                f"Разом: {total:.1f} с"
            )

            # 5. Візуалізація
            reconstructor.visualize_result(
                final_sil, loss_history,
                best_elev, best_azim,
                search_time, opt_time,
            )

        except Exception as exc:
            self._set_status(f"❌  Помилка: {exc}")
            import traceback; traceback.print_exc()

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════
# ТОЧКА ВХОДУ
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ОБОВ'ЯЗКОВО для Windows + macOS (spawn context)
    mp.freeze_support()

    if not os.path.exists(FossilApp.OBJ_PATH):
        print(f"⚠  Попередження: {FossilApp.OBJ_PATH} не знайдено. "
              "Вкажіть правильний шлях у FossilApp.OBJ_PATH.")

    FossilApp().run()