import cv2
import numpy as np
import os

def iou(mask_a, mask_b):
    """Обидві маски — grayscale numpy arrays."""
    a = (mask_a > 127).astype(np.uint8)
    b = (mask_b > 127).astype(np.uint8)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0

def load_and_binarize(path, target_shape=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не знайдено: {path}")
    if target_shape is not None:
        img = cv2.resize(img, (target_shape[1], target_shape[0]))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img

GT_PATH = "masks/image3_mask.png"
gt = load_and_binarize(GT_PATH)
H, W = gt.shape

print(f"Еталон завантажено: {GT_PATH}  ({W}x{H})\n")
print(f"{'Метод':<35} {'IoU':>6}")
print("-" * 43)

results = {}

OVERLAP_PATH = "overlap_result.png"
if os.path.exists(OVERLAP_PATH):
    overlap = cv2.imread(OVERLAP_PATH)
    # overlap = [0=zeros, 1=final_mask(green), 2=gt_mask(red)] 
    model_mask = overlap[:, :, 1]   
    model_mask = cv2.resize(model_mask, (W, H))
    score = iou(gt, model_mask)
    results["combine.py (Nelder-Mead)"] = score
    print(f"{'combine.py (Nelder-Mead)':<35} {score:>6.4f}")
else:
    print(f"{'combine.py (Nelder-Mead)':<35} {'ВІДСУТНІЙ overlap_result.png':>6}")

REMBG_PATH = "masks_code/image3_rembg_mask.png"
if os.path.exists(REMBG_PATH):
    rembg_mask = load_and_binarize(REMBG_PATH, target_shape=(H, W))
    score = iou(gt, rembg_mask)
    results["test_rembg.py (rembg AI)"] = score
    print(f"{'test_rembg.py (rembg AI)':<35} {score:>6.4f}")
else:
    print(f"{'test_rembg.py (rembg AI)':<35} {'ВІДСУТНІЙ masks_code/image3_rembg_mask.png':>6}")

CV_OTSU_PATH = "masks_code/fossil_otsu.png"
CV_SIMPLE_PATH = "masks_code/fossil_simple.png"
CV_ADAPTIVE_PATH = "masks_code/fossil_adaptive.png"

if not os.path.exists(CV_OTSU_PATH):
    print("\n[!] CV маски не знайдено — генерую зараз...")
    image = cv2.imread('images/image3.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh_simple = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 15, 5)
    os.makedirs("masks_code", exist_ok=True)
    cv2.imwrite(CV_SIMPLE_PATH, thresh_simple)
    cv2.imwrite(CV_OTSU_PATH, thresh_otsu)
    cv2.imwrite(CV_ADAPTIVE_PATH, thresh_adaptive)
    print("    CV маски збережено.")

for label, path in [
    ("test_cv.py (Otsu threshold)", CV_OTSU_PATH),
    ("test_cv.py (Simple threshold)", CV_SIMPLE_PATH),
    ("test_cv.py (Adaptive threshold)", CV_ADAPTIVE_PATH),
]:
    if os.path.exists(path):
        m = load_and_binarize(path, target_shape=(H, W))
        score = iou(gt, m)
        results[label] = score
        print(f"{label:<35} {score:>6.4f}")

PYTORCH3D_PATH = "masks_code/pytorch3d_best.png"
if os.path.exists(PYTORCH3D_PATH):
    p3d_mask = load_and_binarize(PYTORCH3D_PATH, target_shape=(H, W))
    score = iou(gt, p3d_mask)
    results["render_silhouette.py (PyTorch3D)"] = score
    print(f"{'render_silhouette.py (PyTorch3D)':<35} {score:>6.4f}")
else:
    print(f"\n[!] render_silhouette.py (PyTorch3D): маска не знайдена.")
    print(f"    Збери найкращу проекцію з render_silhouette.py і збережи як:")
    print(f"    masks_code/pytorch3d_best.png")
    print(f"    Потім запусти цей скрипт ще раз.")

if results:
    print("\n" + "=" * 43)
    best = max(results, key=results.get)
    print(f"Найкращий метод: {best}")
    print(f"IoU:             {results[best]:.4f}")
    print("=" * 43)

   