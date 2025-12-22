import random
import cv2
import numpy as np
import torch  # Added to check for CUDA
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = Path("model.pt")
INPUT_DIR  = Path("./my_test_album")
OUTPUT_DIR = Path("./inference_outputs")

NUM_RANDOM_IMAGES = 50
CONF_THRESHOLD = 0.4
OVERLAY_ALPHA = 0.25

# DEVICE SELECTION: Use GPU if available, else CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# SAFE IMAGE LOADER
# =========================
def load_image_safe(path: Path):
    """
    Loads JPG / PNG / WEBP / BMP safely.
    """
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# PICK RANDOM IMAGES
# =========================
images = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
    images.extend(INPUT_DIR.glob(ext))

if not images:
    raise RuntimeError(f"No images found in {INPUT_DIR}")

sample_images = random.sample(
    images,
    min(NUM_RANDOM_IMAGES, len(images))
)

print(f"Running inference on {len(sample_images)} images")

# =========================
# RUN INFERENCE
# =========================
for img_path in sample_images:

    # Load image safely
    img = load_image_safe(img_path)

    # Run inference using the detected DEVICE
    results = model(
        img,
        conf=CONF_THRESHOLD,
        device=DEVICE,  # Fixed: Now uses the dynamic device variable
        verbose=False
    )

    r = results[0]
    img = img.copy()

    if r.boxes is not None and len(r.boxes) > 0:
        overlay = img.copy()

        # =========================
        # FILL PHASE
        # =========================
        for box in r.boxes:
            # Move box to CPU and convert to numpy for CV2 operations
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),  # Blue
                -1
            )

        # Blend overlay
        img = cv2.addWeighted(
            overlay,
            OVERLAY_ALPHA,
            img,
            1 - OVERLAY_ALPHA,
            0
        )

        # =========================
        # BORDER + LABEL PHASE
        # =========================
        for box in r.boxes:
            # Explicitly move data to CPU to avoid errors during CV2 drawing
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            label_name = r.names[cls_id]

            # Border
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Label
            label_text = f"{label_name} {conf:.2f}"
            cv2.putText(
                img,
                label_text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # =========================
    # SAVE OUTPUT
    # =========================
    out_path = OUTPUT_DIR / f"{img_path.stem}_pred.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")

print("Inference complete.")