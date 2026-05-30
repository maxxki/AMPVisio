#!/usr/bin/env python3
"""
prepare_dataset_ampullen.py – Automatische Dataset-Erstellung aus ampullen_test.mp4
====================================================================================
Da das Video synthetisch generiert wurde (gen_ampullen_video.py), kennen wir
die exakten Bounding Boxes → kein manuelles Annotieren nötig.

Defektklassen:
  0 = sauber     (klar, keine Partikel)
  1 = partikel   (dunkle Schwebstoffe in der Flüssigkeit)
  2 = luftblase  (Gasblase im Liquid)
  3 = riss       (Haarriss im Glas)

Verwendung:
  python prepare_dataset_ampullen.py
  python prepare_dataset_ampullen.py --video ampullen_test.mp4 --out ./dataset_ampullen
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# ── Exakt dieselben Konstanten wie gen_ampullen_video.py ───────────────
W, H    = 1280, 720
FPS     = 30
SPEED   = 3
SEED    = 42

random.seed(SEED)

# Ampullen-Geometrie
AMP_W   = 38
AMP_H   = 160
NECK_W  = 14
NECK_H  = 55
TOTAL_H = AMP_H + NECK_H
BELT_Y  = H // 2 - TOTAL_H // 2

# Defektklassen
CLEAN    = 0
PARTIKEL = 1
BLASE    = 2
RISS     = 3

DEFECT_NAMES = {CLEAN: "sauber", PARTIKEL: "partikel", BLASE: "luftblase", RISS: "riss"}

LIQUID_COLORS = [
    (200, 220, 255),
    (180, 240, 210),
    (190, 215, 255),
]


# ── Exakt dieselbe Spawn-Logik wie gen_ampullen_video.py ───────────────
def make_ampules(total_frames):
    ampules = []
    spawn_interval = FPS * 2
    prng = random.Random(SEED)

    for i in range(total_frames // spawn_interval + 3):
        spawn = i * spawn_interval + prng.randint(-15, 15)

        r = prng.random()
        if r < 0.40:
            defect = CLEAN
        elif r < 0.60:
            defect = PARTIKEL
        elif r < 0.80:
            defect = BLASE
        else:
            defect = RISS

        liquid_col = prng.choice(LIQUID_COLORS)

        # Partikel
        particles = []
        if defect == PARTIKEL:
            n = prng.randint(3, 8)
            for _ in range(n):
                px = prng.randint(4, AMP_W - 8)
                py = prng.randint(15, AMP_H - 20)
                pr = prng.randint(2, 5)
                particles.append((px, py, pr))

        # Blase
        bubble = None
        if defect == BLASE:
            bx = prng.randint(8, AMP_W - 10)
            by = prng.randint(10, AMP_H - 30)
            br = prng.randint(6, 14)
            bubble = (bx, by, br)

        # Riss
        crack = None
        if defect == RISS:
            side = prng.randint(0, 1)
            sx = 2 if side == 0 else AMP_W - 3
            sy = prng.randint(20, AMP_H - 40)
            segs = []
            cx, cy = sx, sy
            for _ in range(prng.randint(3, 6)):
                ex = cx + prng.randint(-12, 12)
                ey = cy + prng.randint(5, 20)
                ex = max(0, min(AMP_W, ex))
                segs.append(((cx, cy), (ex, ey)))
                cx, cy = ex, ey
            crack = segs

        ampules.append({
            "spawn":      spawn,
            "defect":     defect,
            "liquid_col": liquid_col,
            "particles":  particles,
            "bubble":     bubble,
            "crack":      crack,
        })
    return ampules


def get_visible_boxes(frame_idx, ampules):
    """
    Gibt Liste von (x1, y1, x2, y2, cls) zurück für sichtbare Ampullen.
    Bounding Box umschließt den kompletten Ampullen-Körper inkl. Hals.
    """
    boxes = []
    for amp in ampules:
        ax = (frame_idx - amp["spawn"]) * SPEED - AMP_W

        # Sichtbarkeit prüfen: mindestens 50% des Körpers im Bild
        ax2 = ax + AMP_W
        visible_w = min(ax2, W) - max(ax, 0)
        if visible_w < AMP_W * 0.5:
            continue

        # Bounding Box: Hals-Oberkante bis Körper-Unterkante
        x1 = max(0, int(ax))
        y1 = max(0, BELT_Y - NECK_H)          # Oberkante Hals
        x2 = min(W, int(ax + AMP_W))
        y2 = min(H, BELT_Y + AMP_H)           # Unterkante Körper

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append((x1, y1, x2, y2, amp["defect"]))
    return boxes


def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Pixel-BBox → YOLO-Format (normalisiert, center-based)"""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def draw_debug(frame, boxes):
    """Bounding Boxes auf Frame zeichnen (für Debug-Bilder)"""
    colors = {
        CLEAN:    (0,  200, 80),
        PARTIKEL: (0,  60,  220),
        BLASE:    (0,  180, 220),
        RISS:     (0,  30,  200),
    }
    for x1, y1, x2, y2, cls in boxes:
        col   = colors[cls]
        label = DEFECT_NAMES[cls].upper()
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return frame


def prepare(video_path: str, out_dir: str, every_n: int, val_split: float,
            debug_frames: int):

    out   = Path(out_dir)
    video = Path(video_path)

    if not video.exists():
        raise FileNotFoundError(f"Video nicht gefunden: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ampules      = make_ampules(total_frames)

    defect_counts = {k: sum(1 for a in ampules if a["defect"] == k)
                     for k in DEFECT_NAMES}

    print(f"Video:         {video}  ({total_frames} Frames @ {FPS}fps)")
    print(f"Ampullen total: {len(ampules)}")
    for k, v in defect_counts.items():
        print(f"  {DEFECT_NAMES[k]:12s}: {v}")
    print(f"Jeder {every_n}. Frame → ~{total_frames // every_n} Bilder")
    print()

    # Ordner anlegen
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    debug_dir = out / "debug"
    debug_dir.mkdir(exist_ok=True)

    samples     = []
    debug_saved = 0
    frame_idx   = 0

    print("Extrahiere Frames …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            boxes = get_visible_boxes(frame_idx, ampules)
            if boxes:
                samples.append((frame_idx, frame.copy(), boxes))

                if debug_saved < debug_frames:
                    dbg = draw_debug(frame.copy(), boxes)
                    cv2.imwrite(str(debug_dir / f"debug_{frame_idx:05d}.jpg"), dbg)
                    debug_saved += 1

        frame_idx += 1

    cap.release()
    print(f"  → {len(samples)} annotierte Samples")

    # Train / Val Split
    random.shuffle(samples)
    n_val   = max(1, int(len(samples) * val_split))
    val_set = samples[:n_val]
    trn_set = samples[n_val:]

    # Statistik pro Split und Klasse
    stats = {split: {k: 0 for k in DEFECT_NAMES} for split in ("train", "val")}

    def save_split(items, split):
        for fi, frame, boxes in items:
            name  = f"frame_{fi:05d}"
            img_p = out / "images" / split / f"{name}.jpg"
            lbl_p = out / "labels" / split / f"{name}.txt"

            cv2.imwrite(str(img_p), frame)

            lines = []
            for x1, y1, x2, y2, cls in boxes:
                cx, cy, bw, bh = box_to_yolo(x1, y1, x2, y2, W, H)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats[split][cls] += 1

            lbl_p.write_text("\n".join(lines))

    save_split(trn_set, "train")
    save_split(val_set,  "val")

    # dataset.yaml
    names_block = "\n".join(f"  {k}: {v}" for k, v in DEFECT_NAMES.items())
    yaml_content = f"""# AmpuleGuard Dataset – auto-generiert aus {video.name}
path: {out.resolve()}
train: images/train
val:   images/val

nc: 4
names:
{names_block}
"""
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml_content)

    # Zusammenfassung
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║            AmpuleGuard Dataset fertig               ║")
    print("╠══════════════════════════════════════════════════════╣")
    for split, label in (("train", "Train"), ("val", "Val  ")):
        row = "  ".join(f"{DEFECT_NAMES[k][:6]}:{stats[split][k]:3d}"
                        for k in DEFECT_NAMES)
        n   = len(trn_set) if split == "train" else len(val_set)
        print(f"║  {label}: {n:4d} Bilder  {row}  ║")
    print(f"║  Debug:  {debug_saved:4d} annotierte Vorschau-Bilder               ║")
    print(f"║  YAML:   {yaml_path}")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("Nächster Schritt:")
    print(f"  python train_paket_model.py --data {yaml_path} --base yolov8s.pt --epochs 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AmpuleGuard – Dataset aus synthetischem Backlight-Video generieren"
    )
    parser.add_argument("--video",        default="ampullen_test.mp4",
                        help="Pfad zum Eingabe-Video (default: ampullen_test.mp4)")
    parser.add_argument("--out",          default="./dataset_ampullen",
                        help="Ausgabeordner (default: ./dataset_ampullen)")
    parser.add_argument("--every-n",      type=int,   default=6,
                        help="Jeden N-ten Frame extrahieren (default: 6 → ~5fps)")
    parser.add_argument("--val-split",    type=float, default=0.2,
                        help="Anteil Validierungsdaten (default: 0.2)")
    parser.add_argument("--debug-frames", type=int,   default=10,
                        help="Anzahl Debug-Bilder mit eingezeichneten Boxen")
    args = parser.parse_args()

    prepare(args.video, args.out, args.every_n, args.val_split, args.debug_frames)
