#!/usr/bin/env python3
"""
prepare_dataset.py – Automatische Dataset-Erstellung aus paket_test.mp4
=======================================================================
Da das Video synthetisch generiert wurde, kennen wir die exakten
Bounding Boxes → kein manuelles Annotieren nötig.

Klassen:
  0 = intakt  (PKG)
  1 = beschaedigt (DMG)

Verwendung:
  python prepare_dataset.py
  python prepare_dataset.py --video paket_test.mp4 --out ./dataset
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Dieselben Konstanten wie gen_test_video.py ─────────────────────────
W, H    = 1280, 720
FPS     = 30
SPEED   = 5
SEED    = 42

random.seed(SEED)

CLS_INTACT   = 0
CLS_DAMAGED  = 1


def make_packets(total_frames):
    """Exakt dieselbe Logik wie gen_test_video.py"""
    packets = []
    spawn_interval = FPS * 2
    y_positions    = [360, 400, 440]
    rng = random.Random(SEED)

    for i in range(total_frames // spawn_interval + 2):
        spawn = i * spawn_interval + rng.randint(-10, 10)
        w     = rng.randint(130, 200)
        h     = rng.randint(70, 110)
        y     = rng.choice(y_positions) - h // 2
        dmg   = rng.random() < 0.28
        packets.append({"spawn": spawn, "y": y, "w": w, "h": h, "dmg": dmg})
    return packets


def get_visible_boxes(frame_idx, packets):
    """Gibt Liste von (x1,y1,x2,y2,cls) zurück für sichtbare Pakete"""
    boxes = []
    for p in packets:
        x = (frame_idx - p["spawn"]) * SPEED - p["w"]
        x2 = x + p["w"]
        y1 = p["y"]
        y2 = p["y"] + p["h"]

        # Nur Pakete die mindestens zur Hälfte sichtbar sind
        visible_w = min(x2, W) - max(x, 0)
        if visible_w < p["w"] * 0.5:
            continue

        # Clips auf Frame-Grenzen
        x1c = max(0, int(x))
        y1c = max(0, int(y1))
        x2c = min(W, int(x2))
        y2c = min(H, int(y2))

        if x2c <= x1c or y2c <= y1c:
            continue

        cls = CLS_DAMAGED if p["dmg"] else CLS_INTACT
        boxes.append((x1c, y1c, x2c, y2c, cls))
    return boxes


def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Konvertiert Pixel-Bbox → YOLO-Format (normalisiert, center)"""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def draw_debug(frame, boxes):
    """Bounding Boxes auf Frame zeichnen (für Debug-Bilder)"""
    for x1, y1, x2, y2, cls in boxes:
        col  = (0, 80, 200) if cls == CLS_DAMAGED else (0, 180, 80)
        label = "DMG" if cls == CLS_DAMAGED else "PKG"
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return frame


def prepare(video_path: str, out_dir: str, every_n: int, val_split: float,
            debug_frames: int):

    out   = Path(out_dir)
    video = Path(video_path)

    if not video.exists():
        raise FileNotFoundError(f"Video nicht gefunden: {video}")

    cap = cv2.VideoCapture(str(video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    packets      = make_packets(total_frames)

    print(f"Video:        {video}  ({total_frames} Frames)")
    print(f"Pakete total: {len(packets)}  ({sum(p['dmg'] for p in packets)} beschädigt)")
    print(f"Jeder {every_n}. Frame → ~{total_frames // every_n} Bilder")

    # Ordner anlegen
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    debug_dir = out / "debug"
    debug_dir.mkdir(exist_ok=True)

    samples     = []   # (frame_idx, frame, boxes)
    debug_saved = 0
    frame_idx   = 0

    print("Extrahiere Frames …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            boxes = get_visible_boxes(frame_idx, packets)
            if boxes:                        # Frames ohne Pakete überspringen
                samples.append((frame_idx, frame.copy(), boxes))

                # Debug-Bild mit eingezeichneten Boxen speichern
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

    stats = {"train": {"intact": 0, "damaged": 0},
             "val":   {"intact": 0, "damaged": 0}}

    def save_split(items, split):
        for fi, frame, boxes in items:
            name = f"frame_{fi:05d}"
            img_p = out / "images" / split / f"{name}.jpg"
            lbl_p = out / "labels" / split / f"{name}.txt"

            cv2.imwrite(str(img_p), frame)

            lines = []
            for x1, y1, x2, y2, cls in boxes:
                cx, cy, bw, bh = box_to_yolo(x1, y1, x2, y2, W, H)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                key = "damaged" if cls == CLS_DAMAGED else "intact"
                stats[split][key] += 1

            lbl_p.write_text("\n".join(lines))

    save_split(trn_set, "train")
    save_split(val_set,  "val")

    # dataset.yaml schreiben
    yaml_content = f"""# PacketGuard Dataset – auto-generiert aus {video.name}
path: {out.resolve()}
train: images/train
val:   images/val

nc: 2
names:
  0: intakt
  1: beschaedigt
"""
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml_content)

    # Zusammenfassung
    print()
    print("╔══════════════════════════════════════════╗")
    print("║        Dataset fertig                    ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Train:  {len(trn_set):4d} Bilder"
          f"  (intakt {stats['train']['intact']:3d} / dmg {stats['train']['damaged']:3d})  ║")
    print(f"║  Val:    {len(val_set):4d} Bilder"
          f"  (intakt {stats['val']['intact']:3d}   / dmg {stats['val']['damaged']:3d})   ║")
    print(f"║  Debug:  {debug_saved:4d} annotierte Vorschau-Bilder           ║")
    print(f"║  YAML:   {yaml_path}  ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("Nächster Schritt:")
    print(f"  python train_paket_model.py --data {yaml_path} --base yolov8n.pt --epochs 30")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset aus synthetischem Fließband-Video generieren"
    )
    parser.add_argument("--video",       default="paket_test.mp4")
    parser.add_argument("--out",         default="./dataset")
    parser.add_argument("--every-n",     type=int,   default=6,
                        help="Jeden N-ten Frame extrahieren (default: 6 → ~5fps)")
    parser.add_argument("--val-split",   type=float, default=0.2,
                        help="Anteil Validierungsdaten (default: 0.2)")
    parser.add_argument("--debug-frames",type=int,   default=10,
                        help="Anzahl Debug-Bilder mit eingezeichneten Boxen")
    args = parser.parse_args()

    prepare(args.video, args.out, args.every_n, args.val_split, args.debug_frames)
