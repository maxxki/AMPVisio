#!/usr/bin/env python3
"""
train_paket_model_hardened.py – Reproduzierbares YOLOv8 Fine-Tuning
====================================================================
Fixes aus dem Review:
  [TRAIN-1] Reproduzierbarer Seed (random / numpy / torch)
  [TRAIN-2] Dataset-Validierung vor Training
  [TRAIN-3] Pfad-Injection Prevention
  [TRAIN-4] Hash-Export nach Training (für Model Integrity Check)

Verwendung:
  python train_paket_model_hardened.py --data ./dataset/dataset.yaml
  TRAIN_SEED=123 python train_paket_model_hardened.py --data ./dataset/dataset.yaml
"""

import argparse
import hashlib
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PacketGuard-Train")

SEED = int(os.getenv("TRAIN_SEED", "42"))


# ──────────────────────────────────────────────
# [TRAIN-1] REPRODUZIERBARER SEED
# ──────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
    log.info("[TRAIN-1] Seed gesetzt: %d", seed)


# ──────────────────────────────────────────────
# [TRAIN-3] PFAD-INJECTION PREVENTION
# ──────────────────────────────────────────────
def validate_yaml_path(yaml_path: str) -> Path:
    p = Path(yaml_path).resolve()
    cwd = Path.cwd()
    try:
        p.relative_to(cwd)
    except ValueError:
        raise ValueError(
            f"[TRAIN-3] YAML-Pfad außerhalb des Arbeitsverzeichnisses: {p}"
        )
    if p.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"[TRAIN-3] Nur .yaml/.yml erlaubt, nicht: {p.suffix}")
    return p


# ──────────────────────────────────────────────
# [TRAIN-2] DATASET VALIDIERUNG
# ──────────────────────────────────────────────
def validate_dataset(yaml_path: Path) -> dict:
    import yaml

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    required = {"path", "train", "val", "nc", "names"}
    missing  = required - set(cfg.keys())
    if missing:
        raise ValueError(f"[TRAIN-2] Fehlende Felder in YAML: {missing}")

    dataset_root = Path(cfg["path"]).resolve()
    train_img    = dataset_root / cfg["train"]
    val_img      = dataset_root / cfg["val"]

    errors = []
    if not train_img.exists():
        errors.append(f"Train-Bilder nicht gefunden: {train_img}")
    if not val_img.exists():
        errors.append(f"Val-Bilder nicht gefunden: {val_img}")

    if errors:
        raise FileNotFoundError(
            "[TRAIN-2] Dataset-Validierung fehlgeschlagen:\n  " + "\n  ".join(errors)
        )

    train_count = len(list(train_img.glob("*.jpg"))) + len(list(train_img.glob("*.png")))
    val_count   = len(list(val_img.glob("*.jpg")))   + len(list(val_img.glob("*.png")))

    if train_count < 10:
        log.warning("[TRAIN-2] Sehr wenige Trainingsbilder: %d (empfohlen: ≥200)", train_count)
    if val_count < 5:
        log.warning("[TRAIN-2] Sehr wenige Validierungsbilder: %d (empfohlen: ≥50)", val_count)

    log.info("[TRAIN-2] Dataset OK | Train: %d | Val: %d | Klassen: %d",
             train_count, val_count, cfg["nc"])
    return cfg


# ──────────────────────────────────────────────
# [TRAIN-4] HASH-EXPORT nach Training
# ──────────────────────────────────────────────
def export_model_hash(model_path: Path):
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    h = sha256.hexdigest()

    hash_file = model_path.with_suffix(".sha256")
    hash_file.write_text(f"{h}  {model_path.name}\n")

    log.info("[TRAIN-4] SHA-256: %s", h)
    log.info("[TRAIN-4] Hash-Datei: %s", hash_file)
    log.info("")
    log.info("  → In paket_scanner_hardened.py eintragen:")
    log.info("    MODEL_HASHES = {")
    log.info('      "%s": "%s"', model_path.name, h)
    log.info("    }")
    return h


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train(args):
    set_seed(SEED)

    # [TRAIN-3] Pfad validieren
    try:
        yaml_path = validate_yaml_path(args.data)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(1)

    # [TRAIN-2] Dataset validieren
    try:
        cfg = validate_dataset(yaml_path)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(1)

    log.info("╔══════════════════════════════════════╗")
    log.info("║   PacketGuard – Hardened Training     ║")
    log.info("╚══════════════════════════════════════╝")
    log.info("  Basismodell : %s", args.base)
    log.info("  Dataset     : %s", yaml_path)
    log.info("  Klassen     : %s", cfg.get("names"))
    log.info("  Epochen     : %d", args.epochs)
    log.info("  Seed        : %d", SEED)

    from ultralytics import YOLO
    model = YOLO(args.base)

    results = model.train(
        data         = str(yaml_path),
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        name         = "paket_guard_hardened",
        project      = "runs/train",
        patience     = 20,
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.01,
        seed         = SEED,          # [TRAIN-1] Seed im Trainer
        deterministic= True,          # [TRAIN-1]
        augment      = True,
        flipud       = 0.3,
        fliplr       = 0.5,
        mosaic       = 1.0,
        degrees      = 10,
        translate    = 0.1,
        scale        = 0.3,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        device       = args.device,
        workers      = 4,
        plots        = True,
        save         = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"

    if best.exists():
        log.info("✓ Training abgeschlossen: %s", best)
        # [TRAIN-4] Hash exportieren
        export_model_hash(best)
    else:
        log.error("Kein best.pt gefunden – Training evtl. fehlgeschlagen.")
        sys.exit(1)

    # Validierung
    log.info("Starte Validierung …")
    val = model.val(data=str(yaml_path))
    log.info("  mAP50   : %.3f", val.box.map50)
    log.info("  mAP50-95: %.3f", val.box.map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PacketGuard – Hardened YOLOv8 Fine-Tuning"
    )
    parser.add_argument("--data",   required=True,    help="Pfad zu dataset.yaml")
    parser.add_argument("--base",   default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default="")
    train(parser.parse_args())
