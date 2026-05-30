#!/usr/bin/env python3
"""
gen_ampullen_video.py – Synthetisches Backlight-Inspektionsvideo
=================================================================
Generiert ampullen_test.mp4 (1280x720 @ 30fps) mit:
  - Fließband-Animation (Ampullen von rechts nach links)
  - 4 Defektklassen: sauber, partikel, luftblase, riss
  - Backlight-Effekt (heller Hintergrund, dunkle Ampulle)
  - Exakt dieselben Konstanten wie prepare_dataset_ampullen.py

Verwendung:
  python gen_ampullen_video.py
  python gen_ampullen_video.py --out mein_video.mp4 --duration 60 --fps 30
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# ── Konstanten (identisch zu prepare_dataset_ampullen.py) ─────────────────────
W, H    = 1280, 720
FPS     = 30
SPEED   = 3
SEED    = 42

random.seed(SEED)

AMP_W   = 38
AMP_H   = 160
NECK_W  = 14
NECK_H  = 55
TOTAL_H = AMP_H + NECK_H
BELT_Y  = H // 2 - TOTAL_H // 2

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

# Defekt-Farben für Backlight
COL_PARTIKEL = (30,  30,  20)
COL_BLASE    = (230, 245, 255)
COL_RISS     = (60,  55,  50)
COL_GLASS    = (210, 215, 220)
COL_NECK     = (200, 205, 210)
COL_BELT     = (50,  48,  44)
COL_BG_TOP   = (245, 245, 240)
COL_BG_BOT   = (200, 198, 195)
COL_LIGHT    = (255, 255, 250)


# ── Ampullen-Spawn-Logik ───────────────────────────────────────────────────────
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

        particles = []
        if defect == PARTIKEL:
            n = prng.randint(3, 8)
            for _ in range(n):
                px = prng.randint(4, AMP_W - 8)
                py = prng.randint(15, AMP_H - 20)
                pr = prng.randint(2, 5)
                particles.append((px, py, pr))

        bubble = None
        if defect == BLASE:
            bx = prng.randint(8, AMP_W - 10)
            by = prng.randint(10, AMP_H - 30)
            br = prng.randint(6, 14)
            bubble = (bx, by, br)

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


# ── Hintergrund zeichnen ───────────────────────────────────────────────────────
def draw_background(frame):
    # Backlight: heller Streifen in der Mitte
    for y in range(H):
        t = abs(y - H // 2) / (H // 2)
        c = int(255 * (1.0 - 0.3 * t))
        frame[y, :] = (c, c, c - 5)

    # Fließband
    belt_top = BELT_Y + AMP_H + 4
    belt_bot = belt_top + 28
    cv2.rectangle(frame, (0, belt_top), (W, belt_bot), COL_BELT, -1)

    # Fließband-Rippen
    for x in range(0, W, 32):
        cv2.line(frame, (x, belt_top), (x, belt_bot),
                 (70, 68, 62), 1)

    # Lichtreflex oben auf dem Band
    cv2.rectangle(frame, (0, belt_top), (W, belt_top + 3),
                  (80, 78, 72), -1)


# ── Einzelne Ampulle zeichnen ──────────────────────────────────────────────────
def draw_ampule(frame, ax, amp):
    """
    Zeichnet eine Ampulle mit Backlight-Optik.
    ax = linke Kante der Ampulle (float → int)
    """
    ax = int(ax)
    body_top  = BELT_Y
    body_bot  = BELT_Y + AMP_H
    neck_top  = BELT_Y - NECK_H
    neck_bot  = BELT_Y

    liq = amp["liquid_col"]

    # ── Flaschenkörper ──
    # Äußeres Glas (leicht dunkler Rand)
    cv2.rectangle(frame,
                  (ax,       body_top),
                  (ax+AMP_W, body_bot),
                  COL_GLASS, -1)

    # Flüssigkeit innen (Backlight: leuchtend)
    inner_x1 = ax + 4
    inner_x2 = ax + AMP_W - 4
    inner_y1 = body_top + 6
    inner_y2 = body_bot - 4
    cv2.rectangle(frame,
                  (inner_x1, inner_y1),
                  (inner_x2, inner_y2),
                  liq, -1)

    # ── Hals ──
    neck_x1 = ax + (AMP_W - NECK_W) // 2
    neck_x2 = neck_x1 + NECK_W
    cv2.rectangle(frame,
                  (neck_x1, neck_top),
                  (neck_x2, neck_bot),
                  COL_NECK, -1)

    # Hals-Innen (Flüssigkeit)
    cv2.rectangle(frame,
                  (neck_x1 + 3, neck_top + 4),
                  (neck_x2 - 3, neck_bot),
                  liq, -1)

    # ── Verschluss (Spitze) ──
    tip_x = ax + AMP_W // 2
    cv2.circle(frame, (tip_x, neck_top + 4), 5, COL_GLASS, -1)

    # ── Defekte ──
    defect = amp["defect"]

    if defect == PARTIKEL:
        for px, py, pr in amp["particles"]:
            cv2.circle(frame,
                       (ax + 4 + px, body_top + 6 + py),
                       pr, COL_PARTIKEL, -1)

    elif defect == BLASE:
        bx, by, br = amp["bubble"]
        # Blase: heller als Flüssigkeit
        cv2.circle(frame,
                   (ax + 4 + bx, body_top + 6 + by),
                   br, COL_BLASE, -1)
        # Highlight
        cv2.circle(frame,
                   (ax + 4 + bx - br//3, body_top + 6 + by - br//3),
                   max(1, br//3), (255, 255, 255), -1)

    elif defect == RISS and amp["crack"]:
        for (sx, sy), (ex, ey) in amp["crack"]:
            cv2.line(frame,
                     (ax + sx, body_top + sy),
                     (ax + ex, body_top + ey),
                     COL_RISS, 1)

    # ── Lichtreflex auf Glas ──
    cv2.rectangle(frame,
                  (ax + 1,     body_top),
                  (ax + 5,     body_bot),
                  (240, 242, 244), -1)
    cv2.rectangle(frame,
                  (ax,         body_top),
                  (ax + AMP_W, body_top + 2),
                  (240, 242, 244), -1)

    # ── Glasrand (Outline) ──
    cv2.rectangle(frame,
                  (ax,         body_top),
                  (ax + AMP_W, body_bot),
                  (160, 162, 165), 1)
    cv2.rectangle(frame,
                  (neck_x1,    neck_top),
                  (neck_x2,    neck_bot),
                  (160, 162, 165), 1)


# ── Defekt-Label (Debug-Overlay, optional) ────────────────────────────────────
LABEL_COLORS = {
    CLEAN:    (0, 180, 60),
    PARTIKEL: (0, 80,  220),
    BLASE:    (0, 200, 220),
    RISS:     (0, 40,  200),
}

def draw_label(frame, ax, defect):
    label = DEFECT_NAMES[defect].upper()
    col   = LABEL_COLORS[defect]
    ax    = int(ax)
    cv2.putText(frame, label,
                (ax, BELT_Y - NECK_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)


# ── Header / Footer ───────────────────────────────────────────────────────────
def draw_ui(frame, frame_idx, fps_actual):
    # Header
    cv2.rectangle(frame, (0, 0), (W, 36), (20, 18, 16), -1)
    cv2.putText(frame, "AmpVision · Backlight-Inspektion · SYNTHETIC DATA",
                (10, 24), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 200, 180), 1)
    ts = f"Frame {frame_idx:05d}"
    cv2.putText(frame, ts, (W - 140, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    # Footer Legende
    cv2.rectangle(frame, (0, H - 28), (W, H), (20, 18, 16), -1)
    legend = [
        (CLEAN,    "SAUBER"),
        (PARTIKEL, "PARTIKEL"),
        (BLASE,    "BLASE"),
        (RISS,     "RISS"),
    ]
    x = 10
    for cls, name in legend:
        col = LABEL_COLORS[cls]
        cv2.rectangle(frame, (x, H - 20), (x + 12, H - 8), col, -1)
        cv2.putText(frame, name, (x + 16, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        x += 110


# ── Hauptfunktion ─────────────────────────────────────────────────────────────
def generate(out_path: str, duration: int, fps: int,
             show_labels: bool, preview: bool):
    total_frames = duration * fps
    ampules      = make_ampules(total_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter konnte nicht geöffnet werden: {out_path}")

    print(f"Generiere {out_path} ({W}x{H} @ {fps}fps, {duration}s = {total_frames} Frames)")
    print(f"Ampullen: {len(ampules)}")

    defect_counts = {k: 0 for k in DEFECT_NAMES}

    import time
    t0 = time.perf_counter()

    for frame_idx in range(total_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_background(frame)

        for amp in ampules:
            ax = (frame_idx - amp["spawn"]) * SPEED - AMP_W
            ax2 = ax + AMP_W
            if ax2 < -AMP_W or ax > W + AMP_W:
                continue
            if ax2 < 0 or ax > W:
                continue
            draw_ampule(frame, ax, amp)
            if show_labels:
                draw_label(frame, ax, amp["defect"])

            # Statistik nur beim ersten Sichtbarwerden
            visible_w = min(ax2, W) - max(ax, 0)
            if visible_w >= AMP_W * 0.5 and int(ax) == int((frame_idx - amp["spawn"]) * SPEED - AMP_W):
                defect_counts[amp["defect"]] += 0  # gezählt bei Spawn

        draw_ui(frame, frame_idx, fps)
        writer.write(frame)

        if preview:
            cv2.imshow("AmpVision – Generierung", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Abgebrochen.")
                break

        if frame_idx % (fps * 5) == 0:
            elapsed = time.perf_counter() - t0
            pct = frame_idx / total_frames * 100
            eta = (elapsed / max(frame_idx, 1)) * (total_frames - frame_idx)
            print(f"  {pct:5.1f}%  Frame {frame_idx}/{total_frames}  ETA {eta:.0f}s")

    writer.release()
    if preview:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    size_mb = Path(out_path).stat().st_size / 1_048_576

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║           AmpVision – Video fertig                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Datei   : {out_path:<42}║")
    print(f"║  Größe   : {size_mb:.1f} MB{'':<40}║")
    print(f"║  Dauer   : {duration}s @ {fps}fps = {total_frames} Frames{'':<28}║")
    print(f"║  Render  : {elapsed:.1f}s{'':<43}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Nächster Schritt:                                   ║")
    print("║  python prepare_dataset_ampullen.py                  ║")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AmpVision – Synthetisches Backlight-Inspektionsvideo generieren"
    )
    parser.add_argument("--out",      default="ampullen_test.mp4",
                        help="Ausgabe-Video (default: ampullen_test.mp4)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Videolänge in Sekunden (default: 60)")
    parser.add_argument("--fps",      type=int, default=30,
                        help="Framerate (default: 30)")
    parser.add_argument("--labels",   action="store_true",
                        help="Defekt-Labels ins Video einzeichnen (Debug)")
    parser.add_argument("--preview",  action="store_true",
                        help="Live-Vorschau während der Generierung")
    args = parser.parse_args()

    generate(args.out, args.duration, args.fps, args.labels, args.preview)
