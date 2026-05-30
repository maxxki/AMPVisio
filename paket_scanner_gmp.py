#!/usr/bin/env python3
"""
paket_scanner_gmp.py – PacketGuard AI · Level 2 + GMP Audit-Trail
==================================================================
Integriert gmp_audit_trail.GMPAuditTrail als Drop-In-Ersatz für AuditLogger.

Änderungen gegenüber paket_scanner.py (Level 2):
  [GMP-1]  AuditLogger → GMPAuditTrail (SQLite, HMAC-Chain, ALCOA+)
  [GMP-2]  GMPUser + Session-Management (Operator/QA-Rollen)
  [GMP-3]  Batch-ID via ENV oder CLI (--batch-id)
  [GMP-4]  BATCH_START / BATCH_END Events automatisch
  [GMP-5]  ALARM-Event bei Defektrate > GMP_ALARM_THRESHOLD
  [GMP-6]  --verify-log ersetzt durch --verify-audit (SQLite-DB)
  [GMP-7]  --export-audit für FDA-konformen CSV/JSON-Export
  [GMP-8]  GMPConfig.validate() im Startup (Production-Guard)
  [GMP-9]  Graceful Shutdown: LOGOUT + BATCH_END immer geschrieben

Alle Level-2-Fixes bleiben unverändert:
  [BUG-1]  IoU-Tracker · [BUG-2] Trusted Model Dir · [BUG-3] Prod Guard
  [BUG-4]  (ersetzt durch GMPAuditTrail) · [BUG-5] Backpressure
  [BUG-6]  Circuit Breaker · [BUG-7] Kill-Switch · [SUB-8] Source Validation
  [SUB-10] Rate Limit (via GMPConfig.LOG_MAX_PER_SEC)

Verwendung:
  python paket_scanner_gmp.py --demo
  python paket_scanner_gmp.py --demo --batch-id 20260530-0001
  python paket_scanner_gmp.py --source 0 --user-id op01 --user-name "Max Kiefer" --role operator

  # Chain verifizieren
  python paket_scanner_gmp.py --verify-audit audit_gmp.db

  # FDA-Export
  python paket_scanner_gmp.py --export-audit audit_gmp.db --out export.csv --format csv
  python paket_scanner_gmp.py --export-audit audit_gmp.db --batch 20260530-0001 --out export.json --format json

  # Produktion
  ENVIRONMENT=production GMP_HMAC_KEY=<secret> GMP_REZEPTUR_VERSION=v2.1.0 \\
    python paket_scanner_gmp.py --source 0 --model /opt/models/best.pt \\
    --user-id op01 --role operator --batch-id 20260530-0001

Emergency Stop:
  touch /tmp/PACKETGUARD_STOP
  rm    /tmp/PACKETGUARD_STOP
"""

import argparse
import datetime
import fcntl
import hashlib
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Headless-Umgebung: GUI-Stubs für opencv-python-headless ──────────────────
# opencv-python-headless hat imshow/waitKey/destroyAllWindows zwar als Symbole,
# wirft aber cv2.error zur Laufzeit ("function is not implemented").
# Wir ersetzen sie bedingungslos durch No-Ops, damit der Demo-Modus läuft.
cv2.imshow            = lambda *a, **kw: None
cv2.waitKey           = lambda *a, **kw: -1
cv2.destroyAllWindows = lambda *a, **kw: None
cv2.namedWindow       = lambda *a, **kw: None

# ── GMP Audit-Trail importieren ───────────────────────────────────────────────
try:
    from gmp_audit_trail import GMPAuditTrail, GMPUser, GMPConfig, SecurityException
except ImportError as e:
    print(f"[FATAL] gmp_audit_trail.py nicht gefunden: {e}")
    print("  → Stelle sicher, dass gmp_audit_trail.py im selben Verzeichnis liegt.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    ENVIRONMENT        = os.getenv("ENVIRONMENT",        "development")  # [BUG-3]

    # Inferenz
    CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD",     "0.65"))
    IOU_THRESHOLD      = float(os.getenv("IOU_THRESHOLD",      "0.45"))
    EJECT_CONF_MIN     = float(os.getenv("EJECT_CONF_MIN",     "0.80"))
    FRAMES_CONFIRM     = int(os.getenv("FRAMES_CONFIRM",       "2"))
    EJECT_COOLDOWN     = float(os.getenv("EJECT_COOLDOWN",     "1.5"))

    # Klassen
    INTACT_CLASS_ID    = int(os.getenv("INTACT_CLASS_ID",      "0"))
    DAMAGED_CLASS_ID   = int(os.getenv("DAMAGED_CLASS_ID",     "1"))

    # [BUG-2] Trusted Model Directory
    TRUSTED_MODEL_DIR  = Path(os.getenv("TRUSTED_MODEL_DIR",   "./models"))
    MODEL_HASHES: dict = {
        "best.pt": "24ca5462c8393b6c1384d057ee5fa8ff310a32bcd995dd3b975b4d0ebdc3aa14"
    }

    # [BUG-3] Hash-Skip nur in Dev
    SKIP_HASH_CHECK    = os.getenv("SKIP_HASH_CHECK", "0") == "1"

    # [BUG-1] Tracker
    IOU_MATCH_THRESH   = float(os.getenv("IOU_MATCH_THRESH",   "0.40"))
    TRACK_MAX_LOST     = int(os.getenv("TRACK_MAX_LOST",        "8"))

    # [BUG-5] Backpressure
    TARGET_FPS         = float(os.getenv("TARGET_FPS",         "30.0"))
    MAX_FRAME_SKIP     = int(os.getenv("MAX_FRAME_SKIP",        "4"))

    # [BUG-6] Circuit Breaker
    CB_FAILURE_LIMIT   = int(os.getenv("CB_FAILURE_LIMIT",     "3"))

    # [BUG-7] Kill-Switch
    STOP_FILE          = Path(os.getenv("STOP_FILE",  "/tmp/PACKETGUARD_STOP"))

    # [SUB-8] Erlaubtes Source-Basisverzeichnis
    ALLOWED_SOURCE_DIR = Path(os.getenv("ALLOWED_SOURCE_DIR",  ".")).resolve()

    # [GMP-5] Alarm-Schwelle Defektrate
    GMP_ALARM_THRESHOLD = float(os.getenv("GMP_ALARM_THRESHOLD", "0.05"))  # 5 %

    # Logging
    LOG_LEVEL          = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def is_production(cls) -> bool:
        return cls.ENVIRONMENT.lower() in ("production", "prod")


logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PacketGuard")


# ──────────────────────────────────────────────────────────────────────────────
# [BUG-3] PRODUCTION GUARD
# ──────────────────────────────────────────────────────────────────────────────
def check_production_safety():
    if Config.is_production() and Config.SKIP_HASH_CHECK:
        raise RuntimeError(
            "[BUG-3] SKIP_HASH_CHECK=1 in ENVIRONMENT=production ist verboten.\n"
            "        Setze ENVIRONMENT=development oder entferne SKIP_HASH_CHECK."
        )
    if Config.is_production():
        log.info("[ENV] Produktionsmodus aktiv – alle Safeguards erzwungen.")
    else:
        log.warning("[ENV] Development-Modus. Nicht für Produktion verwenden.")


# ──────────────────────────────────────────────────────────────────────────────
# [BUG-2] MODEL VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────
def verify_model(model_path: str) -> Path:
    p = Path(model_path).resolve()
    try:
        p.relative_to(Config.TRUSTED_MODEL_DIR.resolve())
    except ValueError:
        if not Config.SKIP_HASH_CHECK:
            raise ValueError(
                f"[BUG-2] Modell liegt außerhalb TRUSTED_MODEL_DIR:\n"
                f"  Modell : {p}\n"
                f"  Erlaubt: {Config.TRUSTED_MODEL_DIR.resolve()}"
            )
        log.warning("[BUG-2] Trusted-Dir-Check übersprungen (SKIP_HASH_CHECK=1)")

    if not p.exists():
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {p}")

    known_hash = Config.MODEL_HASHES.get(p.name)
    if known_hash:
        log.info("[BUG-2] Prüfe SHA-256 für %s …", p.name)
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual != known_hash:
            raise ValueError(
                f"[BUG-2] Integrity Check FAILED: {p.name}\n"
                f"  Erwartet : {known_hash}\n"
                f"  Berechnet: {actual}"
            )
        log.info("[BUG-2] ✓ Hash OK")
    else:
        log.warning("[BUG-2] Kein Hash für '%s' hinterlegt – nur Dir-Check.", p.name)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# [SUB-8] SOURCE VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
def validate_source(source: str):
    if source.isdigit():
        idx = int(source)
        if not (0 <= idx <= 9):
            raise ValueError(f"Kamera-Index außerhalb 0–9: {idx}")
        return idx
    p = Path(source).resolve()
    try:
        p.relative_to(Config.ALLOWED_SOURCE_DIR)
    except ValueError:
        raise ValueError(
            f"[SUB-8] Pfad außerhalb ALLOWED_SOURCE_DIR:\n"
            f"  Pfad   : {p}\n"
            f"  Erlaubt: {Config.ALLOWED_SOURCE_DIR}"
        )
    if not p.exists():
        raise FileNotFoundError(f"Videodatei nicht gefunden: {p}")
    if p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise ValueError(f"Dateityp nicht erlaubt: {p.suffix}")
    return str(p)


# ──────────────────────────────────────────────────────────────────────────────
# [BUG-1] OBJECT TRACKER (IoU-Matching) – unverändert
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Track:
    id:          int
    bbox:        Tuple[int,int,int,int]
    dmg_streak:  int   = 0
    ok_streak:   int   = 0
    lost_frames: int   = 0
    age:         int   = 0
    last_conf:   float = 0.0
    is_damaged:  bool  = False


def iou(a: Tuple, b: Tuple) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter)


class ObjectTracker:
    def __init__(self):
        self._lock:    threading.RLock  = threading.RLock()
        self._tracks:  Dict[int, Track] = {}
        self._next_id: int = 0

    def update(self, detections: List[dict]) -> List[dict]:
        with self._lock:
            alive = {tid: t for tid,t in self._tracks.items() if t.lost_frames == 0}
            matched_tracks = set()
            results = []
            for det in detections:
                bbox = det["bbox"]
                best_tid, best_iou_val = None, Config.IOU_MATCH_THRESH
                for tid, track in alive.items():
                    if tid in matched_tracks: continue
                    score = iou(bbox, track.bbox)
                    if score > best_iou_val:
                        best_iou_val, best_tid = score, tid
                if best_tid is not None:
                    t = self._tracks[best_tid]
                    t.bbox = bbox; t.last_conf = det["conf"]
                    t.is_damaged = det["is_damaged"]; t.age += 1; t.lost_frames = 0
                    if det["is_damaged"]: t.dmg_streak += 1; t.ok_streak = 0
                    else:                t.ok_streak  += 1; t.dmg_streak = 0
                    matched_tracks.add(best_tid)
                    det["track_id"] = best_tid
                else:
                    tid = self._next_id; self._next_id += 1
                    self._tracks[tid] = Track(
                        id=tid, bbox=bbox,
                        dmg_streak = 1 if det["is_damaged"] else 0,
                        ok_streak  = 0 if det["is_damaged"] else 1,
                        last_conf  = det["conf"], is_damaged = det["is_damaged"],
                    )
                    det["track_id"] = tid
                results.append(det)
            for tid, t in list(self._tracks.items()):
                if tid not in matched_tracks:
                    t.lost_frames += 1
                    if t.lost_frames > Config.TRACK_MAX_LOST:
                        del self._tracks[tid]
            return results

    def get_track(self, tid: int) -> Optional[Track]:
        return self._tracks.get(tid)


# ──────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE – unverändert
# ──────────────────────────────────────────────────────────────────────────────
class DecisionEngine:
    def __init__(self, tracker: ObjectTracker,
                 frames_required: int   = Config.FRAMES_CONFIRM,
                 min_conf:        float = Config.EJECT_CONF_MIN):
        self.tracker         = tracker
        self.frames_required = frames_required
        self.min_conf        = min_conf
        self._ejected_ids:   set = set()
        self._lock           = threading.RLock()

    def evaluate(self, track_id: int, conf: float) -> bool:
        with self._lock:
            if track_id in self._ejected_ids:
                return False
            track = self.tracker.get_track(track_id)
            if track is None: return False
            if conf < self.min_conf: return False
            if track.dmg_streak >= self.frames_required:
                self._ejected_ids.add(track_id)
                if len(self._ejected_ids) > 500:
                    self._ejected_ids = set(list(self._ejected_ids)[-500:])
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# [BUG-6] CIRCUIT BREAKER + ACTUATOR – unverändert
# ──────────────────────────────────────────────────────────────────────────────
class ActuatorLayer:
    def __init__(self, mode="demo", shadow=False, serial_port=None, gpio_pin=17):
        self.mode           = mode
        self.shadow         = shadow
        self.last_eject     = 0.0
        self.gpio_pin       = gpio_pin
        self._ser           = None
        self._GPIO          = None
        self._lock          = threading.RLock()
        self._failure_count = 0
        self._open          = True
        if shadow:
            log.info("[SHADOW] Kein physischer Auswurf – nur Logging.")
            return
        if mode == "serial" and serial_port:
            import serial
            self._ser = serial.Serial(serial_port, 9600, timeout=1)
            log.info("[ACTUATOR] Seriell: %s", serial_port)
        elif mode == "gpio":
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(gpio_pin, GPIO.OUT, initial=GPIO.LOW)
            self._GPIO = GPIO
            log.info("[ACTUATOR] GPIO Pin %d", gpio_pin)

    def trigger(self) -> bool:
        with self._lock:
            if not self._open:
                log.error("[CB] Circuit Breaker OFFEN – Actuator deaktiviert!")
                return False
            now = time.time()
            if now - self.last_eject < Config.EJECT_COOLDOWN:
                return False
            self.last_eject = now
        if self.shadow:
            log.info("[SHADOW] Auswurf (simuliert)")
            return True
        ok = self._do_eject()
        with self._lock:
            if not ok:
                self._failure_count += 1
                log.error("[CB] Actuator-Fehler %d/%d",
                          self._failure_count, Config.CB_FAILURE_LIMIT)
                if self._failure_count >= Config.CB_FAILURE_LIMIT:
                    self._open = False
                    log.critical("[CB] Circuit Breaker ausgelöst! Neustart notwendig.")
            else:
                self._failure_count = 0
        return ok

    def _do_eject(self) -> bool:
        try:
            if self.mode == "serial" and self._ser:
                self._ser.write(b"EJECT\n")
            elif self.mode == "gpio" and self._GPIO:
                self._GPIO.output(self.gpio_pin, self._GPIO.HIGH)
                time.sleep(0.2)
                self._GPIO.output(self.gpio_pin, self._GPIO.LOW)
            else:
                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\033[91m[{ts}] ⚠  AUSWURF\033[0m")
            return True
        except Exception as e:
            log.error("[ACTUATOR] Fehler: %s", e)
            return False

    def close(self):
        for obj, method in [(self._ser, "close"), (self._GPIO, "cleanup")]:
            if obj:
                try: getattr(obj, method)()
                except: pass


# ──────────────────────────────────────────────────────────────────────────────
# OSD – unverändert
# ──────────────────────────────────────────────────────────────────────────────
COL_OK     = (0, 220, 110)
COL_DAMAGE = (0, 60, 255)
COL_TEXT   = (230, 230, 230)
COL_ACCENT = (0, 165, 240)
COL_WARN   = (0, 140, 255)
COL_STOP   = (0, 0, 220)


class OSD:
    def __init__(self, shadow=False):
        self.ok_count    = 0
        self.dmg_count   = 0
        self.eject_count = 0
        self.fps_buf     = deque(maxlen=30)
        self.alert_until = 0.0
        self.shadow      = shadow

    def draw(self, frame, fps):
        h, w = frame.shape[:2]
        now  = time.time()
        self.fps_buf.append(fps)
        avg_fps = sum(self.fps_buf) / len(self.fps_buf)
        stop_active = Config.STOP_FILE.exists()
        if stop_active:
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(w,h),(0,0,120),-1)
            cv2.addWeighted(ov,0.25,frame,0.75,0,frame)
        if now < self.alert_until and not stop_active:
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(w,h),(0,0,180),-1)
            cv2.addWeighted(ov,0.14,frame,0.86,0,frame)
        cv2.rectangle(frame,(0,0),(w,44),(15,15,25),-1)
        mode = " [SHADOW]" if self.shadow else (" [STOP]" if stop_active else "")
        cv2.putText(frame,f"PacketGuard AI · GMP Build{mode}",
                    (10,29),cv2.FONT_HERSHEY_DUPLEX,0.62,
                    COL_STOP if stop_active else COL_ACCENT,1)
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]
        cv2.putText(frame,ts,(w-160,29),cv2.FONT_HERSHEY_DUPLEX,0.58,COL_TEXT,1)
        px = w-200
        cv2.rectangle(frame,(px,54),(w,225),(15,15,25),-1)
        cv2.rectangle(frame,(px,54),(w,225),(40,40,65),1)
        def stat(label,val,y,col):
            cv2.putText(frame,label,(px+8,y),cv2.FONT_HERSHEY_SIMPLEX,0.38,(120,120,135),1)
            cv2.putText(frame,str(val),(px+8,y+18),cv2.FONT_HERSHEY_SIMPLEX,0.66,col,2)
        stat("INTAKT",    self.ok_count,    70,  COL_OK)
        stat("BESCHÄDIGT",self.dmg_count,   110, COL_DAMAGE)
        stat("AUSWÜRFE",  self.eject_count, 150, COL_WARN)
        stat(f"FPS {avg_fps:.0f}","",       190, COL_TEXT)
        cv2.putText(frame,f"CONF≥{Config.EJECT_CONF_MIN:.0%}  FRAMES≥{Config.FRAMES_CONFIRM}",
                    (px+6,214),cv2.FONT_HERSHEY_SIMPLEX,0.32,(100,100,110),1)
        if stop_active:
            txt="⛔  EMERGENCY STOP AKTIV"
            (bw,_),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,0.9,2)
            bx=(w-bw)//2
            cv2.rectangle(frame,(bx-12,h-62),(bx+bw+12,h-10),(0,0,160),-1)
            cv2.putText(frame,txt,(bx,h-22),cv2.FONT_HERSHEY_DUPLEX,0.9,(255,255,255),2)
        elif now < self.alert_until:
            txt="⚠  BESCHÄDIGT – AUSWURF"
            (bw,_),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,0.85,2)
            bx=(w-bw)//2
            cv2.rectangle(frame,(bx-12,h-62),(bx+bw+12,h-10),(0,0,180),-1)
            cv2.putText(frame,txt,(bx,h-22),cv2.FONT_HERSHEY_DUPLEX,0.85,(255,255,255),2)
        return frame

    def reg_ok(self):    self.ok_count    += 1
    def reg_dmg(self):   self.dmg_count   += 1; self.alert_until = time.time() + 1.8
    def reg_eject(self): self.eject_count += 1


# ──────────────────────────────────────────────────────────────────────────────
# DEMO HELPERS – unverändert
# ──────────────────────────────────────────────────────────────────────────────
def make_demo_frame(idx):
    frame = np.full((720,1280,3),(18,20,28),dtype=np.uint8)
    cv2.rectangle(frame,(0,460),(1280,580),(35,32,28),-1)
    for x in range(0,1280,40):
        s = 22+int(10*abs(((x+idx*3)%80)/40-1))
        cv2.rectangle(frame,(x,465),(x+38,578),(s,s-5,s-8),-1)
    cv2.line(frame,(0,462),(1280,462),(60,55,50),2)
    cv2.line(frame,(0,578),(1280,578),(60,55,50),2)
    for ox,dmg in [(0,False),(430,True),(860,False)]:
        px=int((ox+idx*4)%1420)-110; py=390
        col=(30,45,60) if dmg else (45,80,100)
        dk=tuple(max(0,c-15) for c in col)
        cv2.rectangle(frame,(px,py),(px+160,py+80),col,-1)
        cv2.rectangle(frame,(px,py),(px+160,py+80),dk,2)
        cv2.rectangle(frame,(px,py+30),(px+160,py+50),(50,70,100) if dmg else (80,110,60),-1)
        if dmg:
            pts=np.array([[px+20,py+10],[px+55,py+25],[px+40,py+40],[px+70,py+15]],np.int32)
            cv2.polylines(frame,[pts],False,(8,12,18),2)
        cv2.putText(frame,"PKT",(px+60,py+75),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1)
    return frame


def make_demo_detections(idx: int) -> List[dict]:
    detections = []
    for ox, dmg in [(0,False),(430,True),(860,False)]:
        px = int((ox+idx*4)%1420)-110; py=390
        x1,y1,x2,y2 = px,py,px+160,py+80
        if x2 < 0 or x1 > 1280: continue
        detections.append({
            "bbox":       (x1,y1,x2,y2),
            "is_damaged": dmg,
            "conf":       0.91 if dmg else 0.87,
            "cls_id":     Config.DAMAGED_CLASS_ID if dmg else Config.INTACT_CLASS_ID,
            "label":      "beschaedigt" if dmg else "intakt",
        })
    return detections


# ──────────────────────────────────────────────────────────────────────────────
# [GMP-5] DEFEKTRATE-MONITOR
# ──────────────────────────────────────────────────────────────────────────────
class DefectRateMonitor:
    """
    Rollendes Fenster für Defektrate.
    Gibt True zurück wenn Alarm ausgelöst werden soll (einmalig pro Überschreitung).
    """
    def __init__(self, window: int = 100, threshold: float = Config.GMP_ALARM_THRESHOLD):
        self._window    = window
        self._threshold = threshold
        self._history:  deque = deque(maxlen=window)
        self._alarmed   = False   # Verhindert Alarm-Spam

    def update(self, is_damaged: bool) -> bool:
        self._history.append(1 if is_damaged else 0)
        if len(self._history) < self._window:
            return False
        rate = sum(self._history) / self._window
        if rate > self._threshold and not self._alarmed:
            self._alarmed = True
            return True
        if rate <= self._threshold:
            self._alarmed = False
        return False

    @property
    def rate(self) -> float:
        if not self._history: return 0.0
        return sum(self._history) / len(self._history)


# ──────────────────────────────────────────────────────────────────────────────
# HAUPT-LOOP
# ──────────────────────────────────────────────────────────────────────────────
def run(args):
    # [BUG-3] Production Safety Check
    check_production_safety()

    # [GMP-8] GMP Config validieren (wirft RuntimeError in Prod mit Default-Key)
    GMPConfig.validate()

    # Model laden
    model_path = verify_model(args.model)
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    model.fuse()

    # [GMP-2] Benutzer anlegen
    try:
        user = GMPUser(
            user_id=args.user_id,
            display_name=args.user_name,
            role=args.role,
        )
    except SecurityException as e:
        log.error("[GMP] User-Fehler: %s", e)
        sys.exit(1)

    # [GMP-3] Batch-ID bestimmen (CLI > ENV > Auto)
    batch_id = (
        args.batch_id
        or os.getenv("GMP_BATCH_ID")
        or datetime.datetime.utcnow().strftime("AUTO-%Y%m%d-%H%M%S")
    )

    tracker      = ObjectTracker()
    decision     = DecisionEngine(tracker)
    actuator     = ActuatorLayer(
        mode="serial" if args.serial else ("gpio" if args.gpio else "demo"),
        shadow=args.shadow,
        serial_port=args.serial,
        gpio_pin=args.gpio_pin,
    )
    osd          = OSD(shadow=args.shadow)
    defect_mon   = DefectRateMonitor()

    demo_mode = args.demo
    cap = None
    if not demo_mode:
        try:
            src = validate_source(args.source)
        except Exception as e:
            log.error("%s", e); sys.exit(1)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error("Kann Quelle nicht öffnen: %s", src); sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # [GMP-1] GMP Audit-Trail starten (Context Manager schreibt LOGIN + LOGOUT)
    audit_db = Path(os.getenv("GMP_AUDIT_DB", "audit_gmp.db"))

    with GMPAuditTrail(audit_db, user, batch_id=batch_id) as audit:

        # [GMP-4] Batch-Start
        audit.log_event("BATCH_START", {
            "product_code":  os.getenv("GMP_PRODUCT_CODE", "PKT-UNKNOWN"),
            "source":        "demo" if demo_mode else str(args.source),
            "model":         str(model_path),
            "rezeptur":      GMPConfig.REZEPTUR_VERSION,
            "shadow_mode":   args.shadow,
        }, reason=f"Produktionsstart Batch {batch_id}")

        log.info("Gestartet | Batch: %s | User: %s (%s) | Shadow: %s | Env: %s",
                 batch_id, user.user_id, user.role, args.shadow, Config.ENVIRONMENT)
        log.info("[BUG-7] Kill-Switch: touch %s", Config.STOP_FILE)
        log.info("Drücke Q zum Beenden.")

        frame_idx    = 0
        t_prev       = time.perf_counter()
        skip_counter = 0

        try:
            while True:
                # [BUG-7] Emergency Stop prüfen
                if Config.STOP_FILE.exists():
                    time.sleep(0.05)
                    if not demo_mode and cap:
                        ret, _ = cap.read()
                        if not ret: break
                    continue

                # [BUG-5] Backpressure
                if skip_counter > 0:
                    skip_counter -= 1
                    if not demo_mode:
                        ret, _ = cap.read()
                        if not ret: break
                    continue

                frame_idx += 1

                if demo_mode:
                    frame = make_demo_frame(frame_idx)
                    time.sleep(1/30)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        if isinstance(src, str):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                        break

                # Inferenz
                if demo_mode:
                    detections = make_demo_detections(frame_idx)
                else:
                    try:
                        results = model(frame, conf=Config.CONF_THRESHOLD,
                                        iou=Config.IOU_THRESHOLD, verbose=False)[0]
                    except Exception as e:
                        log.error("[INFER] Fehler: %s", e); continue
                    detections = []
                    for box in results.boxes:
                        try:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            conf   = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            detections.append({
                                "bbox":       (x1,y1,x2,y2),
                                "is_damaged": cls_id == Config.DAMAGED_CLASS_ID,
                                "conf":       conf,
                                "cls_id":     cls_id,
                                "label":      model.names[cls_id],
                            })
                        except Exception as e:
                            log.error("Box-Fehler: %s", e)

                # [BUG-1] Tracker
                detections = tracker.update(detections)

                for det in detections:
                    x1,y1,x2,y2 = det["bbox"]
                    tid          = det["track_id"]
                    conf         = det["conf"]
                    is_damaged   = det["is_damaged"]
                    should_eject = decision.evaluate(tid, conf)
                    col    = COL_DAMAGE if is_damaged else COL_OK
                    status = "BESCHÄDIGT" if is_damaged else "INTAKT"

                    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                    tag = f"#{tid} {status} {conf:.0%}"
                    (tw,th),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
                    cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+6,y1),col,-1)
                    cv2.putText(frame,tag,(x1+3,y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

                    track = tracker.get_track(tid)
                    if track and is_damaged:
                        streak_pct = track.dmg_streak / Config.FRAMES_CONFIRM
                        bw = x2-x1
                        cv2.rectangle(frame,(x1,y2+2),(x2,y2+8),(40,40,60),-1)
                        cv2.rectangle(frame,(x1,y2+2),
                                      (x1+int(bw*min(streak_pct,1.0)),y2+8),COL_DAMAGE,-1)

                    if is_damaged: osd.reg_dmg()
                    else:          osd.reg_ok()

                    ejected = False
                    if should_eject:
                        ejected = actuator.trigger()
                        if ejected: osd.reg_eject()

                    # [GMP-1] GMP Audit statt AuditLogger.log()
                    audit.log(status, conf, (x1,y1,x2-x1,y2-y1), ejected, tid, frame_idx)

                    # [GMP-5] Defektrate überwachen
                    if defect_mon.update(is_damaged):
                        audit.log_event("ALARM", {
                            "type":        "HIGH_DEFECT_RATE",
                            "defect_rate": round(defect_mon.rate, 4),
                            "threshold":   Config.GMP_ALARM_THRESHOLD,
                            "frame_idx":   frame_idx,
                            "batch_id":    batch_id,
                        }, reason=f"GMP-Alarm: Defektrate {defect_mon.rate:.1%} > Limit {Config.GMP_ALARM_THRESHOLD:.1%}")
                        log.warning("[GMP-5] ALARM: Defektrate %.1f%%", defect_mon.rate * 100)

                # FPS + Backpressure
                t_now = time.perf_counter()
                fps = 1.0 / max(t_now - t_prev, 1e-9)
                t_prev = t_now
                if fps < Config.TARGET_FPS * 0.6:
                    skip_counter = min(skip_counter + 1, Config.MAX_FRAME_SKIP)
                elif fps > Config.TARGET_FPS * 0.9 and skip_counter > 0:
                    skip_counter -= 1

                frame = osd.draw(frame, fps)
                cv2.imshow("PacketGuard AI · GMP Build", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            # [GMP-4] [GMP-9] Batch-Ende immer schreiben (auch bei Exception)
            audit.log_event("BATCH_END", {
                "total_inspected": osd.ok_count + osd.dmg_count,
                "total_ok":        osd.ok_count,
                "total_damaged":   osd.dmg_count,
                "total_ejected":   osd.eject_count,
                "final_defect_rate": round(defect_mon.rate, 4),
            }, reason=f"Produktionsende Batch {batch_id}")

            if cap: cap.release()
            cv2.destroyAllWindows()
            actuator.close()
            log.info("Beendet | OK:%d DMG:%d EJECT:%d | Batch:%s",
                     osd.ok_count, osd.dmg_count, osd.eject_count, batch_id)
            log.info("[GMP] Audit-DB: %s", audit_db)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="PacketGuard AI · GMP Build (FDA 21 CFR Part 11)"
    )

    # Video / Kamera
    p.add_argument("--source",    default="0",
                   help="Kamera-Index oder Video-Pfad (default: 0)")
    p.add_argument("--model",     default="models/yolov8n.pt")
    p.add_argument("--demo",      action="store_true",
                   help="Demo-Modus ohne echte Kamera")

    # Actuator
    p.add_argument("--serial",    default=None,  help="Serial-Port z.B. /dev/ttyUSB0")
    p.add_argument("--gpio",      action="store_true")
    p.add_argument("--gpio-pin",  type=int, default=17)
    p.add_argument("--shadow",    action="store_true",
                   help="Shadow-Mode: kein physischer Auswurf")

    # [GMP-2] Benutzer
    p.add_argument("--user-id",   default=os.getenv("GMP_USER_ID",   "op01"),
                   help="Benutzer-ID (default: op01 / ENV: GMP_USER_ID)")
    p.add_argument("--user-name", default=os.getenv("GMP_USER_NAME", "Operator"),
                   help="Anzeigename (default: Operator)")
    p.add_argument("--role",      default=os.getenv("GMP_ROLE",      "operator"),
                   choices=["operator","qa","supervisor","admin"],
                   help="GMP-Rolle (default: operator)")

    # [GMP-3] Batch
    p.add_argument("--batch-id",  default=None,
                   help="Batch-ID (default: AUTO-<timestamp> / ENV: GMP_BATCH_ID)")

    # [GMP-6] Audit-Operationen
    p.add_argument("--verify-audit", metavar="DB",
                   help="Hash-Chain einer Audit-DB prüfen und beenden")
    p.add_argument("--export-audit", metavar="DB",
                   help="Audit-DB exportieren")
    p.add_argument("--batch",    metavar="BATCH_ID",
                   help="Filter auf Batch-ID (für --verify-audit / --export-audit)")
    p.add_argument("--format",   choices=["csv","json"], default="csv")
    p.add_argument("--out",      metavar="FILE",
                   help="Ausgabedatei für --export-audit")

    args = p.parse_args()

    # [GMP-6] Verify-Modus
    if args.verify_audit:
        ok = GMPAuditTrail.verify(Path(args.verify_audit), batch_id=args.batch)
        sys.exit(0 if ok else 1)

    # [GMP-7] Export-Modus
    if args.export_audit:
        if not args.out:
            p.error("--export-audit benötigt --out <datei>")
        import sqlite3
        tmp = GMPAuditTrail.__new__(GMPAuditTrail)
        tmp.db_path  = Path(args.export_audit)
        tmp._connect = lambda: sqlite3.connect(str(tmp.db_path))
        tmp.export(Path(args.out), batch_id=args.batch, fmt=args.format)
        sys.exit(0)

    run(args)
