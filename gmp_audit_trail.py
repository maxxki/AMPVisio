#!/usr/bin/env python3
"""
gmp_audit_trail.py – FDA 21 CFR Part 11 konformer Audit-Trail
==============================================================
Ersetzt den CSV-Hash-Chain aus paket_scanner.py durch eine
GMP-konforme SQLite-Lösung mit:

  - Append-Only (SQL-Trigger blockieren DELETE/UPDATE)
  - UTC-Zeitstempel (Zeitzonen-sicher)
  - HMAC-SHA256-Signatur pro Eintrag (HSM-fähig via pkcs11-Shim)
  - Vorwärts-verkettete Signatur-Chain (ALCOA+)
  - Benutzer-ID + Rolle in jedem Eintrag
  - Batch-Verknüpfung
  - Vollständige Chain-Verifikation + FDA-Export (CSV/JSON)

Integration in paket_scanner.py:
  from gmp_audit_trail import GMPAuditTrail, GMPUser, GMPConfig
  # AuditLogger ersetzen durch GMPAuditTrail(db_path, user)

Verwendung standalone:
  python gmp_audit_trail.py --demo
  python gmp_audit_trail.py --verify audit.db
  python gmp_audit_trail.py --export audit.db --batch 20260530-0001
"""

import argparse
import csv
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import sys
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

log = logging.getLogger("GMPAudit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
class GMPConfig:
    """
    Alle sicherheitsrelevanten Parameter.
    In Produktion: nur via Change-Control änderbar (SOP-gesteuert).
    """
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Audit-Datenbank
    AUDIT_DB:    Path = Path(os.getenv("GMP_AUDIT_DB", "audit_gmp.db"))

    # Signatur – HMAC-Key aus Umgebung oder HSM
    # Produktion: HSM_KEY_ID statt Klartext-Key
    _HMAC_KEY: bytes = os.getenv("GMP_HMAC_KEY", "CHANGE-ME-IN-PRODUCTION").encode()
    HSM_ENABLED: bool = os.getenv("GMP_HSM", "0") == "1"
    HSM_KEY_ID:  str  = os.getenv("GMP_HSM_KEY_ID", "pkcs11:token=packetguard;object=audit_sign")

    # Rezeptur-Version – Change-Control-genehmigt
    REZEPTUR_VERSION: str = os.getenv("GMP_REZEPTUR_VERSION", "v1.0.0")

    # Rate Limit (identisch zu bisherigem LOG_MAX_PER_SEC)
    LOG_MAX_PER_SEC: int = int(os.getenv("LOG_MAX_PER_SEC", "60"))

    @classmethod
    def is_production(cls) -> bool:
        return cls.ENVIRONMENT.lower() in ("production", "prod")

    @classmethod
    def validate(cls):
        """Startup-Prüfung: In Produktion darf kein Default-Key aktiv sein."""
        if cls.is_production():
            if cls._HMAC_KEY == b"CHANGE-ME-IN-PRODUCTION" and not cls.HSM_ENABLED:
                raise RuntimeError(
                    "[GMP] FATAL: Default-HMAC-Key in ENVIRONMENT=production.\n"
                    "  → GMP_HMAC_KEY setzen ODER GMP_HSM=1 aktivieren."
                )
            log.info("[GMP] Produktionsmodus – alle Guards aktiv.")
        else:
            log.warning("[GMP] Development-Modus – nicht für Produktion verwenden.")


# ──────────────────────────────────────────────────────────────────────────────
# BENUTZER (minimal – vollständiges Auth-Modul ist nächster Sprint)
# ──────────────────────────────────────────────────────────────────────────────
class SecurityException(Exception):
    """GMP-Security-Verletzung – führt zu sofortigem Audit-Eintrag + Stop."""


@dataclass(frozen=True)
class GMPUser:
    """
    Unveränderlicher Benutzer-Record für eine Session.
    frozen=True: kein nachträgliches Manipulieren möglich.
    """
    user_id:     str   # z.B. "op01" oder LDAP-CN
    display_name: str
    role:        Literal["operator", "qa", "supervisor", "admin"]
    session_id:  str   = ""   # UUID, bei __post_init__ gesetzt
    login_time:  str   = ""   # ISO-UTC, bei __post_init__ gesetzt

    def __post_init__(self):
        # frozen=True erlaubt kein self.x = … → object.__setattr__
        if not self.session_id:
            object.__setattr__(self, "session_id", str(uuid.uuid4()))
        if not self.login_time:
            object.__setattr__(self, "login_time",
                               datetime.now(timezone.utc).isoformat())
        # Produktion: keine Test-Accounts
        if self.user_id.startswith("test_") and GMPConfig.is_production():
            raise SecurityException("Test-Accounts in Produktion verboten.")

    @classmethod
    def system(cls) -> "GMPUser":
        """System-Account für automatisierte Einträge (Startup, Kalibrierung)."""
        return cls(user_id="SYSTEM", display_name="PacketGuard System",
                   role="admin")


# ──────────────────────────────────────────────────────────────────────────────
# SIGNATUR-BACKEND (HMAC lokal oder HSM via PKCS#11-Shim)
# ──────────────────────────────────────────────────────────────────────────────
class SignatureBackend:
    """
    Austauschbares Signatur-Backend.
    Development:  HMAC-SHA256 mit konfigurierbarem Key.
    Production:   ECDSA P-256 via PKCS#11 HSM (Stub – pkcs11-Bibliothek nötig).
    """

    def sign(self, data: bytes) -> str:
        if GMPConfig.HSM_ENABLED:
            return self._sign_hsm(data)
        return self._sign_hmac(data)

    @staticmethod
    def _sign_hmac(data: bytes) -> str:
        return hmac.new(GMPConfig._HMAC_KEY, data, hashlib.sha256).hexdigest()

    @staticmethod
    def _sign_hsm(data: bytes) -> str:
        """
        Produktions-Stub: ECDSA P-256 über PKCS#11.
        Aktivieren mit:  pip install python-pkcs11
        Dann ersetzen durch:
            import pkcs11
            lib = pkcs11.lib(os.getenv("PKCS11_LIB"))
            token = lib.get_token(token_label="packetguard")
            with token.open(user_pin=os.getenv("HSM_PIN")) as session:
                key = session.get_key(label="audit_sign_key")
                return key.sign(data, mechanism=pkcs11.Mechanism.ECDSA_SHA256).hex()
        """
        raise NotImplementedError(
            "[GMP] HSM-Signatur nicht implementiert.\n"
            "  → python-pkcs11 installieren und _sign_hsm() vervollständigen."
        )

    def verify(self, data: bytes, signature: str) -> bool:
        if GMPConfig.HSM_ENABLED:
            # HSM-Verifikation: analog zu sign()
            raise NotImplementedError("HSM-Verifikation: siehe _sign_hsm()")
        expected = self._sign_hmac(data)
        return hmac.compare_digest(expected, signature)


# ──────────────────────────────────────────────────────────────────────────────
# GMP AUDIT-TRAIL
# ──────────────────────────────────────────────────────────────────────────────
class GMPAuditTrail:
    """
    FDA 21 CFR Part 11 konformer Audit-Trail.

    ALCOA+-Prinzipien:
      Attributable  – user_id + role in jedem Eintrag
      Legible        – JSON event_data, lesbare Zeitstempel
      Contemporaneous– Zeitstempel direkt beim Schreiben (UTC)
      Original       – Append-Only, kein DELETE/UPDATE
      Accurate       – SHA-256 + HMAC pro Eintrag
      Complete       – chain_index lückenlos
      Consistent     – prev_sig verknüpft jeden Eintrag mit Vorgänger
      Enduring       – SQLite WAL, FULL-Sync, kein Datenverlust
      Available      – verify() + export() jederzeit aufrufbar

    Drop-In-Ersatz für paket_scanner.AuditLogger:
      audit = GMPAuditTrail(GMPConfig.AUDIT_DB, user)
      audit.log("INTAKT", 0.91, (x,y,w,h), ejected=False, track_id=1, frame_idx=42)
    """

    # Gültige Event-Typen (SQL CHECK-Constraint spiegelt diese Liste)
    EVENT_TYPES = frozenset({
        "INSPECTION", "EJECT", "CALIBRATION", "ALARM",
        "LOGIN", "LOGOUT", "CONFIG_CHANGE", "BATCH_START", "BATCH_END",
    })

    def __init__(self, db_path: Path, user: GMPUser,
                 batch_id: str = "UNSET",
                 max_per_sec: int = GMPConfig.LOG_MAX_PER_SEC):
        GMPConfig.validate()
        self.db_path     = Path(db_path)
        self.user        = user
        self.batch_id    = batch_id
        self.max_per_sec = max_per_sec

        self._lock       = threading.RLock()
        self._signer     = SignatureBackend()
        self._prev_sig   = "0" * 64   # Genesis-Signatur
        self._chain_idx  = 0
        self._rate_count = 0
        self._rate_ts    = 0

        self._init_db()
        self._restore_chain_state()

        # Startup-Eintrag
        self._write("LOGIN", {
            "session_id":  user.session_id,
            "role":        user.role,
            "environment": GMPConfig.ENVIRONMENT,
        }, reason="Session-Start")
        log.info("[GMP] Audit-Trail initialisiert: %s | User: %s | Batch: %s",
                 self.db_path, user.user_id, batch_id)

    # ── Datenbank-Schema ───────────────────────────────────────────────────

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")  # Keine Datenverluste
            conn.execute("PRAGMA foreign_keys=ON")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    chain_index      INTEGER NOT NULL UNIQUE,
                    timestamp_utc    TEXT    NOT NULL,
                    batch_id         TEXT    NOT NULL,
                    rezeptur_version TEXT    NOT NULL,
                    user_id          TEXT    NOT NULL,
                    user_role        TEXT    NOT NULL,
                    session_id       TEXT    NOT NULL,
                    event_type       TEXT    NOT NULL,
                    event_data       TEXT    NOT NULL,
                    reason           TEXT    NOT NULL,
                    record_hash      TEXT    NOT NULL,
                    prev_sig         TEXT    NOT NULL,
                    signature        TEXT    NOT NULL,
                    CHECK (event_type IN (
                        'INSPECTION','EJECT','CALIBRATION','ALARM',
                        'LOGIN','LOGOUT','CONFIG_CHANGE',
                        'BATCH_START','BATCH_END'
                    ))
                )
            """)

            # Append-Only: DELETE und UPDATE verboten
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_no_delete
                BEFORE DELETE ON audit_log
                BEGIN
                    SELECT RAISE(FAIL,
                        'GMP VIOLATION: audit_log ist append-only – kein DELETE erlaubt');
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_no_update
                BEFORE UPDATE ON audit_log
                BEGIN
                    SELECT RAISE(FAIL,
                        'GMP VIOLATION: audit_log ist unveränderlich – kein UPDATE erlaubt');
                END
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_batch     ON audit_log(batch_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp_utc)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user      ON audit_log(user_id)")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    # ── Chain-State wiederherstellen (nach Neustart) ───────────────────────

    def _restore_chain_state(self):
        """Lädt letzten chain_index + signature beim Neustart."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chain_index, signature FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            self._chain_idx = row[0] + 1
            self._prev_sig  = row[1]
            log.info("[GMP] Chain fortgesetzt ab Index %d", self._chain_idx)

    # ── Öffentliches Interface ─────────────────────────────────────────────

    def log(self, status: str, conf: float, bbox: tuple,
            ejected: bool, track_id: int, frame_idx: int) -> bool:
        """
        Drop-In-Ersatz für paket_scanner.AuditLogger.log().
        Gleiche Signatur – intern GMP-konform.
        """
        x, y, w, h = bbox
        is_damaged = status == "BESCHÄDIGT"
        event_type = "EJECT" if ejected else "INSPECTION"

        return self._write(event_type, {
            "status":    status,
            "conf":      round(conf, 4),
            "bbox":      {"x": x, "y": y, "w": w, "h": h},
            "ejected":   ejected,
            "track_id":  track_id,
            "frame_idx": frame_idx,
        }, reason="Routine-Inspektion")

    def log_event(self, event_type: str, data: dict, reason: str) -> bool:
        """Allgemeines GMP-Event (ALARM, CONFIG_CHANGE, CALIBRATION, …)."""
        if event_type not in self.EVENT_TYPES:
            raise ValueError(f"Unbekannter Event-Typ: {event_type}. "
                             f"Erlaubt: {self.EVENT_TYPES}")
        if not reason:
            raise ValueError("[GMP] 'reason' darf nicht leer sein.")
        return self._write(event_type, data, reason=reason)

    # ── Kern-Schreiblogik ──────────────────────────────────────────────────

    def _write(self, event_type: str, data: dict, reason: str) -> bool:
        """Thread-sicheres, rate-limitiertes Schreiben eines GMP-Eintrags."""
        with self._lock:
            # Rate-Limit (identisch zu bisherigem AuditLogger)
            import time
            now_sec = int(time.time())
            if now_sec != self._rate_ts:
                self._rate_ts    = now_sec
                self._rate_count = 0
            if self._rate_count >= self.max_per_sec:
                return False
            self._rate_count += 1

            ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")

            # Record für Hashing serialisieren (deterministisch)
            record = {
                "chain_index":      self._chain_idx,
                "timestamp_utc":    ts,
                "batch_id":         self.batch_id,
                "rezeptur_version": GMPConfig.REZEPTUR_VERSION,
                "user_id":          self.user.user_id,
                "user_role":        self.user.role,
                "session_id":       self.user.session_id,
                "event_type":       event_type,
                "event_data":       json.dumps(data, sort_keys=True),
                "reason":           reason,
            }
            record_bytes = json.dumps(record, sort_keys=True).encode()
            record_hash  = hashlib.sha256(record_bytes).hexdigest()

            # Signatur über (record_hash || prev_sig) – bindet an Chain
            sig_input = (record_hash + self._prev_sig).encode()
            signature = self._signer.sign(sig_input)

            try:
                with self._connect() as conn:
                    conn.execute("""
                        INSERT INTO audit_log (
                            chain_index, timestamp_utc, batch_id, rezeptur_version,
                            user_id, user_role, session_id,
                            event_type, event_data, reason,
                            record_hash, prev_sig, signature
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        self._chain_idx, ts, self.batch_id,
                        GMPConfig.REZEPTUR_VERSION,
                        self.user.user_id, self.user.role, self.user.session_id,
                        event_type, record["event_data"], reason,
                        record_hash, self._prev_sig, signature,
                    ))
                    conn.commit()

                self._prev_sig  = signature
                self._chain_idx += 1
                return True

            except sqlite3.Error as e:
                log.error("[GMP] Schreibfehler: %s", e)
                return False

    # ── Verifikation ───────────────────────────────────────────────────────

    @classmethod
    def verify(cls, db_path: Path,
               batch_id: Optional[str] = None) -> bool:
        """
        Vollständige Chain-Verifikation.
        Prüft: record_hash, Signatur, prev_sig-Verkettung, chain_index-Lücken.

        Wird vor jeder Batch-QA-Freigabe aufgerufen.
        Gibt True zurück wenn die Chain integer ist.
        """
        signer  = SignatureBackend()
        errors  = 0
        prev_sig = "0" * 64
        expected_idx = 0

        log.info("[GMP] Starte Chain-Verifikation: %s", db_path)

        with sqlite3.connect(str(db_path)) as conn:
            query = "SELECT * FROM audit_log ORDER BY chain_index"
            params = ()
            if batch_id:
                query = ("SELECT * FROM audit_log WHERE batch_id=? "
                         "ORDER BY chain_index")
                params = (batch_id,)
            rows = conn.execute(query, params).fetchall()
            cols = [d[0] for d in conn.execute(query, params).description
                    ] if rows else []

        if not rows:
            log.warning("[GMP] Keine Einträge gefunden.")
            return True

        # Spalten-Map aufbauen
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute(
                "SELECT * FROM audit_log ORDER BY chain_index LIMIT 1"
            )
            col_names = [d[0] for d in cur.description]

        col = {name: i for i, name in enumerate(col_names)}

        for row in rows:
            idx  = row[col["chain_index"]]
            rh   = row[col["record_hash"]]
            ps   = row[col["prev_sig"]]
            sig  = row[col["signature"]]

            # 1. Lückenloser chain_index
            if idx != expected_idx and not batch_id:
                log.error("[GMP] Chain-Lücke: erwartet %d, gefunden %d",
                          expected_idx, idx)
                errors += 1
            expected_idx = idx + 1

            # 2. prev_sig-Verkettung
            if ps != prev_sig:
                log.error("[GMP] Chain-Bruch bei Index %d: "
                          "prev_sig stimmt nicht überein", idx)
                errors += 1

            # 3. Record-Hash rekonstruieren
            record = {
                "chain_index":      idx,
                "timestamp_utc":    row[col["timestamp_utc"]],
                "batch_id":         row[col["batch_id"]],
                "rezeptur_version": row[col["rezeptur_version"]],
                "user_id":          row[col["user_id"]],
                "user_role":        row[col["user_role"]],
                "session_id":       row[col["session_id"]],
                "event_type":       row[col["event_type"]],
                "event_data":       row[col["event_data"]],
                "reason":           row[col["reason"]],
            }
            calc_hash = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()

            if calc_hash != rh:
                log.error("[GMP] Hash-Mismatch in Zeile %d!", idx)
                errors += 1

            # 4. Signatur prüfen
            sig_input = (rh + ps).encode()
            if not signer.verify(sig_input, sig):
                log.error("[GMP] Signatur ungültig in Zeile %d!", idx)
                errors += 1

            prev_sig = sig

        if errors == 0:
            log.info("[GMP] ✓ Chain integer – %d Einträge geprüft.", len(rows))
            return True
        else:
            log.error("[GMP] ✗ %d Fehler gefunden – Audit-Trail kompromittiert!",
                      errors)
            return False

    # ── Export ─────────────────────────────────────────────────────────────

    def export(self, output_path: Path,
               batch_id: Optional[str] = None,
               fmt: Literal["csv", "json"] = "csv"):
        """
        Exportiert Audit-Trail für FDA-Einreichung / QA-Review.
        ALCOA+-konform: alle Felder, unveränderter Inhalt.
        """
        with self._connect() as conn:
            query  = "SELECT * FROM audit_log ORDER BY chain_index"
            params = ()
            if batch_id:
                query  = ("SELECT * FROM audit_log WHERE batch_id=? "
                          "ORDER BY chain_index")
                params = (batch_id,)
            cur  = conn.execute(query, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()

        if fmt == "csv":
            with open(output_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)
        else:
            records = [dict(zip(cols, r)) for r in rows]
            with open(output_path, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        log.info("[GMP] Export: %d Einträge → %s", len(rows), output_path)

    # ── Kontext-Manager (für with-Statement) ──────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._write("LOGOUT", {
            "session_id": self.user.session_id,
            "exception":  str(exc_type) if exc_type else None,
        }, reason="Session-Ende")


# ──────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ──────────────────────────────────────────────────────────────────────────────
def _demo():
    """Demonstriert den GMP Audit-Trail mit synthetischen Daten."""
    import random, time as _time

    db = Path("audit_demo.db")
    if db.exists():
        db.unlink()

    operator = GMPUser(user_id="op01", display_name="Max Muster",
                       role="operator")
    qa       = GMPUser(user_id="qa01", display_name="Anna QA",
                       role="qa")

    print("\n── Demo: GMP Audit-Trail ──────────────────────────────\n")

    with GMPAuditTrail(db, operator, batch_id="20260530-0001") as audit:

        # Batch-Start
        audit.log_event("BATCH_START", {
            "product_code": "AMP-INS-001",
            "target_qty":   5000,
        }, reason="Produktionsstart Batch 20260530-0001")

        # Simulated inspections
        rng = random.Random(42)
        ejected = 0
        for i in range(20):
            is_dmg = rng.random() < 0.30
            conf   = rng.uniform(0.80, 0.99) if is_dmg else rng.uniform(0.85, 0.98)
            status = "BESCHÄDIGT" if is_dmg else "INTAKT"
            eject  = is_dmg and conf > 0.88
            if eject:
                ejected += 1
            audit.log(status, conf, (100, 350, 160, 80), eject, i, i)

        # Alarm bei hoher Defektrate
        audit.log_event("ALARM", {
            "type":        "HIGH_DEFECT_RATE",
            "defect_rate": 0.30,
            "threshold":   0.05,
        }, reason="GMP-Alarm: Defekt-Rate 30% > Limit 5%")

        # Batch-Ende
        audit.log_event("BATCH_END", {
            "total_inspected": 20,
            "total_ejected":   ejected,
        }, reason="Produktionsende – übergabe an QA-Prüfung")

    print(f"\n✓ {20} Inspektionen, {ejected} Auswürfe geschrieben → {db}\n")

    # QA: Chain verifizieren
    print("── QA: Chain-Verifikation ─────────────────────────────\n")
    ok = GMPAuditTrail.verify(db)

    # Export
    csv_out = Path("audit_demo_export.csv")
    tmp_audit = GMPAuditTrail.__new__(GMPAuditTrail)
    tmp_audit.db_path = db
    tmp_audit._connect = lambda: sqlite3.connect(str(db))
    tmp_audit.export(csv_out, fmt="csv")
    print(f"\n── Export → {csv_out} ({'OK' if ok else 'FEHLER'})\n")

    # Append-Only-Schutz demonstrieren
    print("── Append-Only-Test: DELETE wird blockiert ────────────\n")
    with sqlite3.connect(str(db)) as conn:
        try:
            conn.execute("DELETE FROM audit_log WHERE id=1")
            print("⚠  DELETE nicht blockiert – Trigger fehlt!")
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            print(f"✓  DELETE korrekt blockiert: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMP Audit-Trail – FDA 21 CFR Part 11"
    )
    parser.add_argument("--demo",   action="store_true",
                        help="Demo mit synthetischen Daten ausführen")
    parser.add_argument("--verify", metavar="DB",
                        help="Hash-Chain einer Datenbank prüfen")
    parser.add_argument("--export", metavar="DB",
                        help="Datenbank exportieren")
    parser.add_argument("--batch",  metavar="BATCH_ID",
                        help="Filter auf Batch-ID (für --verify und --export)")
    parser.add_argument("--format", choices=["csv", "json"], default="csv",
                        help="Export-Format (default: csv)")
    parser.add_argument("--out",    metavar="FILE",
                        help="Ausgabedatei für --export")
    args = parser.parse_args()

    if args.demo:
        _demo()

    elif args.verify:
        ok = GMPAuditTrail.verify(Path(args.verify), batch_id=args.batch)
        sys.exit(0 if ok else 1)

    elif args.export:
        if not args.out:
            parser.error("--export benötigt --out <datei>")
        # minimaler Kontext für Export
        tmp = GMPAuditTrail.__new__(GMPAuditTrail)
        tmp.db_path  = Path(args.export)
        tmp._connect = lambda: sqlite3.connect(str(tmp.db_path))
        tmp.export(Path(args.out), batch_id=args.batch,
                   fmt=args.format)

    else:
        parser.print_help()
