#!/usr/bin/env python3
"""
Generate raw PYTHIA events and store in sharded format for cached analysis.
Output: pythia_finalstate_raw/<LABEL>/shard_XXXXXX/

Run from project root: python scripts/generation/generate_events_raw.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

import pythia8

# -----------------------------
# Configuration
# -----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import count_files_under, pythia_finalstate_raw_dir, write_run_manifest  # noqa: E402

OUTDIR = pythia_finalstate_raw_dir()
OUTDIR.mkdir(parents=True, exist_ok=True)

N_EVENTS_PER_CONFIG = 1_000_000  # Set to 20_000 for quick test; comment out ISRFSR_* in main() to generate only ETA_ON_CRON
EVENTS_PER_SHARD = 10_000

E_E = 18.0
E_P = 275.0
Q2_MIN = 16.0

BASE_SEED = 12345

DEBUG = False
DEBUG_ATTEMPTS = 10_000

DT_FLOAT = np.float32
DT_INT = np.int32
DT_OFF = np.int64


# -----------------------------
# Helpers
# -----------------------------
def p4_from_particle(p):
    return np.array([p.e(), p.px(), p.py(), p.pz()], dtype=DT_FLOAT)


def get_scattered_electron(ev):
    """Prefer status 44; else highest-energy e- with status>0."""
    electrons = []
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() > 0:
            electrons.append(p)
    if not electrons:
        return None
    for e in electrons:
        if e.status() == 44:
            return e
    return max(electrons, key=lambda x: x.e())


def find_incoming_beams(ev):
    """Find incoming electron and proton by id and status (independent of beam order)."""
    e_in = None
    p_in = None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:
            e_in = p4_from_particle(p)
        if p.id() == 2212 and p.status() < 0:
            p_in = p4_from_particle(p)
    return e_in, p_in


def find_k_out(ev):
    """Find outgoing struck quark. Accept ±23, ±63..±68, or any quark with status!=0."""
    quark_ids = (1, 2, 3, 4, 5, 6)
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and abs(p.status()) == 23:
            return p4_from_particle(p)
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and (63 <= abs(p.status()) <= 68):
            return p4_from_particle(p)
    best = None
    bestE = -1.0
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and p.status() != 0:
            if p.e() > bestE:
                bestE = p.e()
                best = p
    if best is None:
        return None
    return p4_from_particle(best)


def shard_dir(label: str, shard_idx: int) -> Path:
    return OUTDIR / label / f"shard_{shard_idx:06d}"


def shard_complete(label: str, shard_idx: int) -> bool:
    d = shard_dir(label, shard_idx)
    return (d / "meta.json").exists() and (d / "offsets.npy").exists() and (d / "pid.npy").exists() and (d / "p4.npy").exists()


def next_shard_to_write(label: str) -> int:
    idx = 0
    while shard_complete(label, idx):
        idx += 1
    return idx


def already_completed_events(label: str, next_shard_idx: int) -> int:
    """Sum events_in_shard from meta.json for completed shards (handles partial last shard)."""
    total = 0
    for i in range(next_shard_idx):
        mp = shard_dir(label, i) / "meta.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            meta = json.load(f)
        total += int(meta.get("events_in_shard", 0))
    return total


def write_shard(
    label: str,
    shard_idx: int,
    e_in_arr,
    p_in_arr,
    e_sc_arr,
    k_out_arr,
    offsets,
    pid,
    p4,
    meta: dict,
):
    d = shard_dir(label, shard_idx)
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / "event_e_in.npy", e_in_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "event_p_in.npy", p_in_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "event_e_sc.npy", e_sc_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "event_k_out.npy", k_out_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "offsets.npy", offsets.astype(DT_OFF, copy=False))
    np.save(d / "pid.npy", pid.astype(DT_INT, copy=False))
    np.save(d / "p4.npy", p4.astype(DT_FLOAT, copy=False))

    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _electron_proton_GeV(cfg: dict) -> Tuple[float, float]:
    """Beam energies (electron, proton) from PYTHIA cfg."""
    if cfg["idA"] == 2212 and cfg["idB"] == 11:
        return float(cfg["eB"]), float(cfg["eA"])
    if cfg["idA"] == 11 and cfg["idB"] == 2212:
        return float(cfg["eA"]), float(cfg["eB"])
    raise ValueError(f"Unhandled beams idA={cfg['idA']} idB={cfg['idB']}")


def build_pythia(
    idA: int,
    idB: int,
    eA: float,
    eB: float,
    colour_reconnect_on: bool,
    isr_fsr_on: bool,
    seed: int,
    phase_space_q2_min: float | None = None,
):
    """Build PYTHIA with configurable beams and options."""
    p = pythia8.Pythia()
    q2min = float(Q2_MIN if phase_space_q2_min is None else phase_space_q2_min)

    p.readString(f"Beams:idA = {idA}")
    p.readString(f"Beams:idB = {idB}")
    p.readString(f"Beams:eA = {eA}")
    p.readString(f"Beams:eB = {eB}")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("HardQCD:all = off")
    p.readString("PDF:lepton = off")
    p.readString(f"PhaseSpace:Q2Min = {q2min}")

    p.readString("HadronLevel:all = on")
    p.readString(f"ColourReconnection:reconnect = {'on' if colour_reconnect_on else 'off'}")

    if isr_fsr_on:
        p.readString("PartonLevel:ISR = on")
        p.readString("PartonLevel:FSR = on")
    else:
        p.readString("PartonLevel:ISR = off")
        p.readString("PartonLevel:FSR = off")

    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {seed}")

    p.init()
    return p


def _config_fingerprint(cfg: dict) -> dict:
    """Serializable fingerprint for a label's PYTHIA config."""
    Ee, Ep = _electron_proton_GeV(cfg)
    q2m = float(cfg.get("q2_min", Q2_MIN))
    return {
        "idA": int(cfg["idA"]),
        "idB": int(cfg["idB"]),
        "eA": float(cfg["eA"]),
        "eB": float(cfg["eB"]),
        "isr_fsr_on": bool(cfg["isr_fsr_on"]),
        "colour_reconnect_on": bool(cfg["colour_reconnect_on"]),
        "seed_offset": int(cfg["seed_offset"]),
        "Q2_MIN": q2m,
        "E_E": Ee,
        "E_P": Ep,
    }


def _check_or_write_fingerprint(label: str, cfg: dict) -> None:
    """If config_fingerprint.json exists, assert it matches cfg; else write it."""
    label_dir = OUTDIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    fp_path = label_dir / "config_fingerprint.json"
    fingerprint = _config_fingerprint(cfg)
    if fp_path.exists():
        with open(fp_path) as f:
            existing = json.load(f)
        if existing != fingerprint:
            raise RuntimeError(
                f"[{label}] config_fingerprint.json does not match current settings. "
                f"Existing: {existing}. Current: {fingerprint}. "
                "Remove or update the fingerprint to avoid mixing incompatible shards."
            )
    else:
        with open(fp_path, "w") as f:
            json.dump(fingerprint, f, indent=2, sort_keys=True)
        print(f"[{label}] wrote config_fingerprint.json")


def generate_config(
    label: str,
    cfg: dict,
    n_events: int,
    events_per_shard: int,
    config_meta: dict = None,
):
    _check_or_write_fingerprint(label, cfg)
    q2_phase = cfg.get("q2_min", Q2_MIN)
    pythia_builder = lambda: build_pythia(
        idA=cfg["idA"],
        idB=cfg["idB"],
        eA=cfg["eA"],
        eB=cfg["eB"],
        colour_reconnect_on=cfg["colour_reconnect_on"],
        isr_fsr_on=cfg["isr_fsr_on"],
        seed=BASE_SEED + cfg["seed_offset"],
        phase_space_q2_min=q2_phase,
    )
    shard_idx = next_shard_to_write(label)
    already_done = already_completed_events(label, shard_idx)
    if already_done >= n_events:
        print(f"[{label}] Already complete ({already_done} >= {n_events}).")
        return

    pythia = pythia_builder()

    if already_done > 0:
        print(f"[{label}] Resuming: skipping {already_done} events to reach shard {shard_idx}.")
        skipped = 0
        while skipped < already_done:
            if not pythia.next():
                continue
            skipped += 1

    produced_total = already_done
    # One bar per label: advances once per accepted/written event (matches saved events).
    with tqdm(
        total=n_events,
        initial=produced_total,
        desc=f"{label}",
        unit="evt",
        dynamic_ncols=True,
        miniters=1,
        smoothing=0.05,
    ) as pbar:
        while produced_total < n_events:
            Ne = min(events_per_shard, n_events - produced_total)

            e_in_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)
            p_in_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)
            e_sc_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)
            k_out_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)

            offsets = np.zeros(Ne + 1, dtype=DT_OFF)
            pid_list = []
            p4_list = []

            kept_events = 0
            attempts = 0

            while kept_events < Ne:
                attempts += 1
                if DEBUG and attempts >= DEBUG_ATTEMPTS:
                    break
                if not pythia.next():
                    continue

                ev = pythia.event

                e_in, p_in = find_incoming_beams(ev)
                if e_in is None or p_in is None:
                    continue

                e_sc = get_scattered_electron(ev)
                if e_sc is None:
                    continue

                k_out = find_k_out(ev)
                if k_out is None:
                    continue

                e_in_arr[kept_events] = e_in
                p_in_arr[kept_events] = p_in
                e_sc_arr[kept_events] = p4_from_particle(e_sc)
                k_out_arr[kept_events] = k_out

                n_this = 0
                for i in range(ev.size()):
                    p = ev[i]
                    if p.isFinal():
                        pid_list.append(int(p.id()))
                        p4_list.append([p.e(), p.px(), p.py(), p.pz()])
                        n_this += 1

                offsets[kept_events + 1] = offsets[kept_events] + n_this
                kept_events += 1
                pbar.update(1)

            if DEBUG:
                print(f"[{label}] DEBUG: attempts={attempts}, kept={kept_events}")
                return

            pid = np.asarray(pid_list, dtype=DT_INT)
            p4 = np.asarray(p4_list, dtype=DT_FLOAT)

            E_e_meta, E_p_meta = _electron_proton_GeV(cfg)
            meta = {
                "label": label,
                "E_e": float(E_e_meta),
                "E_p": float(E_p_meta),
                "Q2_min": float(cfg.get("q2_min", Q2_MIN)),
                "seed": int(pythia.settings.mode("Random:seed")),
                **(config_meta or {}),
                "events_in_shard": int(Ne),
                "attempts_for_shard": int(attempts),
                "particles_in_shard": int(pid.shape[0]),
                "event_index_start": int(produced_total),
                "event_index_end_exclusive": int(produced_total + Ne),
                "format": {
                    "event_vectors": "float32 (E,px,py,pz)",
                    "p4": "float32 (E,px,py,pz)",
                    "pid": "int32",
                    "offsets": "int64 prefix sum into pid/p4",
                    "final_state_only": True,
                },
            }

            write_shard(label, shard_idx, e_in_arr, p_in_arr, e_sc_arr, k_out_arr, offsets, pid, p4, meta)

            produced_total += Ne
            print(
                f"[{label}] shard {shard_idx}: {Ne} events "
                f"(total={produced_total}/{n_events}), acceptance={Ne/attempts:.4f}"
            )
            shard_idx += 1

    print(f"[{label}] done.")
    pythia.stat()


# Label -> (idA, idB, eA, eB, isr_fsr_on, colour_reconnect_on, seed_offset)
# Used for CLI --labels and config_fingerprint; defaults to all three.
LABEL_CONFIGS = {
    "ETA_ON_CRON": {
        "idA": 2212,
        "idB": 11,
        "eA": E_P,
        "eB": E_E,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 100,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on"},
    },
    # Eta vs x–Q coverage grids (same physics setup as ETA_ON_CRON; PhaseSpace Q²_min = 4 GeV² → Q > 2 GeV)
    "ETA_XQ_5x41": {
        "idA": 2212,
        "idB": 11,
        "eA": 41.0,
        "eB": 5.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 210,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_9x41": {
        "idA": 2212,
        "idB": 11,
        "eA": 41.0,
        "eB": 9.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 211,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_9x100": {
        "idA": 2212,
        "idB": 11,
        "eA": 100.0,
        "eB": 9.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 212,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_5x100": {
        "idA": 2212,
        "idB": 11,
        "eA": 100.0,
        "eB": 5.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 214,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_10x100": {
        "idA": 2212,
        "idB": 11,
        "eA": 100.0,
        "eB": 10.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 216,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_10x275": {
        "idA": 2212,
        "idB": 11,
        "eA": 275.0,
        "eB": 10.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 217,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_18x275": {
        "idA": 2212,
        "idB": 11,
        "eA": 275.0,
        "eB": 18.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 218,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_5x275": {
        "idA": 2212,
        "idB": 11,
        "eA": 275.0,
        "eB": 5.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 215,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ETA_XQ_9x275": {
        "idA": 2212,
        "idB": 11,
        "eA": 275.0,
        "eB": 9.0,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 213,
        "q2_min": 4.0,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on", "purpose": "eta_xq_grid"},
    },
    "ISRFSR_ON": {
        "idA": 11,
        "idB": 2212,
        "eA": E_E,
        "eB": E_P,
        "isr_fsr_on": True,
        "colour_reconnect_on": False,
        "seed_offset": 1,
        "config_meta": {"Beams:idA": 11, "Beams:idB": 2212, "ColourReconnection": "off"},
    },
    "ISRFSR_OFF": {
        "idA": 11,
        "idB": 2212,
        "eA": E_E,
        "eB": E_P,
        "isr_fsr_on": False,
        "colour_reconnect_on": False,
        "seed_offset": 0,
        "config_meta": {"Beams:idA": 11, "Beams:idB": 2212, "ColourReconnection": "off"},
    },
}

# Keep default generation focused on legacy labels; eta–x–Q grids use ETA_XQ_* via --labels.
DEFAULT_LABELS = ["ETA_ON_CRON", "ISRFSR_ON", "ISRFSR_OFF"]


def main():
    parser = argparse.ArgumentParser(description="Generate raw PYTHIA shards per label.")
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help=f"Comma-separated labels to generate (default: legacy three). Valid: {list(LABEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--n_events",
        type=int,
        default=N_EVENTS_PER_CONFIG,
        help=f"Target events per selected label (default: {N_EVENTS_PER_CONFIG})",
    )
    parser.add_argument(
        "--events_per_shard",
        type=int,
        default=EVENTS_PER_SHARD,
        help=f"Events per shard (default: {EVENTS_PER_SHARD})",
    )
    args = parser.parse_args()

    requested = [s.strip() for s in args.labels.split(",") if s.strip()]
    unknown = [l for l in requested if l not in LABEL_CONFIGS]
    if unknown:
        parser.error(f"Unknown label(s): {unknown}. Valid: {list(LABEL_CONFIGS.keys())}")

    n_events = args.n_events
    events_per_shard = args.events_per_shard
    print(f"Labels: {requested}, n_events={n_events}, events_per_shard={events_per_shard}")

    root_abs = OUTDIR.resolve()
    print(f"Raw data root (absolute): {root_abs}")

    for label in requested:
        cfg = LABEL_CONFIGS[label].copy()
        config_meta = cfg.pop("config_meta", None)
        try:
            generate_config(
                label=label,
                cfg=cfg,
                n_events=n_events,
                events_per_shard=events_per_shard,
                config_meta=config_meta,
            )
            label_abs = (OUTDIR / label).resolve()
            print(f"Generated label {label} successfully -> {label_abs}")
        except Exception as e:
            print(f"FAILED label {label}: {e}")
            raise

    n_files, capped = count_files_under(OUTDIR)
    manifest_path = write_run_manifest(
        run_label="generate_events_raw",
        script_name="scripts/generation/generate_events_raw.py",
        top_level_dirs_written=[str(OUTDIR)],
        approximate_files_created=n_files,
        approximate_files_capped=capped,
        extra={
            "labels": requested,
            "n_events_per_label": n_events,
            "events_per_shard": events_per_shard,
        },
    )
    print(f"run_manifest={manifest_path}")
    print("\n=== generate_events_raw summary ===")
    print(f"Raw data root (absolute): {root_abs}")
    for lb in requested:
        print(f"  Label directory: {(OUTDIR / lb).resolve()}")


if __name__ == "__main__":
    main()
