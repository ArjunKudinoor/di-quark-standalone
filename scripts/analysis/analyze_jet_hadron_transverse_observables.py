#!/usr/bin/env python3
"""
Analyze three observables built from jet (outgoing struck quark) and hadron transverse momenta.

Jet identification:
  The jet is the **outgoing quark** after the hard scattering: lab-frame 4-momentum ``k_out``
  (stored per event as ``event_k_out``). In the Breit frame,
  ``pT_jet = (P_{jet,x}, P_{jet,y})`` with ``P_jet^{Breit} = boost(k_out)``.
  This matches the parton that fragments into the current hemisphere and carries the
  transverse physics from the hard process (including any injected k_T).

  (Previously the script used a collinear benchmark ``k_in_ref = x * P``, which has
  essentially zero ``pT`` in Breit and washes out jet--hadron correlations.)

Observables (all in Breit frame). The magnitude is taken only after vector addition or
subtraction. Do not use |pT_jet| + |pT_h| or |pT_jet| - |pT_h|.
  - Observable 1: angle between the two transverse vectors (convention [0, pi]).
  - Observable 2: first form the vector sum, then take the magnitude: |pT_jet + pT_h|.
  - Observable 3: first form the vector difference, then take the magnitude: |pT_jet - pT_h|.

Hadron identification:
  Target-fragmentation leading pion: highest-energy hadron in Breit target hemisphere
  (pz_breit > 0) with |pid| == 211. pT_h = (P_pi_breit[1], P_pi_breit[2]) in same frame.

Run from project root: python scripts/analysis/analyze_jet_hadron_transverse_observables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from diquark.paths import analysis_outputs_dir

_JH_OUT = analysis_outputs_dir()
_JH_OUT.mkdir(parents=True, exist_ok=True)

from diquark.analyze_events_raw import (
    FLIP_Z_PTREL,
    PTREL_LABEL_OFF,
    PTREL_LABEL_ON,
    Qmax_ptrel,
    Qmin_ptrel,
    build_LT,
    dot4,
    flip_z,
    is_hadron,
    list_shards,
    load_shard,
    p3,
    xmax_ptrel,
    xmin_ptrel,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Debug: print first N events and special cases; set to 0 to disable
DEBUG_N_EVENTS = 20
# Sign-consistency diagnostic: run with DEBUG_MAX_EVENTS=100 to get sign-check summary
# Set to int to limit total events per label (for quick transverse-plane check); None = process all
DEBUG_MAX_EVENTS = None  # Use 100 for sign-consistency check; None for full run
# Threshold below which pT is considered "very small" for angle (skip angle or flag)
PT_MIN_ANGLE = 1e-6
# Angle convention: "0_to_pi" -> [0, pi], 0 = aligned, pi = back-to-back
ANGLE_CONVENTION = "0_to_pi"

# Binning
ANGLE_BINS = 40
ANGLE_RANGE = (0.0, np.pi)
SUM_DIFF_BINS = 50
SUM_DIFF_MAX_GEV = 5.0
SUM_DIFF_RANGE = (0.0, SUM_DIFF_MAX_GEV)

# Outputs under DIQUARK_DATA_ROOT/.../outputs/analysis
OUTPUT_PREFIX = "jet_hadron_transverse"
HADRON_TAG = "target_leading_pion"
FRAME_TAG = "Breit"


def _pT_vec_breit(p4_breit: np.ndarray) -> np.ndarray:
    """Transverse 2-vector (px, py) from 4-vector in Breit frame."""
    return np.array([float(p4_breit[1]), float(p4_breit[2])], dtype=np.float64)


def _angle_between_2d(a: np.ndarray, b: np.ndarray) -> float | None:
    """
    Angle in [0, pi] between 2D vectors a and b.
    Returns None if either magnitude is below PT_MIN_ANGLE.
    """
    ma = float(np.linalg.norm(a))
    mb = float(np.linalg.norm(b))
    if ma < PT_MIN_ANGLE or mb < PT_MIN_ANGLE:
        return None
    cos_angle = float(np.dot(a, b)) / (ma * mb)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def run_azimuth_origin_diagnostic(label: str, max_events: int | None = 500) -> dict:
    """Diagnose where the preferred azimuthal direction is introduced.
    
    Collects azimuthal angles in lab and Breit frames for q, k_in, jet, hadron
    to determine if the Breit transform is rotating events to align q with +x.
    """
    shards = list_shards(label)
    if not shards:
        print(f"[{label}] No shards found for azimuth diagnostic.")
        return {}
    
    # Collectors
    phi_q_lab_list = []
    phi_kin_lab_list = []
    phi_q_breit_list = []
    phi_kin_breit_list = []
    phi_jet_breit_list = []
    phi_hadron_breit_list = []
    phi_jet_minus_q_breit_list = []
    phi_hadron_minus_jet_breit_list = []
    
    processed = 0
    debug_printed = 0
    DEBUG_PRINT_N = 10
    
    for shard_idx, shard_path in enumerate(shards):
        if max_events is not None and processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        k_out = data["event_k_out"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        Ne = e_in.shape[0]
        
        for ie in range(Ne):
            if max_events is not None and processed >= max_events:
                break
            e_in_ev = flip_z(np.asarray(e_in[ie], dtype=float), FLIP_Z_PTREL)
            p_in_ev = flip_z(np.asarray(p_in[ie], dtype=float), FLIP_Z_PTREL)
            e_sc_ev = flip_z(np.asarray(e_sc[ie], dtype=float), FLIP_Z_PTREL)
            k_out_ev = flip_z(np.asarray(k_out[ie], dtype=float), FLIP_Z_PTREL)
            Ep = float(p_in_ev[0])
            Ee = float(e_in_ev[0])
            q0 = e_in_ev[0] - e_sc_ev[0]
            q1 = e_in_ev[1] - e_sc_ev[1]
            q2 = e_in_ev[2] - e_sc_ev[2]
            q3 = e_in_ev[3] - e_sc_ev[3]
            qmu = np.array([q0, q1, q2, q3], dtype=float)
            Q2 = -(q0*q0 - q1*q1 - q2*q2 - q3*q3)
            if Q2 <= 0:
                continue
            Q = float(np.sqrt(Q2))
            qT = float(np.hypot(q1, q2))
            p_dot_q = p_in_ev[0]*q0 - p_in_ev[1]*q1 - p_in_ev[2]*q2 - p_in_ev[3]*q3
            if p_dot_q == 0:
                continue
            x = Q2 / (2.0 * p_dot_q)
            if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
                continue
            phiq = float(np.arctan2(q2, q1))
            S = 4.0 * Ee * Ep
            y = Q2 / (S * x)
            LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
            if LT is None:
                continue
            boost = lambda v: LT @ np.asarray(v, dtype=float)
            p_breit = boost(p_in_ev)
            P_plus = float(p_breit[0] + p_breit[3])
            if P_plus <= 0:
                continue
            
            # Hadron selection (same as main analysis)
            start = int(offsets[ie])
            end = int(offsets[ie + 1])
            best_tarE = -1.0
            best_tar_breit = None
            best_tar_pid = None
            for j in range(start, end):
                this_pid = int(pid[j])
                if not is_hadron(this_pid):
                    continue
                lab = flip_z(np.asarray(p4_arr[j], dtype=float), FLIP_Z_PTREL)
                trf = boost(lab)
                E_, px_, py_, pz_ = trf
                if pz_ <= 0:
                    continue
                if E_ > best_tarE:
                    best_tarE = E_
                    best_tar_breit = trf
                    best_tar_pid = this_pid
            if best_tar_breit is None or abs(best_tar_pid) != 211:
                continue
            
            # k_in in lab and Breit
            k_in = k_out_ev - qmu
            k_in_breit = boost(k_in)
            k_out_breit = boost(k_out_ev)
            
            # q in Breit
            q_breit = boost(qmu)
            
            # Remnant from incoming parton k_in; xL cut uses P_rem and q
            p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
            p_rem_truth_breit = boost(p_rem_truth)
            # Jet = outgoing quark (not k_in)
            P_jet_breit = k_out_breit
            
            # xL cut
            den = dot4(p_rem_truth_breit, q_breit)
            if den <= 0:
                continue
            xL_exact = dot4(best_tar_breit, q_breit) / den
            if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
                continue
            
            # Azimuthal angles in LAB frame
            phi_q_lab = float(np.arctan2(qmu[2], qmu[1]))
            phi_kin_lab = float(np.arctan2(k_in[2], k_in[1]))
            
            # Azimuthal angles in BREIT frame
            phi_q_breit = float(np.arctan2(q_breit[2], q_breit[1]))
            phi_kin_breit = float(np.arctan2(k_in_breit[2], k_in_breit[1]))
            phi_jet_breit = float(np.arctan2(P_jet_breit[2], P_jet_breit[1]))
            phi_hadron_breit = float(np.arctan2(best_tar_breit[2], best_tar_breit[1]))
            
            # Relative angles
            phi_jet_minus_q_breit = phi_jet_breit - phi_q_breit
            phi_hadron_minus_jet_breit = phi_hadron_breit - phi_jet_breit
            
            # Collect
            phi_q_lab_list.append(phi_q_lab)
            phi_kin_lab_list.append(phi_kin_lab)
            phi_q_breit_list.append(phi_q_breit)
            phi_kin_breit_list.append(phi_kin_breit)
            phi_jet_breit_list.append(phi_jet_breit)
            phi_hadron_breit_list.append(phi_hadron_breit)
            phi_jet_minus_q_breit_list.append(phi_jet_minus_q_breit)
            phi_hadron_minus_jet_breit_list.append(phi_hadron_minus_jet_breit)
            
            # Debug print for first few events
            if debug_printed < DEBUG_PRINT_N:
                print(f"---- Azimuth Diagnostic Event ({shard_idx}, {ie}) [{label}] ----")
                print(f"  q_lab         = [{qmu[1]:.6f}, {qmu[2]:.6f}, {qmu[3]:.6f}]")
                print(f"  q_breit       = [{q_breit[1]:.6f}, {q_breit[2]:.6f}, {q_breit[3]:.6f}]")
                print(f"  k_in_lab      = [{k_in[1]:.6f}, {k_in[2]:.6f}, {k_in[3]:.6f}]")
                print(f"  k_in_breit    = [{k_in_breit[1]:.6f}, {k_in_breit[2]:.6f}, {k_in_breit[3]:.6f}]")
                print(f"  jet_breit     = [{P_jet_breit[1]:.6f}, {P_jet_breit[2]:.6f}, {P_jet_breit[3]:.6f}]")
                print(f"  hadron_breit  = [{best_tar_breit[1]:.6f}, {best_tar_breit[2]:.6f}, {best_tar_breit[3]:.6f}]")
                print(f"  phi_q_lab     = {phi_q_lab:.6f} rad")
                print(f"  phi_q_breit   = {phi_q_breit:.6f} rad")
                print(f"  phi_kin_lab   = {phi_kin_lab:.6f} rad")
                print(f"  phi_kin_breit = {phi_kin_breit:.6f} rad")
                print(f"  phi_jet_breit = {phi_jet_breit:.6f} rad")
                print(f"  phi_had_breit = {phi_hadron_breit:.6f} rad")
                print(f"  phi_jet - phi_q (Breit) = {phi_jet_minus_q_breit:.6f} rad")
                debug_printed += 1
            
            processed += 1
    
    print(f"\n[{label}] Azimuth origin diagnostic: processed {processed} events")
    
    # Convert to arrays
    phi_q_lab_arr = np.asarray(phi_q_lab_list)
    phi_kin_lab_arr = np.asarray(phi_kin_lab_list)
    phi_q_breit_arr = np.asarray(phi_q_breit_list)
    phi_kin_breit_arr = np.asarray(phi_kin_breit_list)
    phi_jet_breit_arr = np.asarray(phi_jet_breit_list)
    phi_hadron_breit_arr = np.asarray(phi_hadron_breit_list)
    phi_jet_minus_q_breit_arr = np.asarray(phi_jet_minus_q_breit_list)
    
    # Compute Fourier moments for each distribution
    def moments(arr, name):
        if len(arr) == 0:
            return
        cos1 = float(np.mean(np.cos(arr)))
        sin1 = float(np.mean(np.sin(arr)))
        cos2 = float(np.mean(np.cos(2*arr)))
        sin2 = float(np.mean(np.sin(2*arr)))
        print(f"  {name}: <cos>={cos1: .6f}, <sin>={sin1: .6f}, <cos(2φ)>={cos2: .6f}, <sin(2φ)>={sin2: .6f}")
        return cos1, sin1, cos2, sin2
    
    print(f"\n[{label}] Fourier moments (expect ~0 if flat):")
    moments(phi_q_lab_arr, "phi_q_lab    ")
    moments(phi_kin_lab_arr, "phi_kin_lab  ")
    moments(phi_q_breit_arr, "phi_q_breit  ")
    moments(phi_kin_breit_arr, "phi_kin_breit")
    moments(phi_jet_breit_arr, "phi_jet_breit")
    moments(phi_hadron_breit_arr, "phi_had_breit")
    
    # Correlation: phi_jet_breit - phi_q_breit
    if len(phi_jet_minus_q_breit_arr) > 0:
        cos_diff = float(np.mean(np.cos(phi_jet_minus_q_breit_arr)))
        sin_diff = float(np.mean(np.sin(phi_jet_minus_q_breit_arr)))
        print(f"\n[{label}] Correlation: phi_jet_breit - phi_q_breit")
        print(f"  <cos(phi_jet - phi_q)> = {cos_diff: .6f}")
        print(f"  <sin(phi_jet - phi_q)> = {sin_diff: .6f}")
    
    # Key diagnostic checks
    print(f"\n[{label}] KEY DIAGNOSTIC CHECKS:")
    q_breit_near_zero = np.sum(np.abs(phi_q_breit_arr) < 0.01)
    jet_breit_near_zero = np.sum(np.abs(phi_jet_breit_arr) < 0.01)
    print(f"  A. Events with |phi_q_breit| < 0.01: {q_breit_near_zero} / {len(phi_q_breit_arr)} = {q_breit_near_zero/len(phi_q_breit_arr):.4f}")
    print(f"  B. Events with |phi_jet_breit| < 0.01: {jet_breit_near_zero} / {len(phi_jet_breit_arr)} = {jet_breit_near_zero/len(phi_jet_breit_arr):.4f}")
    
    # Check if phi_q_lab is flat vs phi_q_breit peaked
    if len(phi_q_lab_arr) > 0 and len(phi_q_breit_arr) > 0:
        std_q_lab = float(np.std(phi_q_lab_arr))
        std_q_breit = float(np.std(phi_q_breit_arr))
        print(f"  C. std(phi_q_lab) = {std_q_lab:.4f}, std(phi_q_breit) = {std_q_breit:.6f}")
        print(f"     Ratio: std(lab)/std(breit) = {std_q_lab/max(std_q_breit, 1e-10):.1f}")
    
    return {
        "phi_q_lab": phi_q_lab_arr,
        "phi_kin_lab": phi_kin_lab_arr,
        "phi_q_breit": phi_q_breit_arr,
        "phi_kin_breit": phi_kin_breit_arr,
        "phi_jet_breit": phi_jet_breit_arr,
        "phi_hadron_breit": phi_hadron_breit_arr,
    }


def run_breit_4vector_diagnostic(label: str, max_events: int | None = 2000) -> dict:
    """Diagnostic of k_in, P (proton), and q (virtual photon) 4-vectors in Breit frame.
    
    Verifies actual directions and transverse components after the Breit transform.
    """
    shards = list_shards(label)
    if not shards:
        print(f"[{label}] No shards found for Breit 4-vector diagnostic.")
        return {}
    
    # Collectors for distributions
    qT_mag_list = []
    PT_mag_list = []
    kinT_mag_list = []
    P_x_list = []
    P_y_list = []
    kin_x_list = []
    kin_y_list = []
    q_x_list = []
    q_y_list = []
    
    # Scalar products
    P_dot_q_list = []
    P_dot_kin_list = []
    q_dot_kin_list = []
    
    processed = 0
    debug_printed = 0
    DEBUG_PRINT_N = 10
    
    for shard_idx, shard_path in enumerate(shards):
        if max_events is not None and processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        k_out = data["event_k_out"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        Ne = e_in.shape[0]
        
        for ie in range(Ne):
            if max_events is not None and processed >= max_events:
                break
            e_in_ev = flip_z(np.asarray(e_in[ie], dtype=float), FLIP_Z_PTREL)
            p_in_ev = flip_z(np.asarray(p_in[ie], dtype=float), FLIP_Z_PTREL)
            e_sc_ev = flip_z(np.asarray(e_sc[ie], dtype=float), FLIP_Z_PTREL)
            k_out_ev = flip_z(np.asarray(k_out[ie], dtype=float), FLIP_Z_PTREL)
            Ep = float(p_in_ev[0])
            Ee = float(e_in_ev[0])
            q0 = e_in_ev[0] - e_sc_ev[0]
            q1 = e_in_ev[1] - e_sc_ev[1]
            q2 = e_in_ev[2] - e_sc_ev[2]
            q3 = e_in_ev[3] - e_sc_ev[3]
            qmu = np.array([q0, q1, q2, q3], dtype=float)
            Q2 = -(q0*q0 - q1*q1 - q2*q2 - q3*q3)
            if Q2 <= 0:
                continue
            Q = float(np.sqrt(Q2))
            qT_lab = float(np.hypot(q1, q2))
            p_dot_q = p_in_ev[0]*q0 - p_in_ev[1]*q1 - p_in_ev[2]*q2 - p_in_ev[3]*q3
            if p_dot_q == 0:
                continue
            x = Q2 / (2.0 * p_dot_q)
            if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
                continue
            phiq = float(np.arctan2(q2, q1))
            S = 4.0 * Ee * Ep
            y = Q2 / (S * x)
            LT = build_LT(Ee, Ep, qmu, x, y, qT_lab, phiq, S)
            if LT is None:
                continue
            boost = lambda v: LT @ np.asarray(v, dtype=float)
            
            # Boost to Breit frame
            q_breit = boost(qmu)
            P_breit = boost(p_in_ev)
            
            P_plus = float(P_breit[0] + P_breit[3])
            if P_plus <= 0:
                continue
            
            # k_in in lab and Breit
            k_in = k_out_ev - qmu
            k_in_breit = boost(k_in)
            
            # Hadron selection (same cuts as main analysis)
            start = int(offsets[ie])
            end = int(offsets[ie + 1])
            best_tarE = -1.0
            best_tar_breit = None
            best_tar_pid = None
            for j in range(start, end):
                this_pid = int(pid[j])
                if not is_hadron(this_pid):
                    continue
                lab = flip_z(np.asarray(p4_arr[j], dtype=float), FLIP_Z_PTREL)
                trf = boost(lab)
                E_, px_, py_, pz_ = trf
                if pz_ <= 0:
                    continue
                if E_ > best_tarE:
                    best_tarE = E_
                    best_tar_breit = trf
                    best_tar_pid = this_pid
            if best_tar_breit is None or abs(best_tar_pid) != 211:
                continue
            
            # xL cut
            p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
            p_rem_truth_breit = boost(p_rem_truth)
            den = dot4(p_rem_truth_breit, q_breit)
            if den <= 0:
                continue
            xL_exact = dot4(best_tar_breit, q_breit) / den
            if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
                continue
            
            # Extract transverse components
            qT_vec = np.array([q_breit[1], q_breit[2]])
            PT_vec = np.array([P_breit[1], P_breit[2]])
            kinT_vec = np.array([k_in_breit[1], k_in_breit[2]])
            
            qT_mag = float(np.linalg.norm(qT_vec))
            PT_mag = float(np.linalg.norm(PT_vec))
            kinT_mag = float(np.linalg.norm(kinT_vec))
            
            # Scalar products (Minkowski metric: + - - -)
            P_dot_q = dot4(P_breit, q_breit)
            P_dot_kin = dot4(P_breit, k_in_breit)
            q_dot_kin = dot4(q_breit, k_in_breit)
            
            # Collect
            qT_mag_list.append(qT_mag)
            PT_mag_list.append(PT_mag)
            kinT_mag_list.append(kinT_mag)
            P_x_list.append(P_breit[1])
            P_y_list.append(P_breit[2])
            kin_x_list.append(k_in_breit[1])
            kin_y_list.append(k_in_breit[2])
            q_x_list.append(q_breit[1])
            q_y_list.append(q_breit[2])
            P_dot_q_list.append(P_dot_q)
            P_dot_kin_list.append(P_dot_kin)
            q_dot_kin_list.append(q_dot_kin)
            
            # Debug print for first few events
            if debug_printed < DEBUG_PRINT_N:
                print(f"\n---- Breit 4-Vector Diagnostic Event ({shard_idx}, {ie}) [{label}] ----")
                print(f"  q_breit     = ({q_breit[0]:.6f}, {q_breit[1]:.6e}, {q_breit[2]:.6e}, {q_breit[3]:.6f})")
                print(f"  P_breit     = ({P_breit[0]:.6f}, {P_breit[1]:.6e}, {P_breit[2]:.6e}, {P_breit[3]:.6f})")
                print(f"  k_in_breit  = ({k_in_breit[0]:.6f}, {k_in_breit[1]:.6e}, {k_in_breit[2]:.6e}, {k_in_breit[3]:.6f})")
                print(f"  |qT|        = {qT_mag:.6e}")
                print(f"  |P_T|       = {PT_mag:.6e}")
                print(f"  |k_in,T|    = {kinT_mag:.6f}")
                if PT_mag > 1e-10:
                    phi_P = float(np.arctan2(P_breit[2], P_breit[1]))
                    print(f"  phi_P       = {phi_P:.6f} rad")
                else:
                    print(f"  phi_P       = undefined (|P_T| ~ 0)")
                if kinT_mag > 1e-10:
                    phi_kin = float(np.arctan2(k_in_breit[2], k_in_breit[1]))
                    print(f"  phi_kin     = {phi_kin:.6f} rad")
                else:
                    print(f"  phi_kin     = undefined (|k_in,T| ~ 0)")
                print(f"  P·q         = {P_dot_q:.6f}")
                print(f"  P·k_in      = {P_dot_kin:.6f}")
                print(f"  q·k_in      = {q_dot_kin:.6f}")
                debug_printed += 1
            
            processed += 1
    
    print(f"\n[{label}] Breit 4-vector diagnostic: processed {processed} events")
    
    # Convert to arrays
    qT_mag_arr = np.asarray(qT_mag_list)
    PT_mag_arr = np.asarray(PT_mag_list)
    kinT_mag_arr = np.asarray(kinT_mag_list)
    P_x_arr = np.asarray(P_x_list)
    P_y_arr = np.asarray(P_y_list)
    kin_x_arr = np.asarray(kin_x_list)
    kin_y_arr = np.asarray(kin_y_list)
    
    n = len(qT_mag_arr)
    if n == 0:
        print(f"[{label}] No events processed.")
        return {}
    
    # Summary statistics
    print(f"\n[{label}] SUMMARY STATISTICS:")
    print(f"  A. |qT| distribution:")
    print(f"     min = {np.min(qT_mag_arr):.6e}, max = {np.max(qT_mag_arr):.6e}, mean = {np.mean(qT_mag_arr):.6e}")
    print(f"  B. |P_T| distribution:")
    print(f"     min = {np.min(PT_mag_arr):.6e}, max = {np.max(PT_mag_arr):.6e}, mean = {np.mean(PT_mag_arr):.6e}")
    print(f"  C. |k_in,T| distribution:")
    print(f"     min = {np.min(kinT_mag_arr):.6f}, max = {np.max(kinT_mag_arr):.6f}, mean = {np.mean(kinT_mag_arr):.6f}")
    
    frac_qT_zero = np.sum(qT_mag_arr < 1e-8) / n
    frac_PT_zero = np.sum(PT_mag_arr < 1e-8) / n
    frac_kinT_zero = np.sum(kinT_mag_arr < 1e-8) / n
    print(f"\n  D. Fraction with |qT| < 1e-8:     {frac_qT_zero:.6f}")
    print(f"  E. Fraction with |P_T| < 1e-8:    {frac_PT_zero:.6f}")
    print(f"  F. Fraction with |k_in,T| < 1e-8: {frac_kinT_zero:.6f}")
    
    print(f"\n  G. Averages:")
    print(f"     <P_x>     = {np.mean(P_x_arr): .6e}")
    print(f"     <P_y>     = {np.mean(P_y_arr): .6e}")
    print(f"     <k_in,x>  = {np.mean(kin_x_arr): .6f}")
    print(f"     <k_in,y>  = {np.mean(kin_y_arr): .6e}")
    print(f"     <|k_in,T|>= {np.mean(kinT_mag_arr): .6f}")
    
    # Check alignment with x-axis
    if np.mean(kinT_mag_arr) > 0.01:
        cos_phi_kin = kin_x_arr / np.maximum(kinT_mag_arr, 1e-10)
        mean_cos = np.mean(cos_phi_kin)
        print(f"\n  H. k_in,T alignment with x-axis:")
        print(f"     <cos(phi_kin)> = {mean_cos: .6f}")
        print(f"     <sin(phi_kin)> = {np.mean(kin_y_arr / np.maximum(kinT_mag_arr, 1e-10)): .6e}")
        if mean_cos > 0.9:
            print(f"     -> k_in,T is STRONGLY aligned with +x axis")
        elif mean_cos > 0.5:
            print(f"     -> k_in,T has preference for +x axis")
        else:
            print(f"     -> k_in,T is NOT strongly aligned with x axis")
    
    # Check proton alignment
    if np.mean(PT_mag_arr) > 1e-6:
        cos_phi_P = P_x_arr / np.maximum(PT_mag_arr, 1e-10)
        print(f"\n  I. P_T alignment:")
        print(f"     <cos(phi_P)> = {np.mean(cos_phi_P): .6e}")
    
    # Plain-language answers
    print(f"\n" + "=" * 70)
    print(f"[{label}] ANSWERS TO KEY QUESTIONS:")
    print("=" * 70)
    
    # 1. Is q_breit purely longitudinal?
    if np.mean(qT_mag_arr) < 1e-6:
        print(f"1. Is q_breit purely longitudinal? YES (mean |qT| = {np.mean(qT_mag_arr):.2e})")
    else:
        print(f"1. Is q_breit purely longitudinal? NO (mean |qT| = {np.mean(qT_mag_arr):.2e})")
    
    # 2. Does proton have nonzero pT?
    if np.mean(PT_mag_arr) > 1e-6:
        print(f"2. Does proton have nonzero pT in Breit? YES (mean |P_T| = {np.mean(PT_mag_arr):.2e})")
    else:
        print(f"2. Does proton have nonzero pT in Breit? NO (mean |P_T| = {np.mean(PT_mag_arr):.2e})")
    
    # 3. Does k_in have transverse momentum?
    if np.mean(kinT_mag_arr) > 0.01:
        print(f"3. Does k_in have transverse momentum? YES (mean |k_in,T| = {np.mean(kinT_mag_arr):.4f} GeV)")
    else:
        print(f"3. Does k_in have transverse momentum? NO (mean |k_in,T| = {np.mean(kinT_mag_arr):.2e})")
    
    # 4. Is k_in,T aligned with x?
    if np.mean(kinT_mag_arr) > 0.01:
        cos_phi_kin = kin_x_arr / np.maximum(kinT_mag_arr, 1e-10)
        mean_cos = np.mean(cos_phi_kin)
        if mean_cos > 0.9:
            print(f"4. Is k_in,T aligned with +x axis? YES (<cos(phi_kin)> = {mean_cos:.4f})")
        else:
            print(f"4. Is k_in,T aligned with +x axis? NO (<cos(phi_kin)> = {mean_cos:.4f})")
    else:
        print(f"4. Is k_in,T aligned with +x axis? N/A (|k_in,T| ~ 0)")
    
    # 5. Does transform align proton pT with x?
    if np.mean(PT_mag_arr) > 1e-6:
        cos_phi_P = P_x_arr / np.maximum(PT_mag_arr, 1e-10)
        mean_cos_P = np.mean(cos_phi_P)
        if abs(mean_cos_P) > 0.9:
            print(f"5. Does transform align P_T with x? YES (<cos(phi_P)> = {mean_cos_P:.4f})")
        else:
            print(f"5. Does transform align P_T with x? NO (<cos(phi_P)> = {mean_cos_P:.4f})")
    else:
        print(f"5. Does transform align P_T with x? N/A (|P_T| ~ 0)")
    
    print("=" * 70)
    
    return {
        "qT_mag": qT_mag_arr,
        "PT_mag": PT_mag_arr,
        "kinT_mag": kinT_mag_arr,
        "kin_x": kin_x_arr,
        "kin_y": kin_y_arr,
        "P_x": P_x_arr,
        "P_y": P_y_arr,
    }


def run_observables_for_label(
    label: str,
    max_events: int | None,
    out_angle: list[float],
    out_sum_mag: list[float],
    out_diff_mag: list[float],
    out_event_ids: list[tuple[int, int]],
    out_pT_jet_vec: list[np.ndarray],
    out_pT_hadron_vec: list[np.ndarray],
    out_debug: list[dict],
    out_phi_jet: list[float],
):
    """Fill observable lists for one label (ISRFSR_OFF or ISRFSR_ON).
    
    out_phi_jet collects the jet azimuthal angle phi_jet = atan2(pT_jet_y, pT_jet_x)
    for the jet azimuth diagnostic study (convention: [-pi, pi)).
    """
    shards = list_shards(label)
    if not shards:
        print(f"[{label}] No shards found.")
        return

    processed = 0
    # Collect candidates for debug: nearly parallel, nearly back-to-back, one pT small
    debug_parallel: list[tuple[int, dict]] = []
    debug_backtoback: list[tuple[int, dict]] = []
    debug_small_pt: list[tuple[int, dict]] = []
    # For hemisphere/alignment summary
    n_h_closer_to_remnant = 0  # angle(hadron, remnant) < angle(hadron, jet)
    n_h_closer_to_jet = 0  # angle(hadron, jet) < angle(hadron, remnant)
    # Sign-consistency diagnostic
    n_jet_eq_kin = 0  # |pT_jet_current - pT_kin| small
    n_jet_eq_minus_kin = 0  # |pT_jet_current + pT_kin| small
    n_angle_hR_near_zero = 0  # angle(hadron, remnant) < 0.2
    sign_check_tol = 1e-6

    for shard_idx, shard_path in enumerate(shards):
        if max_events is not None and processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        k_out = data["event_k_out"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        Ne = e_in.shape[0]

        for ie in range(Ne):
            if max_events is not None and processed >= max_events:
                break
            e_in_ev = flip_z(np.asarray(e_in[ie], dtype=float), FLIP_Z_PTREL)
            p_in_ev = flip_z(np.asarray(p_in[ie], dtype=float), FLIP_Z_PTREL)
            e_sc_ev = flip_z(np.asarray(e_sc[ie], dtype=float), FLIP_Z_PTREL)
            k_out_ev = flip_z(np.asarray(k_out[ie], dtype=float), FLIP_Z_PTREL)
            Ep = float(p_in_ev[0])
            Ee = float(e_in_ev[0])
            q0 = e_in_ev[0] - e_sc_ev[0]
            q1 = e_in_ev[1] - e_sc_ev[1]
            q2 = e_in_ev[2] - e_sc_ev[2]
            q3 = e_in_ev[3] - e_sc_ev[3]
            qmu = np.array([q0, q1, q2, q3], dtype=float)
            Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
            if Q2 <= 0:
                continue
            Q = float(np.sqrt(Q2))
            qT = float(np.hypot(q1, q2))
            p_dot_q = p_in_ev[0] * q0 - p_in_ev[1] * q1 - p_in_ev[2] * q2 - p_in_ev[3] * q3
            if p_dot_q == 0:
                continue
            x = Q2 / (2.0 * p_dot_q)
            if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
                continue
            phiq = float(np.arctan2(q2, q1))
            S = 4.0 * Ee * Ep
            y = Q2 / (S * x)
            LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
            if LT is None:
                continue
            boost = lambda v: LT @ np.asarray(v, dtype=float)
            p_breit = boost(p_in_ev)
            P_plus = float(p_breit[0] + p_breit[3])
            if P_plus <= 0:
                continue

            start = int(offsets[ie])
            end = int(offsets[ie + 1])
            best_tarE = -1.0
            best_tar_breit = None
            best_tar_pid = None
            for j in range(start, end):
                this_pid = int(pid[j])
                if not is_hadron(this_pid):
                    continue
                lab = flip_z(np.asarray(p4_arr[j], dtype=float), FLIP_Z_PTREL)
                trf = boost(lab)
                E_, px_, py_, pz_ = trf
                if pz_ <= 0:
                    continue
                if E_ > best_tarE:
                    best_tarE = E_
                    best_tar_breit = trf
                    best_tar_pid = this_pid
            if best_tar_breit is None or abs(best_tar_pid) != 211:
                continue

            # Remnant from k_in = k_out - q (same as xL convention elsewhere)
            k_in_lab = k_out_ev - qmu
            k_in_breit = boost(k_in_lab)
            p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in_lab
            p_rem_truth_breit = boost(p_rem_truth)
            if np.linalg.norm(p3(p_rem_truth_breit)) <= 0:
                continue
            q_breit = boost(qmu)
            den = dot4(p_rem_truth_breit, q_breit)
            if den <= 0:
                continue
            xL_exact = dot4(best_tar_breit, q_breit) / den
            if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
                continue

            # Jet = outgoing quark k_out (Breit); incoming k_in kept for remnant / diagnostics
            k_out_breit = boost(k_out_ev)
            pT_remnant_breit = _pT_vec_breit(p_rem_truth_breit)
            pT_jet_vec = _pT_vec_breit(k_out_breit)
            pT_jet_current = pT_jet_vec
            pT_jet_mag = float(np.linalg.norm(pT_jet_vec))
            pT_kin = _pT_vec_breit(k_in_breit)

            # Hadron: target-leading pion
            pT_hadron_vec = _pT_vec_breit(best_tar_breit)
            pT_hadron_mag = float(np.linalg.norm(pT_hadron_vec))

            # Observable 1: angle between the two transverse vectors [0, pi]
            angle = _angle_between_2d(pT_jet_vec, pT_hadron_vec)
            angle_hR = _angle_between_2d(pT_hadron_vec, pT_remnant_breit)
            angle_RJ = _angle_between_2d(pT_remnant_breit, pT_jet_vec)
            angle_h_kin = _angle_between_2d(pT_hadron_vec, pT_kin)
            # Compare outgoing (k_out) vs incoming (k_in) transverse in Breit
            diff_minus = np.linalg.norm(pT_jet_current - pT_kin)
            diff_plus = np.linalg.norm(pT_jet_current + pT_kin)
            if diff_minus < sign_check_tol:
                n_jet_eq_kin += 1
            elif diff_plus < sign_check_tol:
                n_jet_eq_minus_kin += 1
            if angle_hR is not None and angle_hR < 0.2:
                n_angle_hR_near_zero += 1
            if angle is not None and angle_hR is not None:
                if angle_hR < angle:
                    n_h_closer_to_remnant += 1
                elif angle < angle_hR:
                    n_h_closer_to_jet += 1

            # Observable 2: first form vector sum, then magnitude: |pT_jet + pT_h|
            pT_sum_vec = pT_jet_vec + pT_hadron_vec
            sum_mag = float(np.linalg.norm(pT_sum_vec))

            # Observable 3: first form vector difference, then magnitude: |pT_jet - pT_h|
            pT_diff_vec = pT_jet_vec - pT_hadron_vec
            diff_mag = float(np.linalg.norm(pT_diff_vec))

            # Jet azimuth diagnostic: phi_jet = atan2(pT_jet_y, pT_jet_x), convention [-pi, pi)
            phi_jet = float(np.arctan2(pT_jet_vec[1], pT_jet_vec[0]))

            event_id = (int(shard_idx), int(ie))
            rec = {
                "event_id": event_id,
                "k_out_4mom_breit": np.asarray(k_out_breit, dtype=np.float64),
                "pT_jet_vec": pT_jet_vec.copy(),
                "hadron_4mom_breit": np.asarray(best_tar_breit, dtype=np.float64),
                "hadron_pT_vec": pT_hadron_vec.copy(),
                "angle": angle,
                "sum_mag": sum_mag,
                "diff_mag": diff_mag,
                "phi_jet": phi_jet,
            }

            if processed < DEBUG_N_EVENTS or len(out_debug) < DEBUG_N_EVENTS:
                out_debug.append(rec)
            if angle is not None and processed < DEBUG_N_EVENTS:
                print(f"---- Event {event_id} (ordinal {processed}) [{label}] ----")
                print("  1. Transverse vectors:")
                print("     pT_hadron       =", pT_hadron_vec)
                print("     pT_remnant      =", pT_remnant_breit)
                print("     pT_jet_current  =", pT_jet_current)
                print("     pT_kin          =", pT_kin, "(k_in = k_out - q, Breit transverse)")
                print(f"  2. Jet azimuth: phi_jet = atan2({pT_jet_vec[1]:.6f}, {pT_jet_vec[0]:.6f}) = {phi_jet:.6f} rad")
                print("  3. Sign check: |pT_jet_current - pT_kin| =", diff_minus)
                print("                 |pT_jet_current + pT_kin| =", diff_plus)
                print("  4. Angles:")
                print("     angle(hadron, remnant)   =", angle_hR)
                print("     angle(hadron, jet_current) =", angle)
                print("     angle(hadron, k_in_direct) =", angle_h_kin)

            # Sanity-check cases
            if angle is not None:
                if angle < 0.1 and pT_jet_mag > 0.01 and pT_hadron_mag > 0.01:
                    debug_parallel.append((processed, rec))
                if angle > np.pi - 0.1 and pT_jet_mag > 0.01 and pT_hadron_mag > 0.01:
                    debug_backtoback.append((processed, rec))
            if pT_jet_mag < 0.01 or pT_hadron_mag < 0.01:
                debug_small_pt.append((processed, rec))

            # Store (include event even if angle is None; use nan for angle in that case)
            out_event_ids.append(event_id)
            out_pT_jet_vec.append(pT_jet_vec)
            out_pT_hadron_vec.append(pT_hadron_vec)
            out_sum_mag.append(sum_mag)
            out_diff_mag.append(diff_mag)
            out_phi_jet.append(phi_jet)
            if angle is not None:
                out_angle.append(angle)
            else:
                out_angle.append(np.nan)
            processed += 1

        print(f"[{label}] shard {shard_idx} ({shard_path.name}): {processed} events")

    # Print sanity-check cases
    if debug_parallel:
        print(f"\n[{label}] Sample events with vectors nearly parallel (angle < 0.1):")
        for idx, (ord_idx, r) in enumerate(debug_parallel[:3]):
            print(f"  ordinal {ord_idx}: angle={r['angle']:.6f} |pT_sum|={r['sum_mag']:.6f} |pT_diff|={r['diff_mag']:.6f}")
    if debug_backtoback:
        print(f"\n[{label}] Sample events with vectors nearly back-to-back (angle > pi-0.1):")
        for idx, (ord_idx, r) in enumerate(debug_backtoback[:3]):
            print(f"  ordinal {ord_idx}: angle={r['angle']:.6f} |pT_sum|={r['sum_mag']:.6f} |pT_diff|={r['diff_mag']:.6f}")
    if debug_small_pt:
        print(f"\n[{label}] Sample events with one pT very small:")
        for idx, (ord_idx, r) in enumerate(debug_small_pt[:3]):
            print(f"  ordinal {ord_idx}: |pT_jet|={np.linalg.norm(r['pT_jet_vec']):.6e} |pT_h|={np.linalg.norm(r['hadron_pT_vec']):.6e} angle={r['angle']}")

    # Summary: hadron alignment with remnant vs jet
    total_compared = n_h_closer_to_remnant + n_h_closer_to_jet
    if total_compared > 0:
        print(f"\n[{label}] Hadron alignment summary (transverse angle):")
        print(f"  angle(hadron, remnant) < angle(hadron, jet): {n_h_closer_to_remnant} / {total_compared} = {n_h_closer_to_remnant / total_compared:.4f}")
        print(f"  angle(hadron, jet) < angle(hadron, remnant): {n_h_closer_to_jet} / {total_compared} = {n_h_closer_to_jet / total_compared:.4f}")

    # Outgoing vs incoming transverse (k_out vs k_in) — usually differ unless collinear limit
    if processed > 0:
        print(f"\n[{label}] k_out vs k_in transverse (Breit):")
        print(f"  pT_jet = pT(k_out):   matches pT(k_in) in {n_jet_eq_kin} / {processed} events (tol {sign_check_tol})")
        print(f"  pT_jet = -pT(k_in):   matches in {n_jet_eq_minus_kin} / {processed} events")
        print(f"  angle(hadron, remnant) < 0.2: {n_angle_hR_near_zero} / {processed}")
        if n_jet_eq_kin == processed:
            print("  NOTE: pT(k_out) equals pT(k_in) in all events (unusual unless collinear).")
        elif n_jet_eq_minus_kin == processed:
            print("  NOTE: pT(k_out) = -pT(k_in) in all events.")
        elif n_jet_eq_kin == 0 and n_jet_eq_minus_kin == 0:
            print("  (Expected for generic events: outgoing quark pT differs from incoming parton pT.)")

    # Jet azimuth Fourier moments: <cos(phi_jet)>, <sin(phi_jet)>, <cos(2*phi_jet)>, <sin(2*phi_jet)>
    if out_phi_jet:
        phi_arr = np.asarray(out_phi_jet, dtype=np.float64)
        n_phi = len(phi_arr)
        mean_cos1 = float(np.mean(np.cos(phi_arr)))
        mean_sin1 = float(np.mean(np.sin(phi_arr)))
        mean_cos2 = float(np.mean(np.cos(2 * phi_arr)))
        mean_sin2 = float(np.mean(np.sin(2 * phi_arr)))
        print(f"\n[{label}] Jet azimuth Fourier moments (phi_jet convention: [-pi, pi)):")
        print(f"  N = {n_phi}")
        print(f"  <cos(phi_jet)>   = {mean_cos1: .6f}")
        print(f"  <sin(phi_jet)>   = {mean_sin1: .6f}")
        print(f"  <cos(2*phi_jet)> = {mean_cos2: .6f}")
        print(f"  <sin(2*phi_jet)> = {mean_sin2: .6f}")
        if abs(mean_cos1) < 0.01 and abs(mean_sin1) < 0.01 and abs(mean_cos2) < 0.01 and abs(mean_sin2) < 0.01:
            print("  -> Jet azimuth is consistent with flat (all moments near zero).")
        else:
            print("  -> Jet azimuth has non-zero Fourier moments; distribution is NOT flat.")


def main():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "stix",
    })
    fontsize = 14

    # -------------------------------------------------------------------------
    # Azimuth Origin Diagnostic: check where preferred azimuthal direction is introduced
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("AZIMUTH ORIGIN DIAGNOSTIC")
    print("=" * 70)
    print("\nRunning azimuth origin diagnostic (ISR/FSR OFF)...")
    azimuth_diag_off = run_azimuth_origin_diagnostic(PTREL_LABEL_OFF, max_events=1000)
    print("\nRunning azimuth origin diagnostic (ISR/FSR ON)...")
    azimuth_diag_on = run_azimuth_origin_diagnostic(PTREL_LABEL_ON, max_events=1000)
    
    # Create histograms for the azimuth diagnostic
    print("\nCreating azimuth origin diagnostic histograms...")
    PHI_BINS = 40
    phi_range = (-np.pi, np.pi)
    bins_phi = np.linspace(phi_range[0], phi_range[1], PHI_BINS + 1)
    binw_phi = (phi_range[1] - phi_range[0]) / PHI_BINS
    centers_phi = 0.5 * (bins_phi[:-1] + bins_phi[1:])
    
    def plot_phi_hist(arr_off, arr_on, name, xlabel, filename):
        if len(arr_off) == 0 and len(arr_on) == 0:
            return
        fig, ax = plt.subplots()
        if len(arr_off) > 0:
            hist_off, _ = np.histogram(arr_off, bins=bins_phi)
            dens_off = hist_off / (len(arr_off) * binw_phi)
            ax.step(centers_phi, dens_off, where="mid", label="ISR/FSR Off", color="blue", linewidth=1.5)
        if len(arr_on) > 0:
            hist_on, _ = np.histogram(arr_on, bins=bins_phi)
            dens_on = hist_on / (len(arr_on) * binw_phi)
            ax.step(centers_phi, dens_on, where="mid", label="ISR/FSR On", color="red", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi}$")
        ax.set_xlim(phi_range)
        ax.legend(loc="best", frameon=False)
        ax.tick_params(direction="in")
        plt.tight_layout()
        out_path = _JH_OUT / filename
        plt.savefig(out_path, format="pdf")
        plt.close(fig)
        print(f"Saved: {out_path.name}")
    
    plot_phi_hist(azimuth_diag_off.get("phi_q_lab", []), azimuth_diag_on.get("phi_q_lab", []),
                  "phi_q_lab", r"$\phi_q$ (lab) [rad]", "azimuth_diag_phi_q_lab.pdf")
    plot_phi_hist(azimuth_diag_off.get("phi_q_breit", []), azimuth_diag_on.get("phi_q_breit", []),
                  "phi_q_breit", r"$\phi_q$ (Breit) [rad]", "azimuth_diag_phi_q_breit.pdf")
    plot_phi_hist(azimuth_diag_off.get("phi_kin_lab", []), azimuth_diag_on.get("phi_kin_lab", []),
                  "phi_kin_lab", r"$\phi_{k_{\mathrm{in}}}$ (lab) [rad]", "azimuth_diag_phi_kin_lab.pdf")
    plot_phi_hist(azimuth_diag_off.get("phi_kin_breit", []), azimuth_diag_on.get("phi_kin_breit", []),
                  "phi_kin_breit", r"$\phi_{k_{\mathrm{in}}}$ (Breit) [rad]", "azimuth_diag_phi_kin_breit.pdf")
    plot_phi_hist(azimuth_diag_off.get("phi_jet_breit", []), azimuth_diag_on.get("phi_jet_breit", []),
                  "phi_jet_breit", r"$\phi_{\mathrm{jet}}$ (Breit) [rad]", "azimuth_diag_phi_jet_breit.pdf")
    plot_phi_hist(azimuth_diag_off.get("phi_hadron_breit", []), azimuth_diag_on.get("phi_hadron_breit", []),
                  "phi_hadron_breit", r"$\phi_{\mathrm{hadron}}$ (Breit) [rad]", "azimuth_diag_phi_hadron_breit.pdf")
    
    # Print summary interpretation
    print("\n" + "=" * 70)
    print("AZIMUTH ORIGIN DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print("""
The Breit-like transform in build_LT() includes a rotation matrix M0:
    M0 = rotation by -phiq in the x-y plane
where phiq = atan2(q_y, q_x) is the azimuthal angle of the virtual photon q
in the lab frame.

This rotation aligns q_T with the +x axis event-by-event.

INTERPRETATION:
- If phi_q_lab is flat but phi_q_breit is sharply peaked at 0:
  -> The transform explicitly rotates each event to align q with +x.
- If phi_jet_breit is also peaked at 0:
  -> The outgoing quark (k_out) may lie approximately along +x in Breit (kinematics-dependent).
- The hadron-jet angle phi_{hJ} is therefore measured in an event-by-event
  rotated coordinate system, NOT in a fixed physical azimuth.
- The non-flat phi_jet distribution is a coordinate artifact, not a physical
  anisotropy in the original events.
""")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Breit 4-Vector Diagnostic: examine q, P, k_in 4-vectors directly
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BREIT 4-VECTOR DIAGNOSTIC")
    print("=" * 70)
    print("\nRunning Breit 4-vector diagnostic (ISR/FSR OFF)...")
    breit_diag_off = run_breit_4vector_diagnostic(PTREL_LABEL_OFF, max_events=2000)
    print("\nRunning Breit 4-vector diagnostic (ISR/FSR ON)...")
    breit_diag_on = run_breit_4vector_diagnostic(PTREL_LABEL_ON, max_events=2000)
    print("=" * 70)

    print("\nProceeding with main observables analysis...\n")

    # -------------------------------------------------------------------------
    # Main observables analysis
    # -------------------------------------------------------------------------

    # Run for both labels
    angle_off: list[float] = []
    sum_mag_off: list[float] = []
    diff_mag_off: list[float] = []
    event_ids_off: list[tuple[int, int]] = []
    pT_jet_vec_off: list[np.ndarray] = []
    pT_hadron_vec_off: list[np.ndarray] = []
    debug_off: list[dict] = []
    phi_jet_off: list[float] = []

    angle_on: list[float] = []
    sum_mag_on: list[float] = []
    diff_mag_on: list[float] = []
    event_ids_on: list[tuple[int, int]] = []
    pT_jet_vec_on: list[np.ndarray] = []
    pT_hadron_vec_on: list[np.ndarray] = []
    debug_on: list[dict] = []
    phi_jet_on: list[float] = []

    print("Running jet–hadron transverse observables (ISR/FSR OFF)...")
    run_observables_for_label(
        PTREL_LABEL_OFF,
        max_events=DEBUG_MAX_EVENTS,
        out_angle=angle_off,
        out_sum_mag=sum_mag_off,
        out_diff_mag=diff_mag_off,
        out_event_ids=event_ids_off,
        out_pT_jet_vec=pT_jet_vec_off,
        out_pT_hadron_vec=pT_hadron_vec_off,
        out_debug=debug_off,
        out_phi_jet=phi_jet_off,
    )
    print("Running jet–hadron transverse observables (ISR/FSR ON)...")
    run_observables_for_label(
        PTREL_LABEL_ON,
        max_events=DEBUG_MAX_EVENTS,
        out_angle=angle_on,
        out_sum_mag=sum_mag_on,
        out_diff_mag=diff_mag_on,
        out_event_ids=event_ids_on,
        out_pT_jet_vec=pT_jet_vec_on,
        out_pT_hadron_vec=pT_hadron_vec_on,
        out_debug=debug_on,
        out_phi_jet=phi_jet_on,
    )

    n_off = len(angle_off)
    n_on = len(angle_on)
    print(f"\nProcessed events: ISRFSR_OFF={n_off}, ISRFSR_ON={n_on}")

    # Convert to arrays (angle can contain nans)
    angle_off_arr = np.asarray(angle_off, dtype=np.float64)
    sum_mag_off_arr = np.asarray(sum_mag_off, dtype=np.float64)
    diff_mag_off_arr = np.asarray(diff_mag_off, dtype=np.float64)
    angle_on_arr = np.asarray(angle_on, dtype=np.float64)
    sum_mag_on_arr = np.asarray(sum_mag_on, dtype=np.float64)
    diff_mag_on_arr = np.asarray(diff_mag_on, dtype=np.float64)

    # Save per-event arrays for later inspection
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_angle_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_OFF.npy", angle_off_arr)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_sum_mag_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_OFF.npy", sum_mag_off_arr)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_diff_mag_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_OFF.npy", diff_mag_off_arr)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_angle_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_ON.npy", angle_on_arr)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_sum_mag_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_ON.npy", sum_mag_on_arr)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_diff_mag_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_ON.npy", diff_mag_on_arr)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_event_ids_ISRFSR_OFF.npy", np.asarray(event_ids_off, dtype=np.int64))
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_event_ids_ISRFSR_ON.npy", np.asarray(event_ids_on, dtype=np.int64))
    # Save pT vectors as (N,2) arrays
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_pT_jet_vec_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_OFF.npy", np.array(pT_jet_vec_off))
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_pT_hadron_vec_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_OFF.npy", np.array(pT_hadron_vec_off))
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_pT_jet_vec_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_ON.npy", np.array(pT_jet_vec_on))
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_pT_hadron_vec_{HADRON_TAG}_{FRAME_TAG}_ISRFSR_ON.npy", np.array(pT_hadron_vec_on))
    # Save phi_jet (jet azimuthal angle) arrays
    phi_jet_off_arr = np.asarray(phi_jet_off, dtype=np.float64)
    phi_jet_on_arr = np.asarray(phi_jet_on, dtype=np.float64)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_phi_jet_{FRAME_TAG}_ISRFSR_OFF.npy", phi_jet_off_arr)
    np.save(_JH_OUT / f"{OUTPUT_PREFIX}_phi_jet_{FRAME_TAG}_ISRFSR_ON.npy", phi_jet_on_arr)

    # Save first few events' debug info (event_id, pT_jet, pT_hadron, angle, sum_mag, diff_mag) for inspection
    def _pack_debug(debug_list: list[dict]) -> np.ndarray:
        rows = []
        for r in debug_list[:100]:
            angle_val = r["angle"] if r["angle"] is not None else np.nan
            rows.append([
                r["event_id"][0], r["event_id"][1],
                r["pT_jet_vec"][0], r["pT_jet_vec"][1],
                r["hadron_pT_vec"][0], r["hadron_pT_vec"][1],
                angle_val, r["sum_mag"], r["diff_mag"],
            ])
        return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 9))

    if debug_off:
        np.save(_JH_OUT / f"{OUTPUT_PREFIX}_debug_sample_ISRFSR_OFF.npy", _pack_debug(debug_off))
    if debug_on:
        np.save(_JH_OUT / f"{OUTPUT_PREFIX}_debug_sample_ISRFSR_ON.npy", _pack_debug(debug_on))
    print("Saved per-event arrays and debug samples.")

    # Histograms: normalized as (1/N) dN/dx so integral over bins * bin_width = 1
    angle_valid_off = angle_off_arr[np.isfinite(angle_off_arr)]
    angle_valid_on = angle_on_arr[np.isfinite(angle_on_arr)]
    n_angle_off = len(angle_valid_off)
    n_angle_on = len(angle_valid_on)

    # Plot 1: angle (convention [0, pi])
    fig1, ax1 = plt.subplots()
    bins_angle = np.linspace(ANGLE_RANGE[0], ANGLE_RANGE[1], ANGLE_BINS + 1)
    hist_off, _ = np.histogram(angle_valid_off, bins=bins_angle)
    hist_on, _ = np.histogram(angle_valid_on, bins=bins_angle)
    binw_angle = (ANGLE_RANGE[1] - ANGLE_RANGE[0]) / ANGLE_BINS
    centers_angle = 0.5 * (bins_angle[:-1] + bins_angle[1:])
    density_off = hist_off / (n_angle_off * binw_angle) if n_angle_off > 0 else hist_off.astype(float)
    density_on = hist_on / (n_angle_on * binw_angle) if n_angle_on > 0 else hist_on.astype(float)
    ax1.step(centers_angle, density_off, where="mid", label="ISR/FSR Off", color="blue", linewidth=1.5)
    ax1.step(centers_angle, density_on, where="mid", label="ISR/FSR On", color="red", linewidth=1.5)
    ax1.set_xlabel(r"$\phi_{hJ}$ [rad]")
    ax1.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$")
    ax1.legend(loc="best", frameon=False)
    ax1.tick_params(direction="in")
    plt.tight_layout()
    out_angle_pdf = _JH_OUT / f"{OUTPUT_PREFIX}_angle_{HADRON_TAG}_{FRAME_TAG}_comparison.pdf"
    plt.savefig(out_angle_pdf, format="pdf")
    plt.close(fig1)
    print(f"Saved: {out_angle_pdf.name}")

    # Plot 2: |pT_sum|
    fig2, ax2 = plt.subplots()
    bins_sd = np.linspace(SUM_DIFF_RANGE[0], SUM_DIFF_RANGE[1], SUM_DIFF_BINS + 1)
    binw_sd = (SUM_DIFF_RANGE[1] - SUM_DIFF_RANGE[0]) / SUM_DIFF_BINS
    centers_sd = 0.5 * (bins_sd[:-1] + bins_sd[1:])
    hist_sum_off, _ = np.histogram(sum_mag_off_arr, bins=bins_sd)
    hist_sum_on, _ = np.histogram(sum_mag_on_arr, bins=bins_sd)
    dens_sum_off = hist_sum_off / (n_off * binw_sd) if n_off > 0 else hist_sum_off.astype(float)
    dens_sum_on = hist_sum_on / (n_on * binw_sd) if n_on > 0 else hist_sum_on.astype(float)
    ax2.step(centers_sd, dens_sum_off, where="mid", label="ISR/FSR Off", color="blue", linewidth=1.5)
    ax2.step(centers_sd, dens_sum_on, where="mid", label="ISR/FSR On", color="red", linewidth=1.5)
    ax2.set_xlabel(r"$\bar{P}_{hJ}$ [GeV]")
    ax2.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\bar{P}_{hJ}}$")
    ax2.legend(loc="best", frameon=False)
    ax2.tick_params(direction="in")
    plt.tight_layout()
    out_sum_pdf = _JH_OUT / f"{OUTPUT_PREFIX}_sum_mag_{HADRON_TAG}_{FRAME_TAG}_comparison.pdf"
    plt.savefig(out_sum_pdf, format="pdf")
    plt.close(fig2)
    print(f"Saved: {out_sum_pdf.name}")

    # Plot 3: |pT_diff|
    fig3, ax3 = plt.subplots()
    hist_diff_off, _ = np.histogram(diff_mag_off_arr, bins=bins_sd)
    hist_diff_on, _ = np.histogram(diff_mag_on_arr, bins=bins_sd)
    dens_diff_off = hist_diff_off / (n_off * binw_sd) if n_off > 0 else hist_diff_off.astype(float)
    dens_diff_on = hist_diff_on / (n_on * binw_sd) if n_on > 0 else hist_diff_on.astype(float)
    ax3.step(centers_sd, dens_diff_off, where="mid", label="ISR/FSR Off", color="blue", linewidth=1.5)
    ax3.step(centers_sd, dens_diff_on, where="mid", label="ISR/FSR On", color="red", linewidth=1.5)
    ax3.set_xlabel(r"$\Delta P_{hJ}$ [GeV]")
    ax3.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\Delta P_{hJ}}$")
    ax3.legend(loc="best", frameon=False)
    ax3.tick_params(direction="in")
    plt.tight_layout()
    out_diff_pdf = _JH_OUT / f"{OUTPUT_PREFIX}_diff_mag_{HADRON_TAG}_{FRAME_TAG}_comparison.pdf"
    plt.savefig(out_diff_pdf, format="pdf")
    plt.close(fig3)
    print(f"Saved: {out_diff_pdf.name}")

    # Plot 4: Jet azimuth diagnostic (phi_jet), convention [-pi, pi)
    fig4, ax4 = plt.subplots()
    PHI_JET_BINS = 40
    phi_jet_range = (-np.pi, np.pi)
    bins_phi = np.linspace(phi_jet_range[0], phi_jet_range[1], PHI_JET_BINS + 1)
    binw_phi = (phi_jet_range[1] - phi_jet_range[0]) / PHI_JET_BINS
    centers_phi = 0.5 * (bins_phi[:-1] + bins_phi[1:])
    n_phi_off = len(phi_jet_off_arr)
    n_phi_on = len(phi_jet_on_arr)
    hist_phi_off, _ = np.histogram(phi_jet_off_arr, bins=bins_phi)
    hist_phi_on, _ = np.histogram(phi_jet_on_arr, bins=bins_phi)
    dens_phi_off = hist_phi_off / (n_phi_off * binw_phi) if n_phi_off > 0 else hist_phi_off.astype(float)
    dens_phi_on = hist_phi_on / (n_phi_on * binw_phi) if n_phi_on > 0 else hist_phi_on.astype(float)
    ax4.step(centers_phi, dens_phi_off, where="mid", label="ISR/FSR Off", color="blue", linewidth=1.5)
    ax4.step(centers_phi, dens_phi_on, where="mid", label="ISR/FSR On", color="red", linewidth=1.5)
    ax4.set_xlabel(r"$\phi_{\mathrm{jet}}$ [rad]")
    ax4.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{\mathrm{jet}}}$")
    ax4.set_xlim(phi_jet_range)
    ax4.legend(loc="best", frameon=False)
    ax4.tick_params(direction="in")
    plt.tight_layout()
    out_phi_jet_pdf = _JH_OUT / f"{OUTPUT_PREFIX}_phi_jet_{FRAME_TAG}_comparison.pdf"
    plt.savefig(out_phi_jet_pdf, format="pdf")
    plt.close(fig4)
    print(f"Saved: {out_phi_jet_pdf.name}")

    # Sanity check: confirm the fix works
    print("\n" + "=" * 70)
    print("SANITY CHECK: Outgoing-quark jet (k_out) in Breit")
    print("=" * 70)
    mean_pT_jet_off = np.mean([np.linalg.norm(v) for v in pT_jet_vec_off]) if pT_jet_vec_off else 0
    mean_pT_jet_on = np.mean([np.linalg.norm(v) for v in pT_jet_vec_on]) if pT_jet_vec_on else 0
    print(f"  mean |pT_jet| (from k_out) ISR/FSR OFF: {mean_pT_jet_off:.6f} GeV")
    print(f"  mean |pT_jet| (from k_out) ISR/FSR ON:  {mean_pT_jet_on:.6f} GeV")
    if mean_pT_jet_off < 1e-8 and mean_pT_jet_on < 1e-8:
        print("  -> WARNING: |pT_jet| ~ 0 for all events; check k_out or frame.")
    else:
        print("  -> Non-zero |pT_jet| expected for outgoing quark in Breit.")
    print("=" * 70)
    
    # Metadata summary
    meta_path = _JH_OUT / f"{OUTPUT_PREFIX}_metadata.txt"
    with open(meta_path, "w") as f:
        f.write("Jet–hadron transverse observables (outgoing quark jet)\n")
        f.write("======================================================\n\n")
        f.write("Jet definition:\n")
        f.write("  P_jet^{lab} = k_out (outgoing struck quark from the event record).\n")
        f.write("  pT_jet is (P_{jet,x}, P_{jet,y}) in the Breit frame after the same boost as the hadron.\n\n")
        f.write("Remnant / x_L:\n")
        f.write("  k_in = k_out - q, P_rem = P - k_in (unchanged convention for fragmentation variable).\n\n")
        f.write(f"Hadron: {HADRON_TAG}\n")
        f.write(f"Frame: {FRAME_TAG}\n")
        f.write(f"Angle convention: {ANGLE_CONVENTION} (0 = aligned, pi = back-to-back)\n\n")
        f.write("Observable 1: angle between pT(k_out) and pT(hadron) [rad].\n")
        f.write("Observable 2: |pT_jet + pT_h| [GeV] (vector sum, then magnitude).\n")
        f.write("Observable 3: |pT_jet - pT_h| [GeV] (vector difference, then magnitude).\n")
    print(f"Saved: {meta_path.name}")
    
    # Print output filenames
    print("\nOutput PDF files (jet = k_out):")
    print(f"  - {out_angle_pdf.name}")
    print(f"  - {out_sum_pdf.name}")
    print(f"  - {out_diff_pdf.name}")
    print(f"  - {out_phi_jet_pdf.name}")
    print("\nDone. Jet transverse observables use the outgoing quark k_out in the Breit frame.")


if __name__ == "__main__":
    main()
