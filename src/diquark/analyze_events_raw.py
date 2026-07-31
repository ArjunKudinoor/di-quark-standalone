#!/usr/bin/env python3
"""
Analyze raw PYTHIA shards (no PYTHIA). Reproduces:
  1. eta_hadron_<e>x<p>_xQ_regions_3x2.pdf for beams (5x41), (9x41), (9x100), (9x275)
  2. eta_hadron_EIC_hardware_QCD_regions.pdf (legacy single-panel, ETA_ON_CRON shards)
  3. eta_hadron_xQ_regions_summary.{json,csv}
  4. pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf
Uses same selection, binning, and fitting as generate_pdf_plots_new.py where applicable.
"""
import hashlib
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from pathlib import Path
from typing import Optional

from diquark.paths import analysis_outputs_dir, pythia_finalstate_raw_dir

# ---------------------------------------------------------------------------
# Plotting style (verbatim from generate_pdf_plots_new.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
})

# Labels for raw shards
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = pythia_finalstate_raw_dir()
_ANALYSIS_DIR = analysis_outputs_dir()
_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Output filenames (under DIQUARK_DATA_ROOT/.../outputs/analysis)
ETA_PDF = _ANALYSIS_DIR / "eta_hadron_EIC_hardware_QCD_regions.pdf"
PTREL_PDF = _ANALYSIS_DIR / "pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf"
ETA_LABEL = "ETA_ON_CRON"
PTREL_LABEL_OFF = "ISRFSR_OFF"
PTREL_LABEL_ON = "ISRFSR_ON"

# Binning (same as original)
ETA_BINS = 50
ETA_RANGE = (2.0, 8.0)
# Pseudorapidity intervals shaded as “EIC coverage” on η_h plots (lab-frame η).
EIC_REGIONS = [(2.0, 3.5), (4.5, 5.9), (6.0, 8.0)]
PTREL_BINS = 60
PTREL_RANGE = (0.01, 2.0)
pTmax_fit = 1.0
eps = 1e-12

# Frame flip: ETA_ON_CRON uses idA=2212,idB=11 (same as generate_pdf_plots_new), no flip.
# ISRFSR_ON/OFF use idA=11,idB=2212; flip pz for pTrel to match reference frame.
FLIP_Z_ETA = False
FLIP_Z_PTREL = True

# Debug pTrel mismatch (old vs new): per-event prints, raw histograms, single-event dump.
# Set to False to disable and restore normal run.
# Interpretation:
#   - If per-event pT vectors differ (vs reference) -> upstream physics bug
#   - If vectors identical but histogram differs -> binning/normalization bug
#   - If pT_rem + pT_jet != ~0 -> LT or remnant/jet definition bug
#   - If flip fixes it -> z-axis sign bug
# OLD vs NEW isolation: USED EVENT HASH differs -> selection/cuts; hashes match but
#   per-event hashes differ -> upstream/LT bug; all match but plots differ -> plot/fit/norm bug.
DEBUG_PTREL = True
DEBUG_EVENT_LIST = {0, 1, 2, 3, 4, 5, 10, 25, 49, 1234}
# Set True only for quick interactive stop when debugging; False for diff runs.
DEBUG_STOP_AT_1234 = False

# Cuts
Qmin_eta, Qmax_eta = 5.0, 15.0
xmin_eta, xmax_eta = 1e-2, 0.5
Qmin_ptrel, Qmax_ptrel = 5.0, 15.0
xmin_ptrel, xmax_ptrel = 1e-3, 0.5


# -----------------------------
# Shard loader
# -----------------------------
def list_shards(label: str):
    """Return sorted list of shard directories for label."""
    d = DATA_ROOT / label
    if not d.exists():
        return []
    subs = [x for x in d.iterdir() if x.is_dir() and x.name.startswith("shard_")]
    subs.sort(key=lambda x: x.name)
    return subs


def flip_z(v, do_flip: bool):
    """Negate pz to convert idA=11,idB=2212 frame to idA=2212,idB=11 frame."""
    if not do_flip:
        return v
    out = np.asarray(v, dtype=float).copy()
    if out.ndim == 1:
        out[3] = -out[3]
    else:
        out[..., 3] = -out[..., 3]
    return out


def load_shard(shard_dir: Path):
    """Load one shard; return dict with event_* (Ne,4), offsets (Ne+1), pid (Np), p4 (Np,4)."""
    return {
        "event_e_in": np.load(shard_dir / "event_e_in.npy"),
        "event_p_in": np.load(shard_dir / "event_p_in.npy"),
        "event_e_sc": np.load(shard_dir / "event_e_sc.npy"),
        "event_k_out": np.load(shard_dir / "event_k_out.npy"),
        "offsets": np.load(shard_dir / "offsets.npy"),
        "pid": np.load(shard_dir / "pid.npy"),
        "p4": np.load(shard_dir / "p4.npy"),
    }


# -----------------------------
# Physics helpers (verbatim from generate_pdf_plots_new.py)
# -----------------------------
def is_hadron(pid):
    return abs(pid) >= 100


def p3(v):
    return np.array([v[1], v[2], v[3]], dtype=float)


def pT_rel_to_axis(p_vec, axis_vec):
    na = np.linalg.norm(axis_vec)
    if na <= 0:
        return None
    n = axis_vec / na
    p_par = np.dot(p_vec, n) * n
    p_perp = p_vec - p_par
    pTrel2 = float(np.dot(p_perp, p_perp))
    pTrel2 = max(pTrel2, 0.0)
    return float(np.sqrt(pTrel2))


def dot4(a, b):
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]


def build_LT(Ee, Ep, qmu, x, y, qT, phiq, S):
    """Build Breit-like transform LT = M4 @ M3 @ M2 @ M1 @ M0 @ Mm1. qmu (E,px,py,pz)."""
    Mm1 = np.array([
        [Ee/np.sqrt(S) + np.sqrt(S)/(4.0*Ee), 0, 0, Ee/np.sqrt(S) - np.sqrt(S)/(4.0*Ee)],
        [0, 1, 0, 0], [0, 0, 1, 0],
        [Ee/np.sqrt(S) - np.sqrt(S)/(4.0*Ee), 0, 0, Ee/np.sqrt(S) + np.sqrt(S)/(4.0*Ee)]
    ])
    M0 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(phiq), np.sin(phiq), 0],
        [0, -np.sin(phiq), np.cos(phiq), 0], [0, 0, 0, 1]
    ])
    den2 = (-qT*qT + S*(1+x)*y)
    if den2 <= 0:
        return None
    denom_M1 = 2.0 * y * np.sqrt(S * (-qT*qT + S*(1+x)*y))
    if denom_M1 == 0:
        return None
    M1 = np.array([
        [(-qT*qT + S*y*(1+x+y)) / denom_M1, 0, 0, (qT*qT + S*y*(-x+y-1)) / denom_M1],
        [0, 1, 0, 0], [0, 0, 1, 0],
        [(qT*qT + S*y*(-x+y-1)) / denom_M1, 0, 0, (-qT*qT + S*y*(1+x+y)) / denom_M1]
    ])
    denom_M2_s1 = np.sqrt(S*(1+x)*y / den2)
    denom_M2_s2 = np.sqrt(S*(1+x)*y)
    M2 = np.array([
        [1, 0, 0, 0],
        [0, 1/denom_M2_s1, 0, qT/denom_M2_s2],
        [0, 0, 1, 0], [0, -qT/denom_M2_s2, 0, 1/denom_M2_s1]
    ])
    num_log = qT + np.sqrt(S*(1+x)*y)
    den_log = np.sqrt(S*(1+x)*y) - qT
    if num_log <= 0 or den_log <= 0:
        return None
    eta_m3 = 0.5*np.log(num_log/den_log)
    denom_M3 = np.sqrt(den2)
    M3 = np.array([
        [np.cosh(eta_m3), -qT/denom_M3, 0, 0],
        [-qT/denom_M3, np.cosh(eta_m3), 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1]
    ])
    denom_M4 = 2*np.sqrt(x*(1+x))
    if denom_M4 == 0:
        return None
    M4 = np.array([
        [(1+2*x)/denom_M4, 0, 0, 1/denom_M4],
        [0, 1, 0, 0], [0, 0, 1, 0],
        [1/denom_M4, 0, 0, (1+2*x)/denom_M4]
    ])
    LT = M4 @ M3 @ M2 @ M1 @ M0 @ Mm1
    return LT


def compute_x_Q_eta_one_event(e_in_ev, p_in_ev, e_sc_ev, offsets, pid, p4_arr, ie, flip_z_eta: bool):
    """
    Inclusive DIS: Q = sqrt(Q²) in GeV from virtual photon four-momentum (not sqrt of PDG Q² notation mix-up).
    Returns (x, Q, eta_h_lab) after Breit-like frame selection of leading forward hadron, or None.
    """
    e_in_ev = flip_z(np.asarray(e_in_ev, dtype=float), flip_z_eta)
    p_in_ev = flip_z(np.asarray(p_in_ev, dtype=float), flip_z_eta)
    e_sc_ev = flip_z(np.asarray(e_sc_ev, dtype=float), flip_z_eta)
    Ep = float(p_in_ev[0])
    Ee = float(e_in_ev[0])
    q0 = e_in_ev[0] - e_sc_ev[0]
    q1 = e_in_ev[1] - e_sc_ev[1]
    q2 = e_in_ev[2] - e_sc_ev[2]
    q3 = e_in_ev[3] - e_sc_ev[3]
    qmu = np.array([q0, q1, q2, q3])
    Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
    if Q2 <= 0:
        return None
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(q1, q2))
    p_dot_q = p_in_ev[0] * q0 - p_in_ev[1] * q1 - p_in_ev[2] * q2 - p_in_ev[3] * q3
    if p_dot_q == 0:
        return None
    x = Q2 / (2.0 * p_dot_q)
    phiq = float(np.arctan2(q2, q1))
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
    if LT is None:
        return None

    start = int(offsets[ie])
    end = int(offsets[ie + 1])
    best_E = -1.0
    best_lab = None
    best_pid = 0
    for j in range(start, end):
        this_pid = int(pid[j])
        if not is_hadron(this_pid):
            continue
        lab = flip_z(np.asarray(p4_arr[j], dtype=float), flip_z_eta)
        trf = LT @ lab
        E_, px_, py_, pz_ = trf
        if pz_ <= 0:
            continue
        if E_ > best_E:
            best_E = E_
            best_lab = lab
            best_pid = this_pid
    if best_lab is None:
        return None
    # Paper requirement (Peng/Rajagopal/Terry, p. 3): leading target hadron must be a charged pion.
    if best_pid not in (211, -211):
        return None
    _E_lab, px_lab, py_lab, pz_lab = best_lab
    p_mag = np.sqrt(px_lab * px_lab + py_lab * py_lab + pz_lab * pz_lab)
    if p_mag <= 0:
        return None
    den = max(p_mag - pz_lab, 1e-12)
    num = max(p_mag + pz_lab, 1e-12)
    eta = float(0.5 * np.log(num / den))
    return float(x), float(Q), float(eta)


def classify_x_bin_index(x):
    """Sea: 0.001 < x < 0.05; Valence: x > 0.05."""
    if 0.001 < x < 0.05:
        return 0
    if x > 0.05:
        return 1
    return None


def classify_Q_bin_index(Q):
    """Q in GeV; strict inequalities for bounded bins."""
    if 2.0 < Q < 5.0:
        return 0
    if 5.0 < Q < 10.0:
        return 1
    if Q > 10.0:
        return 2
    return None


# Beam configs for eta vs (x,Q) grids (must match LABEL_CONFIGS in generate_events_raw.py).
BEAM_CONFIGS_ETA_XQ = [(5, 41), (5, 100), (10, 100), (10, 275), (18, 275)]
ETA_XQ_LABEL_BY_BEAM = {
    (5, 41): "ETA_XQ_5x41",
    (5, 100): "ETA_XQ_5x100",
    (10, 100): "ETA_XQ_10x100",
    (10, 275): "ETA_XQ_10x275",
    (18, 275): "ETA_XQ_18x275",
}

X_REGION_SPECS = [
    ("Sea", 0.001, 0.05),
    ("Valence", 0.05, None),
]
Q_REGION_SPECS = [
    ("2 < Q < 5 GeV", 2.0, 5.0),
    ("5 < Q < 10 GeV", 5.0, 10.0),
    ("Q > 10 GeV", 10.0, None),
]

ETA_XQ_SUMMARY_JSON = _ANALYSIS_DIR / "eta_hadron_xQ_regions_summary.json"
ETA_XQ_SUMMARY_CSV = _ANALYSIS_DIR / "eta_hadron_xQ_regions_summary.csv"


def _beam_pdf_name(Ee: float, Ep: float) -> str:
    return _ANALYSIS_DIR / f"eta_hadron_{int(Ee)}x{int(Ep)}_xQ_regions_3x2.pdf"


def _verify_shard_beams(shard_dir: Path, Ee_nom: float, Ep_nom: float, tol: float = 0.02) -> None:
    mp = shard_dir / "meta.json"
    if not mp.exists():
        return
    meta = json.loads(mp.read_text())
    Ee = float(meta.get("E_e", -1))
    Ep = float(meta.get("E_p", -1))
    if abs(Ee - Ee_nom) > tol or abs(Ep - Ep_nom) > tol:
        raise RuntimeError(
            f"Shard {shard_dir} meta beams E_e={Ee}, E_p={Ep} do not match nominal "
            f"E_e={Ee_nom}, E_p={Ep_nom} (tol={tol}). Wrong label directory?"
        )


def make_eta_hadron_xQ_grid_for_beam(
    Ee_nom: float,
    Ep_nom: float,
    shard_label: str,
    out_pdf: Path,
    flip_z_eta: bool = False,
    max_events: Optional[int] = None,
):
    """
    One 3x2 figure: rows = Q bins, cols = x bins (sea | valence).
    Normalization per panel: (counts in eta bin) / (sum of counts in panel * bin width), ∫dη = 1.
    """
    shards = list_shards(shard_label)
    if not shards:
        print(f"[eta_xq] No shards for label {shard_label}. Generate with:\n"
              f"  python scripts/generation/generate_events_raw.py --labels {shard_label}")
        return None

    _verify_shard_beams(shards[0], Ee_nom, Ep_nom)

    edges = np.linspace(ETA_RANGE[0], ETA_RANGE[1], ETA_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = np.diff(edges)

    counts = np.zeros((3, 2, ETA_BINS), dtype=np.int64)
    n_events_cell = np.zeros((3, 2), dtype=np.int64)
    n_eta_in_window_cell = np.zeros((3, 2), dtype=np.int64)

    processed = 0
    for shard_idx, shard_path in enumerate(shards):
        if max_events is not None and processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        Ne = e_in.shape[0]

        for ie in range(Ne):
            if max_events is not None and processed >= max_events:
                break
            res = compute_x_Q_eta_one_event(
                e_in[ie], p_in[ie], e_sc[ie], offsets, pid, p4_arr, ie, flip_z_eta
            )
            if res is None:
                continue
            x, Q, eta = res
            ix = classify_x_bin_index(x)
            iq = classify_Q_bin_index(Q)
            if ix is None or iq is None:
                continue

            # Beam sanity (per-event lab energies must match nominal for this shard label)
            e_in_ev = flip_z(np.asarray(e_in[ie], dtype=float), flip_z_eta)
            p_in_ev = flip_z(np.asarray(p_in[ie], dtype=float), flip_z_eta)
            if abs(float(e_in_ev[0]) - Ee_nom) > 0.05 or abs(float(p_in_ev[0]) - Ep_nom) > 0.05:
                raise RuntimeError(
                    f"Event beam energies mismatch label {shard_label}: "
                    f"Ee={e_in_ev[0]}, Ep={p_in_ev[0]} vs nominal ({Ee_nom}, {Ep_nom})"
                )

            n_events_cell[iq, ix] += 1
            processed += 1
            if ETA_RANGE[0] <= eta < ETA_RANGE[1]:
                n_eta_in_window_cell[iq, ix] += 1
                bin_idx = np.searchsorted(edges, eta, side="right") - 1
                if 0 <= bin_idx < ETA_BINS:
                    counts[iq, ix, bin_idx] += 1

        print(f"[eta_xq] {shard_label} shard {shard_idx}: {processed} classified events so far")
        if max_events is not None and processed >= max_events:
            break

    # Shared y scale within figure
    ymax = 0.0
    densities = np.zeros_like(counts, dtype=float)
    for iq in range(3):
        for ix in range(2):
            tc = float(counts[iq, ix].sum())
            if tc > 0:
                densities[iq, ix] = counts[iq, ix] / (tc * binw)
                ymax = max(ymax, float(np.max(densities[iq, ix])))

    if ymax <= 0:
        ymax = 1.0

    fontsize = 12
    mpl.rcParams["font.size"] = fontsize
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 12.0), sharex=True)
    fig.suptitle(
        rf"EIC coverage, $e \times p = {int(Ee_nom)} \times {int(Ep_nom)}$ GeV",
        fontsize=fontsize + 2,
        y=0.995,
    )

    for iq in range(3):
        for ix in range(2):
            ax = axes[iq, ix]
            q_lbl, q_lo, q_hi = Q_REGION_SPECS[iq]
            x_lbl, x_lo, x_hi = X_REGION_SPECS[ix]
            if x_lbl == "Sea":
                x_part = r"Sea: $0.001 < x < 0.05$"
            else:
                x_part = r"Valence: $x > 0.05$"
            if q_hi is None:
                q_part = rf"$Q > 10$ GeV"
            else:
                q_part = rf"${q_lo} < Q < {q_hi}$ GeV"
            ax.set_title(f"{x_part}\n{q_part}", fontsize=fontsize - 1)

            n_ev = int(n_events_cell[iq, ix])
            n_eta_w = int(n_eta_in_window_cell[iq, ix])
            tc = counts[iq, ix].sum()

            if n_ev == 0:
                ax.text(
                    0.5,
                    0.55,
                    "No events in bin\n(selection + hadron chain)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=fontsize,
                )
                ax.text(
                    0.5,
                    0.22,
                    r"$N_{\mathrm{evt}}=0$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=fontsize - 1,
                )
            elif tc == 0:
                ax.text(
                    0.5,
                    0.55,
                    r"No $\eta_h$ in plot range "
                    rf"$[{ETA_RANGE[0]}, {ETA_RANGE[1]})$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=fontsize - 1,
                )
                ax.text(
                    0.5,
                    0.22,
                    rf"$N={n_ev}$ in bin, $\eta$ in range $=0$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=fontsize - 1,
                )
            else:
                ax.step(centers, densities[iq, ix], where="mid", color="k", linewidth=1.3)
                warn = ""
                if n_ev < 30:
                    warn = "\n(low statistics)"
                ax.text(
                    0.04,
                    0.92,
                    rf"$N={n_ev}$, $N_\eta={n_eta_w}$" + warn,
                    transform=ax.transAxes,
                    fontsize=fontsize - 2,
                    verticalalignment="top",
                    horizontalalignment="left",
                )

            for eta_lo, eta_hi in EIC_REGIONS:
                ax.axvspan(eta_lo, eta_hi, alpha=0.18, color="C0", zorder=0)

            ax.set_xlim(ETA_RANGE[0], ETA_RANGE[1])
            ax.set_ylim(0.0, ymax * 1.08)
            ax.set_xlabel(r"$\eta_{h}$", fontsize=fontsize - 1)
            if ix == 0:
                ax.set_ylabel(r"$\dfrac{1}{\sigma} \dfrac{d\sigma}{d\eta_h}$", fontsize=fontsize - 1)
            ax.grid(False)
            ax.tick_params(direction="in", labelsize=fontsize - 1)

    handles = [mpatches.Patch(color="C0", alpha=0.18, label=r"EIC coverage bands")]
    axes[0, 0].legend(handles=handles, frameon=False, loc="upper left", fontsize=fontsize - 2)

    plt.subplots_adjust(left=0.12, bottom=0.06, right=0.96, top=0.93, hspace=0.35, wspace=0.22)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"Wrote PDF (absolute): {out_pdf.resolve()}")

    summary_panels = []
    for iq in range(3):
        for ix in range(2):
            q_lbl, q_lo, q_hi = Q_REGION_SPECS[iq]
            x_lbl, x_lo, x_hi = X_REGION_SPECS[ix]
            tc = int(counts[iq, ix].sum())
            summary_panels.append(
                {
                    "shard_label": shard_label,
                    "E_e_GeV": float(Ee_nom),
                    "E_p_GeV": float(Ep_nom),
                    "x_region_label": x_lbl,
                    "x_min": x_lo,
                    "x_max": x_hi,
                    "Q_region_label": q_lbl,
                    "Q_min_GeV": q_lo,
                    "Q_max_GeV": q_hi,
                    "n_events_in_bin": int(n_events_cell[iq, ix]),
                    "n_events_eta_histogram": tc,
                    "n_events_eta_in_plot_range_before_hist": int(n_eta_in_window_cell[iq, ix]),
                    "density_normalization": "(hist counts in eta_bin j) / (sum hist counts * Δη_j); integrates to 1 over η",
                    "eta_plot_range": list(ETA_RANGE),
                    "eta_bins": int(ETA_BINS),
                }
            )
    return summary_panels


def run_eta_xq_grid_all_beams(max_events_per_beam: Optional[int] = None, flip_z_eta: bool = False):
    """Produce four PDFs + JSON + CSV summaries under analysis_outputs_dir()."""
    import csv

    all_summaries = []
    run_log: list[str] = []
    doc = {
        "Q_definition": "Q = sqrt(Q²) in GeV from four-momentum transfer q = k - k' (electron line)",
        "x_definition": "DIS Bjorken x from Q²/(2 P·q) using stored proton and electron momenta",
        "phase_space_note": (
            "ETA_XQ_* shards use PhaseSpace:Q2Min = 4 GeV² in generation so Q > 2 GeV is allowed; "
            "bins still apply strict cuts in Q."
        ),
        "beam_labels": [
            {"E_e_GeV": ek[0], "E_p_GeV": ek[1], "shard_label": lab}
            for ek, lab in ETA_XQ_LABEL_BY_BEAM.items()
        ],
        "panels": [],
    }

    for Ee_nom, Ep_nom in BEAM_CONFIGS_ETA_XQ:
        lbl = ETA_XQ_LABEL_BY_BEAM[(Ee_nom, Ep_nom)]
        outp = _beam_pdf_name(Ee_nom, Ep_nom)
        shard_dir = DATA_ROOT / lbl
        if not list_shards(lbl):
            msg = (
                f"Skipped label {lbl}: no shards found under {shard_dir.resolve()} "
                f"(generate with: python scripts/generation/generate_events_raw.py --labels {lbl})"
            )
            print(msg)
            run_log.append(msg)
            continue
        pan = make_eta_hadron_xQ_grid_for_beam(
            Ee_nom, Ep_nom, lbl, outp, flip_z_eta=flip_z_eta, max_events=max_events_per_beam
        )
        if pan is not None:
            all_summaries.extend(pan)
            ok = f"Generated ETA x-Q figure for {lbl} ({Ee_nom}x{Ep_nom} GeV) -> {outp.resolve()}"
            print(ok)
            run_log.append(ok)

    doc["panels"] = all_summaries
    ETA_XQ_SUMMARY_JSON.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"Saved summary (absolute): {ETA_XQ_SUMMARY_JSON.resolve()}")

    if all_summaries:
        keys = list(all_summaries[0].keys())
        with open(ETA_XQ_SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in all_summaries:
                w.writerow(row)
        print(f"Saved summary (absolute): {ETA_XQ_SUMMARY_CSV.resolve()}")

    print("\n=== eta x-Q grid run summary ===")
    for line in run_log:
        print(line)
    if not any("Generated ETA" in x for x in run_log):
        print("WARNING: No ETA x-Q PDFs were produced (all labels skipped or empty).")


# -----------------------------
# Eta analysis: stream shards, incremental histogram
# -----------------------------
def run_eta_analysis_and_plot(max_events=None):
    shards = list_shards(ETA_LABEL)
    if not shards:
        print(f"[eta] No shards found for {ETA_LABEL}. Run generate_events_raw.py first.")
        return

    edges = np.linspace(ETA_RANGE[0], ETA_RANGE[1], ETA_BINS + 1)
    total_counts = np.zeros(ETA_BINS, dtype=np.int64)
    x_list = []
    Q_list = []
    processed = 0

    for shard_idx, shard_path in enumerate(shards):
        if max_events is not None and processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        Ne = e_in.shape[0]

        for ie in range(Ne):
            if max_events is not None and processed >= max_events:
                break
            res = compute_x_Q_eta_one_event(
                e_in[ie], p_in[ie], e_sc[ie], offsets, pid, p4_arr, ie, FLIP_Z_ETA
            )
            if res is None:
                continue
            x, Q, eta = res
            if not (xmin_eta <= x <= xmax_eta) or not (Qmin_eta <= Q <= Qmax_eta):
                continue
            if ETA_RANGE[0] <= eta < ETA_RANGE[1]:
                bin_idx = np.searchsorted(edges, eta, side="right") - 1
                if 0 <= bin_idx < ETA_BINS:
                    total_counts[bin_idx] += 1
            x_list.append(x)
            Q_list.append(Q)
            processed += 1

        print(f"[eta] shard {shard_idx} ({shard_path.name}): {processed} events so far")
        if max_events is not None and processed >= max_events:
            break

    if processed == 0:
        print("[eta] No events passed cuts.")
        return
    avg_x = np.mean(x_list) if x_list else 0.0
    avg_Q = np.mean(Q_list) if Q_list else 0.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = np.diff(edges)
    n_in_range = total_counts.sum()
    density = total_counts / (float(n_in_range) * binw) if n_in_range > 0 else total_counts.astype(float)

    fontsize = 15
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    fig, ax = plt.subplots()
    ax.step(centers, density, where='mid', color='k', linewidth=1.5)
    for eta_lo, eta_hi in EIC_REGIONS:
        ax.axvspan(eta_lo, eta_hi, alpha=0.18, color='C0', zorder=0)
    ax.set_xlabel(r"$\eta_{h}$", fontsize=fontsize)
    ax.set_ylabel(r"$\dfrac{1}{\sigma} \dfrac{d\sigma}{d\eta_h}$", fontsize=fontsize)
    ax.grid(False)
    ax.tick_params(direction='in', labelsize=fontsize)
    handles = [mpatches.Patch(color='C0', alpha=0.18, label=r"EIC coverage")]
    ax.legend(handles=handles, frameon=False, loc='upper left', fontsize=fontsize)
    if x_list and Q_list:
        text_str = f"$\\langle x \\rangle = {avg_x:.4f}$\n$\\langle Q \\rangle = {avg_Q:.2f}$ GeV"
        ax.text(0.05, 0.87, text_str, transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', horizontalalignment='left')
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(ETA_PDF, format="pdf")
    plt.close(fig)
    print(f"Saved: {ETA_PDF} ({processed} events)")


# -----------------------------
# pTrel analysis: stream both configs, collect lists then fit/plot
# -----------------------------
def _stable_hash_array(x: np.ndarray, ndigits: int = 10) -> str:
    y = np.round(np.asarray(x, dtype=float), ndigits)
    return hashlib.md5(y.tobytes()).hexdigest()


def _stable_hash_ints(x: np.ndarray) -> str:
    return hashlib.md5(np.asarray(x, dtype=np.int64).tobytes()).hexdigest()


def _print_ptrel_dataset_info(label: str, total_events: int):
    """Print dataset info and config fingerprint for pTrel debug."""
    shard_base_path = DATA_ROOT / label
    print("=== DATASET INFO ===")
    print("Label:", label)
    print("Shard base path:", shard_base_path)
    print("Total events loaded:", total_events)
    fp_path = shard_base_path / "config_fingerprint.json"
    if fp_path.exists():
        print("Fingerprint contents:")
        print(json.dumps(json.load(fp_path.open()), indent=2))
    else:
        print("No fingerprint found.")
    print("====================")


def _run_ptrel_from_shards(label, max_events, out_pTrel, out_x, out_Q):
    shards = list_shards(label)
    if not shards:
        print(f"[pTrel] No shards for {label}.")
        return
    processed = 0
    used_event_ids = []
    used_event_triplets = []
    debug_records = []
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
            # Lorentz-invariant p·q
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

            # Remnant: k_in = k_out - q, P_rem = P_proton - k_in (lab frame)
            k_in = k_out_ev - qmu
            p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
            p_rem_truth_breit = boost(p_rem_truth)
            axis_rem = p3(p_rem_truth_breit)
            if np.linalg.norm(axis_rem) <= 0:
                continue
            pTrel_rem = pT_rel_to_axis(p3(best_tar_breit), axis_rem)
            if pTrel_rem is None:
                continue
            q_breit = boost(qmu)
            den = dot4(p_rem_truth_breit, q_breit)
            if den <= 0:
                continue
            xL_exact = dot4(best_tar_breit, q_breit) / den
            if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
                continue

            # Jet axis in Breit: P_jet_breit = P_proton_breit - P_remnant_breit (NOT q_breit)
            p_proton_breit = np.asarray(p_breit, dtype=float)
            P_jet_breit = p_proton_breit - p_rem_truth_breit
            pT_rem = axis_rem
            pT_jet = p3(P_jet_breit)
            pT_pi = p3(best_tar_breit)
            event_id = (int(shard_idx), int(ie))
            event_index = processed

            # 3-momenta in Breit (rem_trf = p_rem_truth_breit, pion_trf = best_tar_breit)
            p_prot = p3(p_breit)
            p_rem = p3(p_rem_truth_breit)
            p_jet = p3(P_jet_breit)
            p_pi = p3(best_tar_breit)
            diff3 = (p_rem + p_jet) - p_prot
            diffT = p_rem[:2] + p_jet[:2]

            if DEBUG_PTREL and (event_index < 50 or event_index in DEBUG_EVENT_LIST):
                print("==== EVENT", event_id, "(ordinal", event_index, ") ====")
                print("x,Q,y =", x, Q, y)
                print("p_prot =", p_prot, "hash", _stable_hash_array(p_prot))
                print("p_rem  =", p_rem, "hash", _stable_hash_array(p_rem))
                print("p_jet  =", p_jet, "hash", _stable_hash_array(p_jet))
                print("p_pi   =", p_pi, "hash", _stable_hash_array(p_pi))
                print("diff3(rem+jet-prot) =", diff3, "norm", np.linalg.norm(diff3))
                print("diffT(rem+jet)_xy    =", diffT, "norm", np.linalg.norm(diffT))
                print("pTrel_rem =", pTrel_rem)
                print("====================")

            # Breit 3-momentum conservation check
            if DEBUG_PTREL and np.linalg.norm(diff3) > 1e-3:
                raise RuntimeError(
                    "Breit 3-momentum conservation violated: (pT_rem + pT_jet) - p3(proton_breit) = %s"
                    % diff3
                )
            if DEBUG_PTREL and event_index == 1234:
                print("DEBUG EVENT 1234")
                print("Remnant lab:", p_rem_truth)
                print("Remnant Breit:", p_rem_truth_breit)
                print("Jet Breit (P_proton_breit - P_remnant_breit):", P_jet_breit)
                print("Done.")

            if DEBUG_PTREL and event_index in DEBUG_EVENT_LIST:
                # Row: shard_idx, local_idx, x, Q, y, pTrel_rem, p_pi_xyz, p_rem_xyz, p_jet_xyz (3-mom from 4-vec [1:4])
                debug_records.append([
                    float(event_id[0]), float(event_id[1]),
                    float(x), float(Q), float(y), float(pTrel_rem),
                    float(best_tar_breit[1]), float(best_tar_breit[2]), float(best_tar_breit[3]),
                    float(p_rem_truth_breit[1]), float(p_rem_truth_breit[2]), float(p_rem_truth_breit[3]),
                    float(P_jet_breit[1]), float(P_jet_breit[2]), float(P_jet_breit[3]),
                ])

            out_pTrel.append(pTrel_rem)
            out_x.append(x)
            out_Q.append(Q)
            used_event_ids.append(event_id)
            used_event_triplets.append((event_id[0], event_id[1], int(best_tar_pid)))
            processed += 1
            if DEBUG_PTREL and DEBUG_STOP_AT_1234 and event_index == 1234:
                print(f"[pTrel {label}] stopped after event 1234 for debug")
                break

        print(f"[pTrel {label}] shard {shard_idx} ({shard_path.name}): {len(out_pTrel)} events")
        if DEBUG_PTREL and DEBUG_STOP_AT_1234 and processed > 0 and event_index == 1234:
            break

    if DEBUG_PTREL and used_event_ids:
        used_event_ids = np.asarray(used_event_ids, dtype=np.int64)
        print(f"[{label}] USED EVENT COUNT =", used_event_ids.shape[0])
        print(f"[{label}] USED EVENT HASH  =", _stable_hash_ints(used_event_ids))
        np.save(_ANALYSIS_DIR / f"used_event_ids_{label}_NEW.npy", used_event_ids)
        if used_event_triplets:
            used_event_triplets = np.asarray(used_event_triplets, dtype=np.int64)
            print(f"[{label}] USED TRIPLETS HASH =", _stable_hash_ints(used_event_triplets.reshape(-1)))
            np.save(_ANALYSIS_DIR / f"used_event_triplets_{label}_NEW.npy", used_event_triplets)
        if debug_records:
            debug_records = np.array(debug_records, dtype=np.float64)
            np.save(_ANALYSIS_DIR / f"debug_records_{label}_NEW.npy", debug_records)
    return processed


def fit_gaussian_single_width(pT_list, label_suffix=""):
    """Same as generate_pdf_plots_new.py."""
    if len(pT_list) == 0:
        return None, None, None, None, None, None
    counts, edges = np.histogram(pT_list, bins=PTREL_BINS, range=PTREL_RANGE, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = edges[1:] - edges[:-1]
    y2d = counts / centers
    N = len(pT_list)
    n_i_est = np.maximum(N * counts * binw, 1.0)
    sigma_counts = np.sqrt(n_i_est) / (N * binw)
    sigma_y2d = sigma_counts / centers
    mask = (centers < pTmax_fit) & (y2d > 0) & np.isfinite(y2d) & np.isfinite(sigma_y2d)
    if np.sum(mask) < 3:
        return None, None, None, None, None, None
    X = centers[mask]**2
    Y = np.log(y2d[mask] + eps)
    sigma_lnY = sigma_y2d[mask] / (y2d[mask] + eps)
    w = 1.0 / np.maximum(sigma_lnY, 1e-6)**2
    Y_mean = np.average(Y, weights=w)
    Y_centered = Y - Y_mean
    X_mean = np.average(X, weights=w)
    X_centered = X - X_mean
    numerator = np.sum(w * X_centered * Y_centered)
    denominator = np.sum(w * X_centered**2)
    c1 = numerator / denominator if denominator > 0 else 0.0
    if c1 < -1e-8:
        B_fit = -1.0 / c1
    else:
        B_fit = 1.0
    B_fit = min(B_fit, 10.0)
    p_integrate = np.linspace(0.01, 2.0, 1000)
    integrand = p_integrate * np.exp(-(p_integrate**2) / B_fit)
    integral = np.trapz(integrand, p_integrate)
    A_fit = 1.0 / integral if integral > 0 else 1.0
    return centers, y2d, A_fit, B_fit, label_suffix, len(pT_list)


def run_ptrel_comparison_and_plot(max_events=None):
    pTrel_off = []
    x_off = []
    Q_off = []
    pTrel_on = []
    x_on = []
    Q_on = []

    print("Running pTrel from shards (ISR/FSR OFF)...")
    _run_ptrel_from_shards(PTREL_LABEL_OFF, max_events, pTrel_off, x_off, Q_off)
    if DEBUG_PTREL:
        _print_ptrel_dataset_info(PTREL_LABEL_OFF, len(pTrel_off))
    print("Running pTrel from shards (ISR/FSR ON)...")
    _run_ptrel_from_shards(PTREL_LABEL_ON, max_events, pTrel_on, x_on, Q_on)
    if DEBUG_PTREL:
        _print_ptrel_dataset_info(PTREL_LABEL_ON, len(pTrel_on))

    if DEBUG_PTREL and pTrel_off and pTrel_on:
        # Temporary: raw counts only (no density, no normalization)
        bins = np.linspace(0, 2.0, 41)
        hist_off, _ = np.histogram(pTrel_off, bins=bins)
        hist_on, _ = np.histogram(pTrel_on, bins=bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        fig_d, ax_d = plt.subplots()
        ax_d.step(bin_centers, hist_off, where="mid", label="ISR/FSR Off (raw)", color="blue")
        ax_d.step(bin_centers, hist_on, where="mid", label="ISR/FSR On (raw)", color="red")
        ax_d.set_xlabel(r"$P_{h\perp}$")
        ax_d.set_ylabel("Raw counts")
        ax_d.legend()
        ax_d.set_title("pTrel debug: raw counts only")
        plt.savefig(_ANALYSIS_DIR / "pTrel_debug_raw_counts.pdf", format="pdf")
        plt.close(fig_d)
        print("Saved: pTrel_debug_raw_counts.pdf (raw counts)")

    centers_off, y2d_off, A_off, B_off, _, n_off = fit_gaussian_single_width(pTrel_off, "OFF")
    centers_on, y2d_on, A_on, B_on, _, n_on = fit_gaussian_single_width(pTrel_on, "ON")

    if centers_off is None or centers_on is None:
        print("Could not fit one or both datasets.")
        return
    avg_x_off = np.mean(x_off) if x_off else 0.0
    avg_Q_off = np.mean(Q_off) if Q_off else 0.0
    avg_x_on = np.mean(x_on) if x_on else 0.0
    avg_Q_on = np.mean(Q_on) if Q_on else 0.0
    print("Gaussian fit results:")
    print(f"  ISR/FSR OFF: B = {B_off:.6f}, A = {A_off:.6f}, N = {n_off}")
    print(f"  ISR/FSR ON:  B = {B_on:.6f}, A = {A_on:.6f}, N = {n_on}")

    fontsize = 20.25
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(centers_off, y2d_off, where='mid', label='PYTHIA: ISR/FSR Off', linewidth=2, alpha=0.8, color='blue')
    ax.step(centers_on, y2d_on, where='mid', label='PYTHIA: ISR/FSR On', linewidth=2, alpha=0.8, color='red')
    p_grid = np.linspace(0.01, 2.0, 200)
    ax.plot(p_grid, A_off * np.exp(-(p_grid**2) / B_off), 'b--', alpha=0.7, linewidth=1.5, label='Fit a')
    ax.plot(p_grid, A_on * np.exp(-(p_grid**2) / B_on), 'r--', alpha=0.7, linewidth=1.5, label='Fit b')
    ax.set_xlabel(r"$P_{h\perp}$", fontsize=fontsize)
    ax.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{d\sigma}{d P_{h\perp}}$", fontsize=fontsize)
    ax.tick_params(direction='in', labelsize=fontsize)
    ax.legend(loc='best', frameon=False, fontsize=fontsize)
    if avg_x_off > 0 and avg_Q_off > 0:
        text_str = f"$\\langle x \\rangle = {avg_x_off:.4f}$, $\\langle Q \\rangle = {avg_Q_off:.2f}$ GeV"
        ax.text(0.98, 0.45, text_str, transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', horizontalalignment='right')
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(PTREL_PDF, format="pdf")
    plt.close(fig)
    print(f"Saved: {PTREL_PDF}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Analyzing raw shards -> eta x-Q grid PDFs (four beams)")
    run_eta_xq_grid_all_beams()
    print("Analyzing raw shards ->", ETA_PDF)
    run_eta_analysis_and_plot()
    print("Analyzing raw shards ->", PTREL_PDF)
    run_ptrel_comparison_and_plot()
    print("Done.")
