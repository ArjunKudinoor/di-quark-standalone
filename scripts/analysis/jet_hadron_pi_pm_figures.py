#!/usr/bin/env python3
"""
Step histograms for split π+/π− jet–hadron observables.

Uses the same binning and (1/σ) dσ/dx normalization style as
analyze_jet_hadron_transverse_observables.py. No PYTHIA dependency.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_jh():
    path = PROJECT_ROOT / "scripts" / "analysis" / "analyze_jet_hadron_transverse_observables.py"
    spec = importlib.util.spec_from_file_location("_jh_obs_mod_fig", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


JH = _load_jh()

# Fixed y-range for per-bin R_{pi-/pi+} ratio strips (below density panels).
PI_PM_RATIO_YLIM: Tuple[float, float] = (0.9, 1.1)
# Wider range for azimuthal decorrelation (phi_hJ) in split layout.
PI_PM_RATIO_YLIM_AZIMUTH: Tuple[float, float] = (0.8, 1.2)
def _phi_hj_combined_ylims_from_values(
    vp: np.ndarray,
    vm_arrays: Sequence[np.ndarray],
    bins: np.ndarray,
    binw: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Data-driven y-limits for combined φ_hJ panels from the curves actually plotted.

    Density: [0, max(1/σ dσ/dφ) × 1.08 + small pad]. Ratio: span of all R_{π⁻/π⁺} curves
    with margin ≥ 4% absolute or 12% of span (whichever is larger).
    """
    dens_hi = 0.0
    for v in (vp, *vm_arrays):
        if len(v) == 0:
            continue
        h, _ = np.histogram(v, bins=bins)
        dens_hi = max(dens_hi, float(np.max(h)) / (float(len(v)) * binw))
    density_ylim = (0.0, dens_hi * 1.08 + 0.02)

    ratio_chunks: List[np.ndarray] = []
    for vm in vm_arrays:
        _, ratio, _, valid = _ratio_series(vp, vm, bins)
        r = ratio[valid]
        if len(r):
            ratio_chunks.append(r)
    if ratio_chunks:
        all_r = np.concatenate(ratio_chunks)
        rmin, rmax = float(np.min(all_r)), float(np.max(all_r))
        span = rmax - rmin
        margin = max(0.04, 0.12 * span)
        ratio_ylim = (max(0.0, rmin - margin), rmax + margin)
    else:
        ratio_ylim = PI_PM_RATIO_YLIM_AZIMUTH
    return density_ylim, ratio_ylim


def _phi_hj_combined_log_ylims_from_values(
    vp: np.ndarray,
    vm_arrays: Sequence[np.ndarray],
    bins: np.ndarray,
    binw: float,
) -> Tuple[float, float]:
    """Positive y-limits for log-scaled (1/σ) dσ/dφ panels."""
    pos: List[float] = []
    for v in (vp, *vm_arrays):
        if len(v) == 0:
            continue
        h, _ = np.histogram(v, bins=bins)
        d = h / (float(len(v)) * binw)
        d = d[d > 0]
        if len(d):
            pos.extend(d.tolist())
    if not pos:
        return 1e-4, 1.0
    lo, hi = float(np.min(pos)), float(np.max(pos))
    return lo * 0.7, hi * 1.5


def _ratio_ylim_for_obs_key(obs_key: str) -> Tuple[float, float]:
    return PI_PM_RATIO_YLIM_AZIMUTH if obs_key == "phi_hJ" else PI_PM_RATIO_YLIM


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """CSV → list[dict] for plotting; normalize ok column and NaN."""
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        d: Dict[str, Any] = {}
        for k, v in row.items():
            if pd.isna(v):
                d[k] = None
            elif k == "ok" and isinstance(v, str):
                d[k] = v.strip().lower() in ("true", "1", "yes")
            else:
                d[k] = v
        records.append(d)
    return records


def _vals_for(
    records: List[Dict[str, Any]],
    sample: str,
    pion: str,
    key: str,
    *,
    finite_only: bool,
) -> np.ndarray:
    raw: List[float] = []
    for r in records:
        if r.get("sample") != sample or r.get("pion") != pion:
            continue
        if not r.get("ok"):
            continue
        if key not in r or r[key] is None:
            continue
        try:
            raw.append(float(r[key]))
        except (TypeError, ValueError):
            continue
    arr = np.asarray(raw, dtype=np.float64)
    if finite_only:
        arr = arr[np.isfinite(arr)]
    return arr


def _vals_for_combined(
    records: List[Dict[str, Any]],
    pion: str,
    key: str,
    *,
    finite_only: bool,
) -> np.ndarray:
    """All rows with ``ok`` and matching ``pion``, ignoring ``sample`` (pooled subsamples)."""
    raw: List[float] = []
    for r in records:
        if r.get("pion") != pion:
            continue
        if not r.get("ok"):
            continue
        if key not in r or r[key] is None:
            continue
        try:
            raw.append(float(r[key]))
        except (TypeError, ValueError):
            continue
    arr = np.asarray(raw, dtype=np.float64)
    if finite_only:
        arr = arr[np.isfinite(arr)]
    return arr


def _add_step_curve(
    ax: Any,
    values: np.ndarray,
    bins: np.ndarray,
    binw: float,
    label: str,
    color: str,
    *,
    omit_zero_bins: bool = False,
) -> None:
    if len(values) == 0:
        return
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(values, bins=bins)
    n = len(values)
    density = hist / (n * binw) if n > 0 else hist.astype(float)
    if omit_zero_bins:
        density = np.where(density > 0, density, np.nan)
    ax.step(centers, density, where="mid", label=label, color=color, linewidth=1.5)


def _add_pi_minus_over_pi_plus_ratio_panel(
    ax: Any,
    vp: np.ndarray,
    vm: np.ndarray,
    bins: np.ndarray,
    *,
    ylim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "best",
) -> None:
    """
    Plot ((1/sigma^- ) d sigma^- / dx) / ((1/sigma^+) d sigma^+ / dx) per bin.

    With histogram counts this is:
      (hm / n_m) / (hp / n_p)
    since the common bin-width factor cancels.
    """
    n_p, n_m = int(len(vp)), int(len(vm))
    hp, _ = np.histogram(vp, bins=bins)
    hm, _ = np.histogram(vm, bins=bins)
    hp = hp.astype(np.float64)
    hm = hm.astype(np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ratio = np.full_like(hp, np.nan, dtype=np.float64)
    err = np.full_like(hp, np.nan, dtype=np.float64)
    if n_p > 0 and n_m > 0:
        for i in range(len(hp)):
            Np, Nm = float(hp[i]), float(hm[i])
            if Np > 0 and Nm > 0:
                ratio[i] = (Nm / float(n_m)) / (Np / float(n_p))
                # Poisson-style uncertainty from bin counts (n_p, n_m treated as fixed sample sizes).
                err[i] = ratio[i] * float(np.sqrt(1.0 / Nm + 1.0 / Np))
    valid = np.isfinite(ratio) & (hp > 0) & (hm > 0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
    ratio_label = r"$R_{\pi^-/\pi^+}$"
    if np.any(valid):
        ax.errorbar(
            centers[valid],
            ratio[valid],
            yerr=err[valid],
            fmt="o",
            markersize=3,
            color="red",
            elinewidth=0.8,
            capsize=2,
            label=ratio_label,
        )
    else:
        ax.plot([], [], "o", color="red", markersize=3, label=ratio_label)
    ax.legend(loc=legend_loc, frameon=False, fontsize=8)
    ax.set_ylabel(r"$R_{\pi^-/\pi^+}$")
    lo, hi = ylim if ylim is not None else PI_PM_RATIO_YLIM
    ax.set_ylim(lo, hi)
    ax.tick_params(direction="in")


def _ratio_series(
    vp: np.ndarray,
    vm: np.ndarray,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centers, ratio, err, and validity mask for R_{pi-/pi+}."""
    n_p, n_m = int(len(vp)), int(len(vm))
    hp, _ = np.histogram(vp, bins=bins)
    hm, _ = np.histogram(vm, bins=bins)
    hp = hp.astype(np.float64)
    hm = hm.astype(np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ratio = np.full_like(hp, np.nan, dtype=np.float64)
    err = np.full_like(hp, np.nan, dtype=np.float64)
    if n_p > 0 and n_m > 0:
        for i in range(len(hp)):
            Np, Nm = float(hp[i]), float(hm[i])
            if Np > 0 and Nm > 0:
                ratio[i] = (Nm / float(n_m)) / (Np / float(n_p))
                err[i] = ratio[i] * float(np.sqrt(1.0 / Nm + 1.0 / Np))
    valid = np.isfinite(ratio) & (hp > 0) & (hm > 0)
    return centers, ratio, err, valid


def write_combined_pi_pm_angle_multi_signal_pdf(
    *,
    records_piplus: List[Dict[str, Any]],
    signal_records: Sequence[Tuple[str, str, List[Dict[str, Any]]]],
    figure_dir: Path,
    filename_suffix: str = "",
    title: str = "",
    caption: str = "",
    density_ylim: Optional[Tuple[float, float]] = None,
    ratio_ylim: Optional[Tuple[float, float]] = None,
    density_yscale: str = "linear",
) -> Path:
    """
    Angle-only combined layout:
    - one shared pi+ density curve
    - multiple pi- density curves (e.g. 5% and 10% signal)
    - one ratio strip with matching R_{pi-/pi+} curves
    """
    log_density = str(density_yscale).strip().lower() == "log"
    bins = np.linspace(JH.ANGLE_RANGE[0], JH.ANGLE_RANGE[1], JH.ANGLE_BINS + 1)
    binw = (JH.ANGLE_RANGE[1] - JH.ANGLE_RANGE[0]) / JH.ANGLE_BINS
    vp = _vals_for_combined(records_piplus, "piplus", "phi_hJ", finite_only=True)
    vm_arrays = [
        _vals_for_combined(recs, "piminus", "phi_hJ", finite_only=True)
        for _, _, recs in signal_records
    ]
    auto_dens, auto_ratio = _phi_hj_combined_ylims_from_values(vp, vm_arrays, bins, binw)
    if density_ylim is None:
        density_ylim = auto_dens
    if ratio_ylim is None:
        ratio_ylim = auto_ratio

    fig, ax_m = plt.subplots(1, 1, figsize=(6.0, 4.2))
    _add_step_curve(
        ax_m, vp, bins, binw, r"$\pi^+$", "blue", omit_zero_bins=log_density
    )
    for (label, color, recs), vm in zip(signal_records, vm_arrays):
        _add_step_curve(
            ax_m, vm, bins, binw, label, color, omit_zero_bins=log_density
        )

    ax_m.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$")
    if title and str(title).strip():
        ax_m.set_title(title)
    if caption and str(caption).strip():
        ax_m.text(
            0.03,
            0.97,
            str(caption).strip(),
            transform=ax_m.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="0.25",
            linespacing=1.35,
        )
    if log_density:
        ax_m.set_yscale("log")
        lo, hi = _phi_hj_combined_log_ylims_from_values(vp, vm_arrays, bins, binw)
        ax_m.set_ylim(lo, hi)
    elif density_ylim is not None:
        ax_m.set_ylim(float(density_ylim[0]), float(density_ylim[1]))
    ax_m.legend(loc="center left", bbox_to_anchor=(0.0, 0.38), frameon=False)
    ax_m.tick_params(direction="in")

    div = make_axes_locatable(ax_m)
    ax_r = div.append_axes("bottom", size="32%", pad=0.07, sharex=ax_m)
    ax_r.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
    for (ratio_label, color, recs), vm in zip(signal_records, vm_arrays):
        centers, ratio, err, valid = _ratio_series(vp, vm, bins)
        label = ratio_label.replace(r"$\pi^-$", r"$R_{\pi^-/\pi^+}$")
        if np.any(valid):
            ax_r.errorbar(
                centers[valid],
                ratio[valid],
                yerr=err[valid],
                fmt="o",
                markersize=3,
                color=color,
                elinewidth=0.8,
                capsize=2,
                label=label,
            )
        else:
            ax_r.plot([], [], "o", color=color, markersize=3, label=label)
    ax_r.legend(
        loc="upper left",
        ncol=max(1, len(signal_records)),
        frameon=False,
        fontsize=8,
    )
    ax_r.set_ylabel(r"$R_{\pi^-/\pi^+}$")
    ax_r.set_ylim(float(ratio_ylim[0]), float(ratio_ylim[1]))
    ax_r.tick_params(direction="in")

    ax_m.tick_params(labelbottom=False)
    ax_r.set_xlabel(r"$\phi_{hJ}$ [rad]")
    plt.tight_layout()
    suf = filename_suffix if filename_suffix else ""
    out = (
        figure_dir
        / f"{JH.OUTPUT_PREFIX}_angle_{JH.HADRON_TAG}_{JH.FRAME_TAG}_combined_pi_pm{suf}.pdf"
    )
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def write_combined_pi_pm_angle_triptych_pdfs(
    *,
    records_piplus: List[Dict[str, Any]],
    panel_records: Sequence[Tuple[str, str, str, List[Dict[str, Any]]]],
    figure_dir: Path,
    filename_suffix: str = "",
    caption: str = "",
    density_ylim: Optional[Tuple[float, float]] = None,
    ratio_ylim: Optional[Tuple[float, float]] = None,
    write_standalone_panels: bool = False,
) -> List[Path]:
    """
    Horizontal 1×3 layout: density panel + R_{π⁻/π⁺} strip per column.

    Each column: shared π⁺ reference + one π⁻ curve, ratio below.
    ``panel_records`` entries are ``(panel_title, pi_minus_label, color, records)``.
    """
    if len(panel_records) != 3:
        raise ValueError("triptych layout requires exactly three panels")

    bins = np.linspace(JH.ANGLE_RANGE[0], JH.ANGLE_RANGE[1], JH.ANGLE_BINS + 1)
    binw = (JH.ANGLE_RANGE[1] - JH.ANGLE_RANGE[0]) / JH.ANGLE_BINS
    vp = _vals_for_combined(records_piplus, "piplus", "phi_hJ", finite_only=True)
    vm_arrays = [
        _vals_for_combined(recs, "piminus", "phi_hJ", finite_only=True)
        for _, _, _, recs in panel_records
    ]
    auto_dens, auto_ratio = _phi_hj_combined_ylims_from_values(vp, vm_arrays, bins, binw)
    if density_ylim is None:
        density_ylim = auto_dens
    if ratio_ylim is None:
        ratio_ylim = auto_ratio

    ylabel = r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$"
    xlabel = r"$\phi_{hJ}$ [rad]"
    ratio_ylabel = r"$R_{\pi^-/\pi^+}$"
    suf = filename_suffix if filename_suffix else ""
    base = (
        f"{JH.OUTPUT_PREFIX}_angle_{JH.HADRON_TAG}_{JH.FRAME_TAG}_combined_pi_pm{suf}"
    )

    def _draw_density(ax: Any, panel_title: str, piminus_label: str, color: str, vm: np.ndarray) -> None:
        _add_step_curve(ax, vp, bins, binw, r"$\pi^{+}$", "blue")
        _add_step_curve(ax, vm, bins, binw, piminus_label, color)
        ax.set_title(panel_title)
        ax.set_xlim(float(JH.ANGLE_RANGE[0]), float(JH.ANGLE_RANGE[1]))
        ax.set_ylim(float(density_ylim[0]), float(density_ylim[1]))
        ax.legend(loc="best", frameon=False)
        ax.tick_params(direction="in")

    def _draw_ratio(ax_r: Any, vm: np.ndarray, color: str) -> None:
        centers, ratio, err, valid = _ratio_series(vp, vm, bins)
        ax_r.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
        if np.any(valid):
            ax_r.errorbar(
                centers[valid],
                ratio[valid],
                yerr=err[valid],
                fmt="o",
                markersize=3,
                color=color,
                elinewidth=0.8,
                capsize=2,
            )
        ax_r.set_ylim(float(ratio_ylim[0]), float(ratio_ylim[1]))
        ax_r.tick_params(direction="in")

    def _draw_column(
        ax_m: Any,
        panel_title: str,
        piminus_label: str,
        color: str,
        recs: List[Dict[str, Any]],
        *,
        show_density_ylabel: bool,
        show_ratio_ylabel: bool,
        show_xlabel: bool,
    ) -> Any:
        vm = _vals_for_combined(recs, "piminus", "phi_hJ", finite_only=True)
        _draw_density(ax_m, panel_title, piminus_label, color, vm)
        if show_density_ylabel:
            ax_m.set_ylabel(ylabel)
        ax_m.tick_params(labelbottom=False)
        div = make_axes_locatable(ax_m)
        ax_r = div.append_axes("bottom", size="32%", pad=0.07, sharex=ax_m)
        _draw_ratio(ax_r, vm, color)
        if show_ratio_ylabel:
            ax_r.set_ylabel(ratio_ylabel)
        if show_xlabel:
            ax_r.set_xlabel(xlabel)
        else:
            ax_r.tick_params(labelbottom=False)
        return ax_r

    written: List[Path] = []

    fig, axes_m = plt.subplots(1, 3, figsize=(18.0, 5.5), sharex=True, sharey=True)
    for ix, (panel_title, piminus_label, color, recs) in enumerate(panel_records):
        _draw_column(
            axes_m[ix],
            panel_title,
            piminus_label,
            color,
            recs,
            show_density_ylabel=(ix == 0),
            show_ratio_ylabel=(ix == 0),
            show_xlabel=True,
        )
    if caption and str(caption).strip():
        axes_m[0].text(
            0.03,
            0.97,
            str(caption).strip(),
            transform=axes_m[0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="0.25",
            linespacing=1.35,
        )
    plt.tight_layout()
    out = figure_dir / f"{base}_triptych.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    written.append(out)

    if write_standalone_panels:
        slug_map = {
            "Unmodified": "unmodified",
            "5% modified": "5pct_modified",
            "10% modified": "10pct_modified",
        }
        for panel_title, piminus_label, color, recs in panel_records:
            slug = slug_map.get(panel_title, panel_title.lower().replace(" ", "_").replace("%", "pct"))
            fig_s, ax_s = plt.subplots(1, 1, figsize=(6.0, 4.2))
            # Talk single-panel override: π⁻ is red for all three cases
            # (combined triptych keeps the grey/purple/red convention).
            _draw_column(
                ax_s,
                panel_title,
                piminus_label,
                "red",
                recs,
                show_density_ylabel=True,
                show_ratio_ylabel=True,
                show_xlabel=True,
            )
            if caption and str(caption).strip():
                ax_s.text(
                    0.03,
                    0.97,
                    str(caption).strip(),
                    transform=ax_s.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="0.25",
                    linespacing=1.35,
                )
            plt.tight_layout()
            out_s = figure_dir / f"{base}_triptych_panel_{slug}.pdf"
            fig_s.savefig(out_s, format="pdf")
            plt.close(fig_s)
            written.append(out_s)

    return written


def _write_one_pi_pm_figure_two_rows(
    *,
    records: List[Dict[str, Any]],
    samples: List[Tuple[str, str]],
    figure_dir: Path,
    file_key: str,
    bins: np.ndarray,
    binw: float,
    obs_key: str,
    finite_only: bool,
    ylabel_dens: str,
    xlabel: str,
    show_panel_titles: bool = True,
) -> Path:
    """
    Two stacked **panels** (altered / unchanged). Each panel = density on top + π⁻/π⁺ ratio
    strip below (``make_axes_locatable``), so the figure reads as two rows, not four siblings.
    """
    fig, ax_mains = plt.subplots(2, 1, figsize=(6.0, 9.0), sharex=True)
    for ix, (ax_m, (samp, stitle)) in enumerate(zip(ax_mains, samples)):
        vp = _vals_for(records, samp, "piplus", obs_key, finite_only=finite_only)
        vm = _vals_for(records, samp, "piminus", obs_key, finite_only=finite_only)
        _add_step_curve(ax_m, vp, bins, binw, r"$\pi^+$", "blue")
        _add_step_curve(ax_m, vm, bins, binw, r"$\pi^-$", "red")
        ax_m.set_ylabel(ylabel_dens)
        if stitle and show_panel_titles:
            ax_m.set_title(stitle)
        h, lab = ax_m.get_legend_handles_labels()
        if lab:
            ax_m.legend(loc="best", frameon=False)
        ax_m.tick_params(direction="in")
        div = make_axes_locatable(ax_m)
        ax_r = div.append_axes("bottom", size="32%", pad=0.07, sharex=ax_m)
        _add_pi_minus_over_pi_plus_ratio_panel(
            ax_r, vp, vm, bins, ylim=_ratio_ylim_for_obs_key(obs_key)
        )
        ax_m.tick_params(labelbottom=False)
        if ix < len(ax_mains) - 1:
            ax_r.tick_params(labelbottom=False)
        else:
            ax_r.set_xlabel(xlabel)
    plt.tight_layout()
    out = (
        figure_dir
        / f"{JH.OUTPUT_PREFIX}_{file_key}_{JH.HADRON_TAG}_{JH.FRAME_TAG}_split_pi_pm_comparison.pdf"
    )
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def _write_one_pi_pm_figure_combined(
    *,
    records: List[Dict[str, Any]],
    figure_dir: Path,
    file_key: str,
    bins: np.ndarray,
    binw: float,
    obs_key: str,
    finite_only: bool,
    ylabel_dens: str,
    xlabel: str,
    title: str,
    filename_suffix: str = "",
    caption: str = "",
    density_ylim: Optional[Tuple[float, float]] = None,
    ratio_ylim: Optional[Tuple[float, float]] = None,
) -> Path:
    """
    One observable: top = π⁺ and π⁻ (pooled over all ``sample`` values), bottom = R_{π⁻/π⁺}.
    """
    fig, ax_m = plt.subplots(1, 1, figsize=(6.0, 4.2))
    vp = _vals_for_combined(records, "piplus", obs_key, finite_only=finite_only)
    vm = _vals_for_combined(records, "piminus", obs_key, finite_only=finite_only)
    _add_step_curve(ax_m, vp, bins, binw, r"$\pi^+$", "blue")
    _add_step_curve(ax_m, vm, bins, binw, r"$\pi^-$", "red")
    ax_m.set_ylabel(ylabel_dens)
    if title and str(title).strip():
        ax_m.set_title(title)
    if caption and str(caption).strip():
        ax_m.text(
            0.03,
            0.97,
            str(caption).strip(),
            transform=ax_m.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="0.25",
            linespacing=1.35,
        )
    h, lab = ax_m.get_legend_handles_labels()
    if lab:
        ax_m.legend(loc="lower right", frameon=False)
    if density_ylim is not None:
        ax_m.set_ylim(float(density_ylim[0]), float(density_ylim[1]))
    ax_m.tick_params(direction="in")
    div = make_axes_locatable(ax_m)
    ax_r = div.append_axes("bottom", size="32%", pad=0.07, sharex=ax_m)
    rylim = ratio_ylim if ratio_ylim is not None else _ratio_ylim_for_obs_key(obs_key)
    _add_pi_minus_over_pi_plus_ratio_panel(
        ax_r, vp, vm, bins, ylim=rylim, legend_loc="upper left"
    )
    ax_m.tick_params(labelbottom=False)
    ax_r.set_xlabel(xlabel)
    plt.tight_layout()
    suf = filename_suffix if filename_suffix else ""
    out = (
        figure_dir
        / f"{JH.OUTPUT_PREFIX}_{file_key}_{JH.HADRON_TAG}_{JH.FRAME_TAG}_combined_pi_pm{suf}.pdf"
    )
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


_COMBINED_KEYS = ("angle", "sum_mag", "diff_mag", "Pbar_t")


def write_combined_pi_pm_comparison_pdfs(
    records: List[Dict[str, Any]],
    figure_dir: Path,
    *,
    title: str = r"$\pi^\pm$ (pooled over subsamples)",
    filename_suffix: str = "",
    include_combined_keys: Optional[Sequence[str]] = None,
    show_title: bool = True,
    caption: str = "",
) -> List[Path]:
    """
    Up to four PDFs (angle, sum, diff, P̄_t). Each file: one π⁺/π⁻ density panel + R_{π⁻/π⁺} below,
    pooling all CSV rows that pass the ``ok`` filter (any ``sample``).

    ``include_combined_keys``: subset of ``("angle", "sum_mag", "diff_mag", "Pbar_t")``;
    default writes all four.

    ``show_title``: when False, no main-axis title is drawn (split layout: use
    ``write_split_pi_pm_comparison_pdfs(..., show_panel_titles=False)``).
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    want = (
        {str(k).strip() for k in include_combined_keys if str(k).strip()}
        if include_combined_keys
        else set(_COMBINED_KEYS)
    )
    unknown = want - set(_COMBINED_KEYS)
    if unknown:
        raise ValueError(
            f"unknown include_combined_keys {sorted(unknown)}; "
            f"allowed: {list(_COMBINED_KEYS)}"
        )

    bins_a = np.linspace(JH.ANGLE_RANGE[0], JH.ANGLE_RANGE[1], JH.ANGLE_BINS + 1)
    binw_a = (JH.ANGLE_RANGE[1] - JH.ANGLE_RANGE[0]) / JH.ANGLE_BINS
    plot_title = title if show_title else ""
    if "angle" in want:
        vp_a = _vals_for_combined(records, "piplus", "phi_hJ", finite_only=True)
        vm_a = _vals_for_combined(records, "piminus", "phi_hJ", finite_only=True)
        dens_ylim, ratio_ylim = _phi_hj_combined_ylims_from_values(
            vp_a, (vm_a,), bins_a, binw_a
        )
        written.append(
            _write_one_pi_pm_figure_combined(
                records=records,
                figure_dir=figure_dir,
                file_key="angle",
                bins=bins_a,
                binw=binw_a,
                obs_key="phi_hJ",
                finite_only=True,
                ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$",
                xlabel=r"$\phi_{hJ}$ [rad]",
                title=plot_title,
                filename_suffix=filename_suffix,
                caption=caption,
                density_ylim=dens_ylim,
                ratio_ylim=ratio_ylim,
            )
        )

    bins_sd = np.linspace(JH.SUM_DIFF_RANGE[0], JH.SUM_DIFF_RANGE[1], JH.SUM_DIFF_BINS + 1)
    binw_sd = (JH.SUM_DIFF_RANGE[1] - JH.SUM_DIFF_RANGE[0]) / JH.SUM_DIFF_BINS

    if "sum_mag" in want:
        written.append(
            _write_one_pi_pm_figure_combined(
                records=records,
                figure_dir=figure_dir,
                file_key="sum_mag",
                bins=bins_sd,
                binw=binw_sd,
                obs_key="sum_mag_GeV",
                finite_only=False,
                ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\bar{P}_{hJ}}$",
                xlabel=r"$\bar{P}_{hJ}$ [GeV] ($|p_{T,\mathrm{jet}}+p_{T,h}|$)",
                title=plot_title,
                filename_suffix=filename_suffix,
                caption=caption,
            )
        )
    if "diff_mag" in want:
        written.append(
            _write_one_pi_pm_figure_combined(
                records=records,
                figure_dir=figure_dir,
                file_key="diff_mag",
                bins=bins_sd,
                binw=binw_sd,
                obs_key="delta_P_hJ_GeV",
                finite_only=False,
                ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\Delta P_{hJ}}$",
                xlabel=r"$\Delta P_{hJ}$ [GeV]",
                title=plot_title,
                filename_suffix=filename_suffix,
                caption=caption,
            )
        )
    if "Pbar_t" in want:
        written.append(
            _write_one_pi_pm_figure_combined(
                records=records,
                figure_dir=figure_dir,
                file_key="Pbar_t",
                bins=bins_sd,
                binw=binw_sd,
                obs_key="Pbar_t_GeV",
                finite_only=False,
                ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\bar P_t|}$",
                xlabel=r"$|\bar P_t|$ [GeV] ($\frac{1}{2}|p_{T,\mathrm{jet}}+p_{T,h}|$)",
                title=plot_title,
                filename_suffix=filename_suffix,
                caption=caption,
            )
        )

    return written


def write_split_pi_pm_comparison_pdfs(
    records: List[Dict[str, Any]],
    figure_dir: Path,
    *,
    show_panel_titles: bool = True,
) -> List[Path]:
    """
    Write comparison PDFs: φ_hJ, |pT_jet+pT_h|, |pT_jet−pT_h|, and |P̄_t| (half-sum magnitude).
    **Two** stacked panels (altered / unchanged). Each panel shows π⁺/π⁻ densities with a
    **N(π⁻)/N(π⁺)** ratio strip directly beneath it (not a separate third/fourth subplot row).

    Each event contributes at most one row; ``pion`` tags the leading charged pion’s sign.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Tuple[str, str]] = [
        ("altered", "Altered (split)"),
        ("unchanged", "Unchanged"),
    ]
    written: List[Path] = []

    bins_a = np.linspace(JH.ANGLE_RANGE[0], JH.ANGLE_RANGE[1], JH.ANGLE_BINS + 1)
    binw_a = (JH.ANGLE_RANGE[1] - JH.ANGLE_RANGE[0]) / JH.ANGLE_BINS
    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="angle",
            bins=bins_a,
            binw=binw_a,
            obs_key="phi_hJ",
            finite_only=True,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$",
            xlabel=r"$\phi_{hJ}$ [rad]",
            show_panel_titles=show_panel_titles,
        )
    )

    bins_sd = np.linspace(JH.SUM_DIFF_RANGE[0], JH.SUM_DIFF_RANGE[1], JH.SUM_DIFF_BINS + 1)
    binw_sd = (JH.SUM_DIFF_RANGE[1] - JH.SUM_DIFF_RANGE[0]) / JH.SUM_DIFF_BINS

    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="sum_mag",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="sum_mag_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\bar{P}_{hJ}}$",
            xlabel=r"$\bar{P}_{hJ}$ [GeV] ($|p_{T,\mathrm{jet}}+p_{T,h}|$)",
            show_panel_titles=show_panel_titles,
        )
    )
    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="diff_mag",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="delta_P_hJ_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\Delta P_{hJ}}$",
            xlabel=r"$\Delta P_{hJ}$ [GeV]",
            show_panel_titles=show_panel_titles,
        )
    )
    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="Pbar_t",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="Pbar_t_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\bar P_t|}$",
            xlabel=r"$|\bar P_t|$ [GeV] ($\frac{1}{2}|p_{T,\mathrm{jet}}+p_{T,h}|$)",
            show_panel_titles=show_panel_titles,
        )
    )

    return written
