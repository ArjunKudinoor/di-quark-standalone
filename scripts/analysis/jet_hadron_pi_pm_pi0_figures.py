#!/usr/bin/env python3
"""The original Fig. 3 two-panel layout, extended by one pion curve and ratio."""
from pathlib import Path
import numpy as np
from jet_hadron_pi_pm_figures import (
    JH, plt, make_axes_locatable, _add_step_curve, _ratio_series,
)

DENSITY_YLIM = (0.0, 1.1)
RATIO_YLIM = (0.8, 1.3)


def write_combined_pi_pm_pi0_angle_pdf(*, vp, vm, v0, fraction, output, caption='', piplus_fraction=None):
    """Plot the supplied distributions; label pi+ with its actual SRC fraction."""
    bins = np.linspace(JH.ANGLE_RANGE[0], JH.ANGLE_RANGE[1], JH.ANGLE_BINS + 1)
    binw = (JH.ANGLE_RANGE[1] - JH.ANGLE_RANGE[0]) / JH.ANGLE_BINS
    # Identical dimensions, step curves, labels, caption and axes-divider layout.
    fig, ax_m = plt.subplots(1, 1, figsize=(6.0, 4.2))
    piplus_fraction = fraction if piplus_fraction is None else piplus_fraction
    _add_step_curve(ax_m, vp, bins, binw, rf'$\pi^+~({int(round(piplus_fraction*100))}\%)$', 'blue')
    _add_step_curve(ax_m, vm, bins, binw, rf'$\pi^-~({int(round(fraction*100))}\%)$', 'red')
    _add_step_curve(ax_m, v0, bins, binw, rf'$\pi^0~({int(round(fraction*100))}\%)$', 'green')
    ax_m.set_ylabel(r'$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$')
    if caption and str(caption).strip():
        ax_m.text(.03, .97, str(caption).strip(), transform=ax_m.transAxes,
                  ha='left', va='top', fontsize=10, color='0.25', linespacing=1.35)
    ax_m.set_ylim(*DENSITY_YLIM)
    ax_m.legend(loc='center left', bbox_to_anchor=(0., .38), frameon=False)
    ax_m.tick_params(direction='in')
    div = make_axes_locatable(ax_m)
    ax_r = div.append_axes('bottom', size='32%', pad=.07, sharex=ax_m)
    ax_r.axhline(1., color='gray', linestyle='--', linewidth=.8, alpha=.8)
    for values, color, label in [(vm, 'red', r'$R_{\pi^-/\pi^+}$'), (v0, 'green', r'$R_{\pi^0/\pi^+}$')]:
        centers, ratio, err, valid = _ratio_series(vp, values, bins)
        if np.any(valid):
            ax_r.errorbar(centers[valid], ratio[valid], yerr=err[valid], fmt='o', markersize=3,
                          color=color, elinewidth=.8, capsize=2, label=label)
        else:
            ax_r.plot([], [], 'o', color=color, markersize=3, label=label)
    ax_r.legend(loc='upper left', ncol=2, frameon=False, fontsize=8)
    ax_r.set_ylabel(r'$R_{\pi/\pi^+}$')
    ax_r.set_ylim(*RATIO_YLIM)
    ax_r.tick_params(direction='in')
    ax_m.tick_params(labelbottom=False)
    ax_r.set_xlabel(r'$\phi_{hJ}$ [rad]')
    plt.tight_layout()
    with Path(output).open('xb') as stream:
        fig.savefig(stream, format='pdf')
    plt.close(fig)


def write_piplus_src_comparison_pdf(*, values_by_fraction, output, caption=''):
    """Unit-normalized pi+ at 0/5/10% SRC, with 5/0 and 10/0 ratio strips."""
    bins = np.linspace(*JH.ANGLE_RANGE, JH.ANGLE_BINS + 1)
    binw = bins[1] - bins[0]
    baseline = values_by_fraction[0.]
    fig, ax_m = plt.subplots(1, 1, figsize=(6.0, 4.2))
    for f, color in [(0., 'blue'), (.05, 'purple'), (.10, 'red')]:
        _add_step_curve(ax_m, values_by_fraction[f], bins, binw,
                        rf'$\pi^+~({int(round(f*100))}\%)$', color)
    ax_m.set_ylabel(r'$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$')
    if caption and str(caption).strip():
        ax_m.text(.03, .97, str(caption).strip(), transform=ax_m.transAxes,
                  ha='left', va='top', fontsize=10, color='0.25', linespacing=1.35)
    ax_m.set_ylim(*DENSITY_YLIM)
    ax_m.legend(loc='center left', bbox_to_anchor=(0., .38), frameon=False)
    ax_m.tick_params(direction='in')
    ax_r = make_axes_locatable(ax_m).append_axes('bottom', size='32%', pad=.07, sharex=ax_m)
    ax_r.axhline(1., color='gray', linestyle='--', linewidth=.8, alpha=.8)
    for f, color in [(.05, 'purple'), (.10, 'red')]:
        centers, ratio, err, valid = _ratio_series(baseline, values_by_fraction[f], bins)
        label = rf'${int(round(f*100))}\%/0\%$'
        ax_r.errorbar(centers[valid], ratio[valid], yerr=err[valid], fmt='o', markersize=3,
                      color=color, elinewidth=.8, capsize=2, label=label)
    ax_r.legend(loc='upper left', ncol=2, frameon=False, fontsize=8)
    ax_r.set_ylabel(r'$R_{\pi^+}^{f/0}$')
    ax_r.set_ylim(*RATIO_YLIM)
    ax_r.tick_params(direction='in')
    ax_m.tick_params(labelbottom=False)
    ax_r.set_xlabel(r'$\phi_{hJ}$ [rad]')
    plt.tight_layout()
    with Path(output).open('xb') as stream:
        fig.savefig(stream, format='pdf')
    plt.close(fig)
