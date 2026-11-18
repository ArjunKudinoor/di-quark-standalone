#!/usr/bin/env python3
"""Fig. 3 two-panel p/n renderer using the existing curve and ratio helpers."""
from pathlib import Path
import numpy as np
from jet_hadron_pi_pm_figures import JH, plt, make_axes_locatable, _add_step_curve, _ratio_series

DENSITY_YLIM = (0.0, 3.5)
RATIO_YLIM = (0.8, 1.6)


def write_baryon_angle_pdf(*, protons, neutrons, fraction, output, caption=''):
    bins = np.linspace(*JH.ANGLE_RANGE, JH.ANGLE_BINS + 1)
    binw = bins[1] - bins[0]
    fig, ax_m = plt.subplots(1, 1, figsize=(6.0, 4.2))
    percent = int(round(100*fraction))
    _add_step_curve(ax_m, protons, bins, binw, rf'$p~({percent}\%)$', 'blue')
    _add_step_curve(ax_m, neutrons, bins, binw, rf'$n~({percent}\%)$', 'red')
    ax_m.set_ylabel(r'$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$')
    if caption and str(caption).strip():
        ax_m.text(.03, .97, str(caption).strip(), transform=ax_m.transAxes,
                  ha='left', va='top', fontsize=10, color='0.25', linespacing=1.35)
    ax_m.set_ylim(*DENSITY_YLIM)
    ax_m.legend(loc='center left', bbox_to_anchor=(0., .38), frameon=False)
    ax_m.tick_params(direction='in')
    ax_r = make_axes_locatable(ax_m).append_axes('bottom', size='32%', pad=.07, sharex=ax_m)
    ax_r.axhline(1., color='gray', linestyle='--', linewidth=.8, alpha=.8)
    centers, ratio, error, valid = _ratio_series(protons, neutrons, bins)
    ax_r.errorbar(centers[valid], ratio[valid], yerr=error[valid], fmt='o', markersize=3,
                  color='red', elinewidth=.8, capsize=2, label=r'$R_{n/p}$')
    ax_r.legend(loc='upper left', frameon=False, fontsize=8)
    ax_r.set_ylabel(r'$R_{n/p}$')
    ax_r.set_ylim(*RATIO_YLIM)
    ax_r.tick_params(direction='in')
    ax_m.tick_params(labelbottom=False)
    ax_r.set_xlabel(r'$\phi_{hJ}$ [rad]')
    plt.tight_layout()
    with Path(output).open('wb') as stream:
        fig.savefig(stream, format='pdf')
    plt.close(fig)


def write_baryon_src_comparison_pdf(*, species, values_by_fraction, output, caption=''):
    """Compare one species across SRC pools, using its 0% shape as denominator."""
    if species not in ('proton', 'neutron'):
        raise ValueError('species must be proton or neutron')
    symbol = 'p' if species == 'proton' else 'n'
    bins = np.linspace(*JH.ANGLE_RANGE, JH.ANGLE_BINS + 1)
    binw = bins[1] - bins[0]
    baseline = values_by_fraction[0.]
    fig, ax_m = plt.subplots(1, 1, figsize=(6.0, 4.2))
    for fraction, color in [(0., 'blue'), (.05, 'purple'), (.10, 'red')]:
        percent = int(round(100*fraction))
        _add_step_curve(ax_m, values_by_fraction[fraction], bins, binw,
                        rf'${symbol}~({percent}\%)$', color)
    ax_m.set_ylabel(r'$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$')
    if caption and str(caption).strip():
        ax_m.text(.03, .97, str(caption).strip(), transform=ax_m.transAxes,
                  ha='left', va='top', fontsize=10, color='0.25', linespacing=1.35)
    ax_m.set_ylim(*DENSITY_YLIM)
    ax_m.legend(loc='center left', bbox_to_anchor=(0., .38), frameon=False)
    ax_m.tick_params(direction='in')
    ax_r = make_axes_locatable(ax_m).append_axes('bottom', size='32%', pad=.07, sharex=ax_m)
    ax_r.axhline(1., color='gray', linestyle='--', linewidth=.8, alpha=.8)
    for fraction, color in [(.05, 'purple'), (.10, 'red')]:
        centers, ratio, error, valid = _ratio_series(baseline, values_by_fraction[fraction], bins)
        percent = int(round(100*fraction))
        label = rf'$R_{{{symbol}}}^{{{percent}\%/0\%}}$'
        # The original helper marks empty denominator/numerator bins invalid;
        # omit these points rather than drawing infinities or artificial zeros.
        ax_r.errorbar(centers[valid], ratio[valid], yerr=error[valid], fmt='o', markersize=3,
                      color=color, elinewidth=.8, capsize=2, label=label)
    ax_r.legend(loc='upper left', ncol=2, frameon=False, fontsize=8)
    ax_r.set_ylabel(rf'$R_{{{symbol}}}^{{f/0}}$')
    ax_r.set_ylim(*RATIO_YLIM)
    ax_r.tick_params(direction='in')
    ax_m.tick_params(labelbottom=False)
    ax_r.set_xlabel(r'$\phi_{hJ}$ [rad]')
    plt.tight_layout()
    with Path(output).open('wb') as stream:
        fig.savefig(stream, format='pdf')
    plt.close(fig)
