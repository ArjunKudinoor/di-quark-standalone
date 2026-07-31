#!/usr/bin/env python3
"""
Build PDF comparison plots from jet_hadron_pi_pm_observables.csv (no PYTHIA).

Example (CSV under your data root, default ``~/Data/Di-quark-pythia-standalone-nosync/outputs/``):
  python scripts/analysis/plot_jet_hadron_pi_pm_observables.py \\
    --csv ~/Data/Di-quark-pythia-standalone-nosync/outputs/dis_isr_parton_dataset/split_90_10/jet_hadron_pi_pm_observables.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import pandas as pd  # noqa: E402

from jet_hadron_pi_pm_figures import (  # noqa: E402
    dataframe_to_records,
    write_combined_pi_pm_angle_multi_signal_pdf,
    write_combined_pi_pm_angle_triptych_pdfs,
    write_combined_pi_pm_comparison_pdfs,
    write_split_pi_pm_comparison_pdfs,
)

_COMBINED_KEYS_ALLOWED = ("angle", "sum_mag", "diff_mag", "Pbar_t")


def _truthy_ok_mask(df: pd.DataFrame) -> pd.Series:
    if "ok" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["ok"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def subsample_altered_unchanged_for_pool(
    df: pd.DataFrame,
    altered_fraction: float,
    *,
    seed: int,
) -> Tuple[pd.DataFrame, int, int, float]:
    """
    Return a subset of rows such that among included ``altered`` + ``unchanged`` rows,
    the fraction of ``sample == 'altered'`` is as close as possible to ``altered_fraction``,
    using random subsampling without replacement within each arm.

    Returns ``(subset_df, n_altered, n_unchanged, achieved_fraction)``.
    """
    f = float(altered_fraction)
    if not (0.0 < f < 1.0) or not math.isfinite(f):
        raise ValueError("altered_fraction must lie in (0, 1).")

    if "sample" not in df.columns:
        raise ValueError("CSV needs a ``sample`` column for altered/unchanged balancing.")

    m = _truthy_ok_mask(df)
    alt = df.loc[m & (df["sample"].astype(str) == "altered")]
    unc = df.loc[m & (df["sample"].astype(str) == "unchanged")]
    na, nu = len(alt), len(unc)
    if na == 0 or nu == 0:
        return df.loc[m].copy(), na, nu, (float(na) / float(na + nu)) if (na + nu) else 0.0

    need_unc_per_alt = (1.0 - f) / f

    if nu >= na * need_unc_per_alt:
        n_unc_take = int(min(nu, round(na * need_unc_per_alt)))
        alt_s = alt
        unc_s = unc.sample(n=n_unc_take, random_state=seed)
    else:
        unc_s = unc
        n_alt_take = int(min(na, max(1, round(nu * f / (1.0 - f)))))
        alt_s = alt.sample(n=n_alt_take, random_state=seed)

    out = pd.concat([alt_s, unc_s], axis=0)
    if "event_id" in out.columns:
        out = out.sort_values(["event_id", "sample"])
    else:
        out = out.sort_index()
    n_a, n_u = len(alt_s), len(unc_s)
    frac = float(n_a) / float(n_a + n_u) if (n_a + n_u) else 0.0
    return out.reset_index(drop=True), n_a, n_u, frac


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot π± jet–hadron observables from CSV.")
    ap.add_argument("--csv", type=Path, required=True, help="jet_hadron_pi_pm_observables.csv")
    ap.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Output directory for PDFs (default: same directory as --csv).",
    )
    ap.add_argument(
        "--layout",
        choices=("split", "combined", "triptych"),
        default="split",
        help=(
            "split: altered vs unchanged (two main rows per PDF). "
            "combined: pool all rows (one π± panel + ratio per PDF; filenames contain '_combined_pi_pm'). "
            "triptych: 1×3 horizontal panels (0%, 5%, 10% π⁻) with R_{π⁻/π⁺} ratio strips."
        ),
    )
    ap.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title for combined layout (default: generic pooled subsamples title).",
    )
    ap.add_argument(
        "--exclude-samples",
        default="",
        help="Comma-separated sample labels to drop before plotting (e.g. 'altered' omits signal).",
    )
    ap.add_argument(
        "--combined-filename-suffix",
        default="",
        help=(
            "Suffix before .pdf for combined layout (e.g. '_unchanged_only'). "
            "If --exclude-samples includes 'altered' and this is empty, defaults to '_unchanged_only'."
        ),
    )
    ap.add_argument(
        "--only-ok-rows",
        action="store_true",
        help="Keep only rows where ok is truthy (much faster on large CSVs with few passing events).",
    )
    ap.add_argument(
        "--pooled-altered-fraction",
        type=float,
        default=None,
        help=(
            "For --layout combined only: randomly subsample rows so that about this fraction "
            "of pooled (truthy-ok) rows have sample=='altered' and the rest sample=='unchanged'. "
            "Uses --pooled-balance-seed for reproducibility."
        ),
    )
    ap.add_argument(
        "--pooled-balance-seed",
        type=int,
        default=42,
        help="RNG seed for --pooled-altered-fraction subsampling (default: 42).",
    )
    ap.add_argument(
        "--compare-pooled-altered-fractions",
        type=str,
        default="",
        help=(
            "Combined-layout angle-only overlay using one shared pi+ curve and multiple pi- curves "
            "from pooled altered fractions, e.g. '0,0.05,0.10' (0 = unchanged-only pi-)."
        ),
    )
    ap.add_argument(
        "--combined-only",
        type=str,
        default="",
        help=(
            "Comma-separated subset of combined PDF keys: angle, sum_mag, diff_mag, Pbar_t. "
            "Example: 'angle' writes only the φ_hJ combined figure. "
            "Only used with --layout combined."
        ),
    )
    ap.add_argument(
        "--no-title",
        action="store_true",
        help="Omit figure titles (combined main-axis title and split row panel titles).",
    )
    ap.add_argument(
        "--caption",
        type=str,
        default="",
        help=(
            "Optional multi-line text (use \\n) on combined-layout main panel, "
            "e.g. kinematics cuts (matplotlib mathtext/LaTeX)."
        ),
    )
    ap.add_argument(
        "--log-density",
        action="store_true",
        help="Use log scale for the (1/σ) dσ/dx density panel (ratio panel stays linear).",
    )
    ap.add_argument(
        "--triptych-standalone-panels",
        action="store_true",
        help="With --layout triptych, also write each of the three panels as its own PDF.",
    )
    args = ap.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv, low_memory=False)
    if args.only_ok_rows:
        if "ok" not in df.columns:
            raise SystemExit("CSV has no ``ok`` column; omit --only-ok-rows.")
        om = df["ok"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        df = df.loc[om]
        print(f"Using {len(df):,} ok rows for plotting.")
    if args.exclude_samples.strip():
        drop = {s.strip() for s in args.exclude_samples.split(",") if s.strip()}
        df = df[~df["sample"].astype(str).isin(drop)]
        print(f"After --exclude-samples: {len(df):,} rows.")

    if args.pooled_altered_fraction is not None:
        if args.layout != "combined":
            raise SystemExit("--pooled-altered-fraction requires --layout combined.")
        df, n_a, n_u, frac = subsample_altered_unchanged_for_pool(
            df,
            float(args.pooled_altered_fraction),
            seed=int(args.pooled_balance_seed),
        )
        print(
            f"Pooled subsample: altered={n_a:,} unchanged={n_u:,} "
            f"(altered fraction {frac:.4f}, target {float(args.pooled_altered_fraction):.4f})."
        )

    records = dataframe_to_records(df)
    figure_dir = args.figure_dir or args.csv.parent
    show_title = not bool(args.no_title)
    if args.layout == "split":
        if args.pooled_altered_fraction is not None:
            raise SystemExit("--pooled-altered-fraction is only valid with --layout combined.")
        if args.compare_pooled_altered_fractions.strip():
            raise SystemExit("--compare-pooled-altered-fractions is only valid with --layout combined.")
        if args.combined_only.strip():
            raise SystemExit("--combined-only is only valid with --layout combined.")
        if args.triptych_standalone_panels:
            raise SystemExit("--triptych-standalone-panels requires --layout triptych.")
        paths = write_split_pi_pm_comparison_pdfs(
            records, figure_dir, show_panel_titles=show_title
        )
    elif args.layout == "triptych":
        if args.pooled_altered_fraction is not None:
            raise SystemExit("--pooled-altered-fraction is not valid with --layout triptych.")
        if args.compare_pooled_altered_fractions.strip():
            raise SystemExit("--compare-pooled-altered-fractions is not valid with --layout triptych.")
        if args.combined_only.strip():
            raise SystemExit("--combined-only is not valid with --layout triptych.")
        if args.log_density:
            raise SystemExit("--log-density is not valid with --layout triptych.")

        m = _truthy_ok_mask(df)
        unc_df = df.loc[m & (df["sample"].astype(str) == "unchanged")].copy()
        if len(unc_df) == 0:
            raise SystemExit("No unchanged rows available for shared pi+ reference.")
        records_piplus = dataframe_to_records(unc_df)

        triptych_spec = [
            (0.0, "Unmodified", r"$\pi^{-}$", "gray"),
            (0.05, "5% modified", r"$\pi^{-}\ \left(5\,\%\right)$", "purple"),
            (0.10, "10% modified", r"$\pi^{-}\ \left(10\,\%\right)$", "red"),
        ]
        panel_records = []
        for ix, (f, panel_title, piminus_label, color) in enumerate(triptych_spec):
            if f == 0.0:
                mix_df = unc_df.copy()
                n_a, n_u, got = 0, len(mix_df), 0.0
            else:
                mix_df, n_a, n_u, got = subsample_altered_unchanged_for_pool(
                    df.loc[m].copy(),
                    f,
                    seed=int(args.pooled_balance_seed) + ix,
                )
            print(
                f"Triptych panel {panel_title!r}: altered={n_a:,} unchanged={n_u:,} "
                f"(target f={f:.4f}, achieved {got:.4f})"
            )
            panel_records.append(
                (panel_title, piminus_label, color, dataframe_to_records(mix_df))
            )

        suf = args.combined_filename_suffix.strip()
        if suf and not suf.startswith("_"):
            suf = "_" + suf
        paths = write_combined_pi_pm_angle_triptych_pdfs(
            records_piplus=records_piplus,
            panel_records=panel_records,
            figure_dir=figure_dir,
            filename_suffix=suf,
            caption=args.caption.strip(),
            write_standalone_panels=bool(args.triptych_standalone_panels),
        )
    else:
        title = args.title.strip() or r"$\pi^\pm$ (pooled over subsamples)"
        suf = args.combined_filename_suffix.strip()
        if not suf and args.exclude_samples.strip():
            drop = {s.strip() for s in args.exclude_samples.split(",") if s.strip()}
            if "altered" in drop:
                suf = "_unchanged_only"
        if suf and not suf.startswith("_"):
            suf = "_" + suf
        include_keys: Optional[Tuple[str, ...]] = None
        if args.combined_only.strip():
            include_keys = tuple(
                x.strip()
                for x in args.combined_only.split(",")
                if x.strip()
            )
            bad = [k for k in include_keys if k not in _COMBINED_KEYS_ALLOWED]
            if bad:
                raise SystemExit(
                    f"unknown --combined-only keys {bad}; allowed: {list(_COMBINED_KEYS_ALLOWED)}"
                )
        if args.compare_pooled_altered_fractions.strip():
            if args.pooled_altered_fraction is not None:
                raise SystemExit(
                    "Use either --pooled-altered-fraction or --compare-pooled-altered-fractions, not both."
                )
            if include_keys is not None and include_keys != ("angle",):
                raise SystemExit(
                    "--compare-pooled-altered-fractions currently supports --combined-only angle."
                )
            if include_keys is None:
                include_keys = ("angle",)
            frac_items = [
                x.strip() for x in args.compare_pooled_altered_fractions.split(",") if x.strip()
            ]
            fracs: List[float] = [float(x) for x in frac_items]
            if len(fracs) < 1:
                raise SystemExit("--compare-pooled-altered-fractions needs at least one fraction.")
            if any((not math.isfinite(f) or f < 0.0 or f >= 1.0) for f in fracs):
                raise SystemExit("All compare fractions must lie in [0, 1).")

            m = _truthy_ok_mask(df)
            unc_df = df.loc[m & (df["sample"].astype(str) == "unchanged")].copy()
            if len(unc_df) == 0:
                raise SystemExit("No unchanged rows available for shared pi+ denominator.")
            records_piplus = dataframe_to_records(unc_df)

            signal_records = []
            palette = ["gray", "purple", "red", "darkorange", "green", "magenta"]
            for ix, f in enumerate(fracs):
                if f == 0.0:
                    mix_df = unc_df.copy()
                    n_a, n_u, got = 0, len(mix_df), 0.0
                    print(
                        f"Compare pooled subsample f=0.0000: altered=0 unchanged={n_u:,} "
                        f"(achieved {got:.4f})"
                    )
                else:
                    mix_df, n_a, n_u, got = subsample_altered_unchanged_for_pool(
                        df.loc[m].copy(),
                        f,
                        seed=int(args.pooled_balance_seed) + ix,
                    )
                    print(
                        f"Compare pooled subsample f={f:.4f}: altered={n_a:,} unchanged={n_u:,} "
                        f"(achieved {got:.4f})"
                    )
                signal_records.append(
                    (
                        rf"$\pi^-~({int(round(f * 100.0))}\%)$",
                        palette[ix % len(palette)],
                        dataframe_to_records(mix_df),
                    )
                )

            out = write_combined_pi_pm_angle_multi_signal_pdf(
                records_piplus=records_piplus,
                signal_records=signal_records,
                figure_dir=figure_dir,
                filename_suffix=suf,
                title=title if show_title else "",
                caption=args.caption.strip(),
                density_yscale="log" if args.log_density else "linear",
            )
            paths = [out]
        else:
            paths = write_combined_pi_pm_comparison_pdfs(
                records,
                figure_dir,
                title=title,
                filename_suffix=suf,
                include_combined_keys=include_keys,
                show_title=show_title,
                caption=args.caption.strip(),
            )
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
