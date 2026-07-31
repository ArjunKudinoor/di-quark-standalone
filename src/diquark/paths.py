"""
Resolve repository vs external data directories.

Large simulation outputs default to ``~/Data/Di-quark-pythia-standalone-nosync/`` (outside the repo;
the ``-nosync`` name is a macOS convention to discourage iCloud sync of heavy data).

Override with ``DIQUARK_DATA_ROOT`` or ``DIQUARK_OUTPUT_ROOT`` (same meaning; first wins).

If the resolved root lies under ``~/Documents`` or ``~/Desktop``, a one-time
:class:`UserWarning` is emitted (those locations are often iCloud-synced).
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_DATA_DIRNAME = "Di-quark-pythia-standalone-nosync"
_warned_risky_root: Optional[str] = None

__all__ = [
    "repo_root",
    "diquark_data_root",
    "outputs_dir",
    "pythia_finalstate_raw_dir",
    "four_observables_outputs_dir",
    "analysis_outputs_dir",
    "write_run_manifest",
    "count_files_under",
]


def repo_root() -> Path:
    """Project root (directory containing ``src/`` and ``scripts/``)."""
    return Path(__file__).resolve().parents[2]


def _path_is_under(path: Path, ancestor: Path) -> bool:
    """True if ``path`` is ``ancestor`` or contained in it (after resolving)."""
    try:
        path.resolve().relative_to(ancestor.resolve())
        return True
    except ValueError:
        return False


def _warn_if_icloud_prone_data_root(root: Path) -> None:
    """Warn once per process if the data root lies under Desktop or Documents.

    Must only be called from :func:`diquark_data_root` so ``stacklevel=3`` attributes
    the warning to the caller of :func:`diquark_data_root` (e.g. ``outputs_dir`` or scripts).
    """
    global _warned_risky_root
    key = str(root.resolve())
    if _warned_risky_root == key:
        return
    home = Path.home()
    for folder, label in (("Documents", "Documents"), ("Desktop", "Desktop")):
        risky = home / folder
        if _path_is_under(root, risky):
            _warned_risky_root = key
            warnings.warn(
                f"DIQUARK_DATA_ROOT resolves under {label} ({root}). "
                "Large simulation outputs should not be stored in iCloud-synced locations "
                f"such as ~/{label}. Prefer ~/Data/{_DEFAULT_DATA_DIRNAME} or set "
                "DIQUARK_DATA_ROOT to another non-synced path. See README.",
                UserWarning,
                stacklevel=3,
            )
            return


def diquark_data_root() -> Path:
    """Root for all generated data (shards, ``outputs/``, run manifests)."""
    for key in ("DIQUARK_DATA_ROOT", "DIQUARK_OUTPUT_ROOT"):
        v = os.environ.get(key)
        if v:
            root = Path(v).expanduser().resolve()
            _warn_if_icloud_prone_data_root(root)
            return root
    root = (Path.home() / "Data" / _DEFAULT_DATA_DIRNAME).resolve()
    _warn_if_icloud_prone_data_root(root)
    return root


def outputs_dir() -> Path:
    return diquark_data_root() / "outputs"


def pythia_finalstate_raw_dir() -> Path:
    return diquark_data_root() / "pythia_finalstate_raw"


def four_observables_outputs_dir() -> Path:
    return diquark_data_root() / "four_observables_outputs"


def analysis_outputs_dir() -> Path:
    """PDF/NPY that previously lived at repo root (eta/pTrel plots, transverse arrays, etc.)."""
    return outputs_dir() / "analysis"


def write_run_manifest(
    *,
    run_label: str,
    script_name: str,
    top_level_dirs_written: Optional[List[str]] = None,
    approximate_files_created: Optional[int] = None,
    approximate_files_capped: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Append a small JSON record under ``<data_root>/run_manifests/``.
    """
    root = diquark_data_root()
    manifest_dir = root / "run_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    body: Dict[str, Any] = {
        "run_label": run_label,
        "script": script_name,
        "timestamp_utc": ts,
        "resolved_data_root": str(root),
        "major_top_level_dirs_written": top_level_dirs_written or [],
        "approximate_files_created": approximate_files_created,
        "approximate_files_capped": approximate_files_capped,
    }
    if extra:
        body["extra"] = extra
    path = manifest_dir / f"{run_label}_{ts}.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return path


def count_files_under(root: Path, limit: int = 50_000) -> Tuple[int, bool]:
    """
    Best-effort file count under ``root`` (bounded). Returns (count, hit_limit).
    """
    n = 0
    try:
        for p in root.rglob("*"):
            if p.is_file():
                n += 1
                if n >= limit:
                    return n, True
    except OSError:
        return n, False
    return n, False
