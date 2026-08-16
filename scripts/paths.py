#!/usr/bin/env python3
"""Centralized data-path interface for TransDANN_Liver_Cancer.

Project storage policy (2026-08-02):
  - Lightweight, paper-critical artifacts (code, docs, tables, figures, JSON
    results, logs) live on the system disk, in the git repository.
  - Heavy, regenerable artifacts (trained model weights ``*.pth``, hidden
    feature arrays ``*.npy``) live on the large /data drive to keep the
    system disk small. The repo holds symlinks that point into the data root,
    so every existing script keeps working with its original ``results/...``
    paths (the symlink IS the transparent interface for backward compat).

Resolution order for the data root:
  1. $TRANSDANN_DATA_ROOT      # explicit override (e.g. a different server)
  2. /data/<user>/projects/transdann_liver_cancer   # default data drive
  3. <repo>/results            # fallback: no /data mounted (fresh clone)

Usage:
  python3 scripts/paths.py --print      # print all resolved paths (default)
  python3 scripts/paths.py --relink     # (re)create results/ symlinks into data root
  python3 scripts/paths.py --exists     # exit 0 only if data root is reachable

In scripts:
  import paths   (or: from scripts import paths)
  paths.experiments_dir()   # Path to results/experiments/ (real or symlinked)
  paths.lihc_experiments_dir()
  paths.results_dir()
  paths.data_root()         # the resolved data root (may equal results/ if no /data)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Default location of the heavy data drive (same machine layout as the
# sibling FM-HGL project under /data/<user>/projects/).
_DEFAULT_DATA_ROOT = Path("/data") / os.environ.get("USER", "user") / "projects" / "transdann_liver_cancer"

# Sub-paths relative to the data root that are heavy/regenerable.
_HEAVY_SUBDIRS = ("experiments", "lihc_experiments")


def data_root() -> Path:
    """Return the resolved data root (exists-checked, with fallback)."""
    override = os.environ.get("TRANSDANN_DATA_ROOT")
    candidates = ([Path(override)] if override else []) + [_DEFAULT_DATA_ROOT, RESULTS_DIR]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return RESULTS_DIR


def _heavy_dir(name: str) -> Path:
    """Path to a heavy sub-dir. If it lives on the data drive, return it
    there directly (do not depend on the symlink); otherwise return the
    repo-local path (no /data mounted)."""
    root = data_root()
    return root / "results" / name if root != RESULTS_DIR else RESULTS_DIR / name


def experiments_dir() -> Path:
    return _heavy_dir("experiments")


def lihc_experiments_dir() -> Path:
    return _heavy_dir("lihc_experiments")


def results_dir() -> Path:
    """Repo-local results dir (code-adjacent; tables/figures/logs stay here)."""
    return RESULTS_DIR


def relink(force: bool = False) -> None:
    """(Re)create repo-local symlinks pointing into the data root.

    Existing scripts reference ``results/experiments`` and
    ``results/lihc_experiments``; these symlinks keep those paths resolving
    when the data lives on /data. Safe to run after a fresh clone / move.
    """
    root = data_root()
    if root == RESULTS_DIR:
        print("[paths] no external data root found; using repo-local results/ (no symlinks needed)")
        return
    for name in _HEAVY_SUBDIRS:
        target = root / "results" / name
        link = RESULTS_DIR / name
        if not target.is_dir():
            print(f"[paths] WARN: data root subdir missing: {target}")
            continue
        if link.is_symlink():
            if force or not link.exists():
                link.unlink()
                link.symlink_to(target, target_is_directory=True)
                print(f"[paths] relinked {link} -> {target}")
            else:
                print(f"[paths] ok (already linked): {link} -> {target}")
        elif link.exists():
            print(f"[paths] WARN: {link} exists as a real dir; not replacing. "
                  f"Move it to the data root first, then --relink --force.")
        else:
            link.symlink_to(target, target_is_directory=True)
            print(f"[paths] linked {link} -> {target}")


def print_paths() -> None:
    print(f"project_root      : {PROJECT_ROOT}")
    print(f"results_dir       : {results_dir()}")
    print(f"data_root         : {data_root()}")
    print(f"experiments_dir   : {experiments_dir()}")
    print(f"lihc_experiments  : {lihc_experiments_dir()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="print resolved paths (default)")
    ap.add_argument("--relink", action="store_true", help="(re)create results/ symlinks into the data root")
    ap.add_argument("--force", action="store_true", help="with --relink, replace existing symlinks")
    ap.add_argument("--exists", action="store_true", help="exit 0 iff data root reachable")
    args = ap.parse_args()

    if args.exists:
        root = data_root()
        return 0 if root != RESULTS_DIR else 1
    if args.relink:
        relink(force=args.force)
        return 0
    print_paths()
    return 0


if __name__ == "__main__":
    sys.exit(main())
