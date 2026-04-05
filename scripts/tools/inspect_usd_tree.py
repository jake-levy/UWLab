#!/usr/bin/env python3

"""Print a USD prim hierarchy as a tree."""

from __future__ import annotations

import argparse
import os
import tempfile

from pxr import Usd, UsdGeom

try:
    from isaaclab.utils.assets import retrieve_file_path
except Exception:
    retrieve_file_path = None

try:
    from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
except Exception:
    UWLAB_CLOUD_ASSETS_DIR = None


DEFAULT_PAT_VENTION_PATH = (
    f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd"
    if UWLAB_CLOUD_ASSETS_DIR is not None
    else None
)


def _resolve_asset_path(asset_path: str) -> str:
    """Resolve local/remote asset paths when IsaacLab helpers are available."""
    if retrieve_file_path is None:
        return asset_path
    download_dir = os.path.join(tempfile.gettempdir(), "uwlab_usd_inspect")
    os.makedirs(download_dir, exist_ok=True)
    return retrieve_file_path(asset_path, download_dir=download_dir)


def _visibility_label(prim: Usd.Prim) -> str:
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return ""
    visibility = imageable.GetVisibilityAttr().Get()
    if visibility is None or visibility == UsdGeom.Tokens.inherited:
        return ""
    return f" visibility={visibility}"


def _kind_labels(prim: Usd.Prim) -> str:
    labels: list[str] = []
    if prim.IsA(UsdGeom.Mesh):
        labels.append("Mesh")
    elif prim.IsA(UsdGeom.Xform):
        labels.append("Xform")
    if prim.HasAuthoredReferences():
        labels.append("ref")
    if prim.HasAuthoredInherits():
        labels.append("inherits")
    return f" [{' '.join(labels)}]" if labels else ""


def _print_tree(prim: Usd.Prim, depth: int, max_depth: int | None, show_paths: bool) -> None:
    if max_depth is not None and depth > max_depth:
        return

    indent = "  " * depth
    label = prim.GetName() or "/"
    type_name = prim.GetTypeName() or "PseudoRoot"
    path_suffix = f" path={prim.GetPath()}" if show_paths else ""
    print(f"{indent}{label} <{type_name}>{_kind_labels(prim)}{_visibility_label(prim)}{path_suffix}")

    for child in prim.GetChildren():
        _print_tree(child, depth + 1, max_depth, show_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the prim hierarchy of a USD asset.")
    parser.add_argument(
        "usd_path",
        nargs="?",
        default=DEFAULT_PAT_VENTION_PATH,
        help="Path to the USD file. Defaults to pat_vention.usd when UWLab assets are available.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum tree depth to print.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print full prim paths in addition to tree labels.",
    )
    parser.add_argument(
        "--default-prim-only",
        action="store_true",
        help="Start from the stage default prim instead of the pseudo-root.",
    )
    args = parser.parse_args()

    if args.usd_path is None:
        raise SystemExit("No USD path provided and default pat_vention.usd path is unavailable.")

    resolved_path = _resolve_asset_path(args.usd_path)
    if not resolved_path:
        raise SystemExit(f"Failed to resolve asset path: {args.usd_path}")

    stage = Usd.Stage.Open(resolved_path)
    if stage is None:
        raise SystemExit(f"Failed to open USD stage: {resolved_path}")

    print(f"USD: {args.usd_path}")
    print(f"Resolved: {resolved_path}")

    root = stage.GetDefaultPrim() if args.default_prim_only else stage.GetPseudoRoot()
    if not root:
        raise SystemExit("USD stage does not have a valid root prim.")

    _print_tree(root, depth=0, max_depth=args.max_depth, show_paths=args.show_paths)


if __name__ == "__main__":
    main()
