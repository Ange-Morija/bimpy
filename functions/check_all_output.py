#!/usr/bin/env python3
"""
functions/check_all_output_progress.py

Scan all IFC files in the 'output/' directory, show progress bars (global + per-file),
and write per-file JSON reports plus a combined report.

Requirements:
    pip install ifcopenshell
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional

import ifcopenshell

# ---------- Progress helpers ----------
def print_progress_line(prefix: str, current: int, total: int, bar_len: int = 30) -> None:
    if total <= 0:
        sys.stdout.write(f"\r{prefix} {current}/{total}")
        sys.stdout.flush()
        return
    frac = float(current) / float(total)
    filled = int(round(bar_len * frac))
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = frac * 100.0
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

# ---------- Per-file check (with per-entity progress) ----------
def check_file(ifc_path: Path, sample_limit: int = 20, entity_progress_interval: int = 2000) -> Dict[str, Any]:
    """
    Open the IFC, iterate IfcRoot entities and gather summary.
    Shows a lightweight per-entity progress indicator (based on total IfcRoot count).
    """
    model = ifcopenshell.open(str(ifc_path))

    summary: Dict[str, Any] = {}
    summary["path"] = str(ifc_path)
    summary["size_bytes"] = ifc_path.stat().st_size
    # Convert header to string to avoid JSON serialization issues
    try:
        summary["header"] = str(getattr(model, "header", None))
    except Exception:
        summary["header"] = None
    summary["schema"] = getattr(model, "schema", None)

    # First get counts we can cheaply compute
    try:
        total_roots = len(model.by_type("IfcRoot"))
    except Exception:
        total_roots = None

    # iterate IfcRoot and gather distribution + GlobalId map
    type_counter = Counter()
    gid_map = defaultdict(list)

    # We'll also count IfcProduct + representation items
    product_with_repr_count = 0
    product_with_items_count = 0
    product_with_repr_empty = 0
    sample_products = []

    # If total_roots is known, show progress. Otherwise, iterate without progress.
    processed = 0
    roots_iter = list(model.by_type("IfcRoot"))
    total = len(roots_iter)
    for e in roots_iter:
        processed += 1
        t = e.is_a()
        type_counter[t] += 1
        gid = getattr(e, "GlobalId", None)
        if gid:
            gid_map[gid].append(t)

        # If this entity is IfcProduct, check representation quickly
        if t == "IfcProduct":
            rep = getattr(e, "Representation", None)
            if rep and getattr(rep, "Representations", None):
                # count items across representations
                items_total = 0
                for r in rep.Representations:
                    items = getattr(r, "Items", None)
                    if items:
                        items_total += len(items)
                product_with_repr_count += 1
                if items_total > 0:
                    product_with_items_count += 1
                else:
                    product_with_repr_empty += 1
                if len(sample_products) < sample_limit:
                    sample_products.append({
                        "type": t,
                        "global_id": str(getattr(e, "GlobalId", None)),
                        "name": str(getattr(e, "Name", None)) if getattr(e, "Name", None) is not None else None,
                        "items_count": items_total
                    })

        # progress update
        if total and (processed % entity_progress_interval == 0 or processed == total):
            print_progress_line(f"  Entities", processed, total, bar_len=30)

    # geometric contexts
    grc = model.by_type("IfcGeometricRepresentationContext")
    summary["IfcGeometricRepresentationContext_count"] = len(grc)
    summary["IfcGeometricRepresentationContext_sample"] = [
        {"ContextType": getattr(c, "ContextType", None), "CoordinateSpaceDimension": getattr(c, "CoordinateSpaceDimension", None)}
        for c in grc[:5]
    ]

    # prepare counts and results
    summary["total_IfcRoot"] = total
    # Convert top_types to simple list of (type, count)
    summary["top_types"] = [(t, int(c)) for t, c in type_counter.most_common(40)]

    duplicates = {gid: types for gid, types in gid_map.items() if len(types) > 1}
    # ensure keys/values are serializable (strings and lists)
    summary["duplicate_globalids_count"] = len(duplicates)
    summary["duplicate_globalids_sample"] = [{ "globalid": str(gid), "types": list(duplicates[gid]) } for gid in list(duplicates)[:50]]

    summary["IfcProduct_total"] = int(type_counter.get("IfcProduct", 0))
    summary["IfcProduct_with_representation_count"] = int(product_with_repr_count)
    summary["IfcProduct_with_representation_with_items"] = int(product_with_items_count)
    summary["IfcProduct_with_representation_empty"] = int(product_with_repr_empty)
    summary["sample_products_with_representation"] = sample_products
    summary["likely_no_geometry"] = (product_with_items_count == 0)

    return summary

# ---------- Main: scan output/ and run checks with a global progress bar ----------
def main():
    out_dir = Path("output")
    if not out_dir.exists():
        print("Le dossier 'output/' n'existe pas. Crée un dossier 'output' ou place-y des fichiers .ifc")
        return

    files = sorted(out_dir.glob("*.ifc"))
    if not files:
        print("Aucun fichier .ifc trouvé dans output/")
        return

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    total_files = len(files)
    for idx, f in enumerate(files, start=1):
        # global progress bar
        print_progress_line("Files", idx-1, total_files, bar_len=40)
        print(f"\nProcessing file {idx}/{total_files}: {f.name}")
        # per-file processing with progress printed inside
        summary = check_file(f)
        # write per-file JSON (use default=str as fallback)
        outpath = reports_dir / (f.name + "_report.json")
        with open(outpath, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
        print(f"  Report written to: {outpath}")
        combined.append(summary)
        # update files progress to current
        print_progress_line("Files", idx, total_files, bar_len=40)

    # write combined report (use default=str as fallback)
    combined_path = reports_dir.parent / "combined_report.json"
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False, default=str)

    print(f"\nAll done. Per-file reports in: {reports_dir}")
    print(f"Combined report: {combined_path}")

if __name__ == "__main__":
    main()
