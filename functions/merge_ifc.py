"""Utilities for detecting duplicate GlobalIds and merging IFC files.

Public API:
 - gather_input_files(inputs, input_dir, recursive)
 - setup_logger(log_path, verbose_console)
 - detect_duplicates(input_paths, logger)
 - merge_ifcs(output_path, input_paths, remap_guids, dry_run, logger)

This module is a cleaned and documented version of the original script.
"""

from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Dict
import logging
import sys
import json
import csv

try:
    import ifcopenshell
    import ifcopenshell.util.element
    import ifcopenshell.guid
except Exception:  # pragma: no cover - dependency
    ifcopenshell = None


def setup_logger(log_path: Optional[str], verbose_console: bool = True) -> logging.Logger:
    logger = logging.getLogger("bimpy.merge_ifc")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose_console else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    return logger


def gather_input_files(inputs: Optional[List[str]], input_dir: Optional[str], recursive: bool = False) -> List[str]:
    files: List[str] = []
    if input_dir:
        p = Path(input_dir)
        if recursive:
            files.extend(sorted([str(f) for f in p.rglob("*.ifc")]))
        else:
            files.extend(sorted([str(f) for f in p.glob("*.ifc")]))
    if inputs:
        files.extend([str(Path(x)) for x in inputs])
    # dedupe preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def collect_guids_from_file(path: str, logger: Optional[logging.Logger] = None) -> Dict[str, List[Dict]]:
    if ifcopenshell is None:
        raise RuntimeError("ifcopenshell not installed")
    model = ifcopenshell.open(path)
    guids = defaultdict(list)
    for e in model.by_type("IfcRoot"):
        gid = getattr(e, "GlobalId", None)
        if gid:
            guids[str(gid)].append({"type": e.is_a(), "entity": e})
    if logger:
        logger.debug(f"Collected {len(list(model.by_type('IfcRoot')))} IfcRoot from {path}")
    return guids


def detect_duplicates(input_paths: List[str], logger: Optional[logging.Logger] = None) -> Dict[str, List[Dict]]:
    occurrences = defaultdict(list)
    for p in input_paths:
        if logger:
            logger.info(f"Scanning: {p}")
        guids = collect_guids_from_file(p, logger=logger)
        for gid, entries in guids.items():
            for ent in entries:
                occurrences[gid].append({"file": p, "type": ent["type"]})
    duplicates = {g: occ for g, occ in occurrences.items() if len(occ) > 1}
    if logger:
        logger.info(f"Found {len(duplicates)} duplicate GlobalId(s)")
    return duplicates


def write_json_report(report: dict, outpath: str):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def write_csv_report(report: dict, outpath: str):
    rows = []
    for gid, occ in report.items():
        for entry in occ:
            rows.append({"GlobalId": gid, "file": entry.get("file"), "ifc_type": entry.get("type")})
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline='', encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["GlobalId", "file", "ifc_type"])
        writer.writeheader()
        writer.writerows(rows)


def merge_ifcs(output_path: str, input_paths: List[str], remap_guids: bool = False, dry_run: bool = False, logger: Optional[logging.Logger] = None, verbose: bool = True) -> List[Dict]:
    if ifcopenshell is None:
        raise RuntimeError("ifcopenshell not installed")
    if not input_paths:
        raise ValueError("No input files provided")
    base_path = input_paths[0]
    base = ifcopenshell.open(base_path)

    base_guids = set()
    for e in base.by_type("IfcRoot"):
        gid = getattr(e, "GlobalId", None)
        if gid:
            base_guids.add(str(gid))

    base_projects = base.by_type("IfcProject")
    original_project = base_projects[0] if base_projects else None

    remap_actions = []

    for src in input_paths[1:]:
        src_model = ifcopenshell.open(src)
        for ent in src_model.by_type("IfcRoot"):
            merged = base.add(ent)
            mgid = getattr(merged, "GlobalId", None)
            if mgid:
                mgid = str(mgid)
                if mgid in base_guids:
                    if remap_guids:
                        new_gid = ifcopenshell.guid.new()
                        try:
                            merged.GlobalId = new_gid
                        except Exception:
                            setattr(merged, "GlobalId", new_gid)
                        remap_actions.append({"file": src, "old": mgid, "new": new_gid})
                        base_guids.add(str(new_gid))
                    else:
                        raise RuntimeError(f"GlobalId collision: {mgid} in {src}")
                else:
                    base_guids.add(mgid)
        # rewire project references if needed
        if original_project:
            for p in base.by_type("IfcProject"):
                if p is not original_project:
                    try:
                        base.remove(p)
                    except Exception:
                        pass

    if dry_run:
        if logger:
            logger.info("Dry-run: no file written")
        return remap_actions

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    base.write(str(outp))
    if logger:
        logger.info(f"Wrote merged IFC to: {outp}")
    return remap_actions

