"""
merge_ifc.py

Extended script: scan a directory for IFC files (optionally recursive), detect duplicate GlobalIds, and merge all found IFCs into one output file with optional GUID-remapping.

New features in this version:
- --base option to force a specific base file (already present)
- Progress display (simple terminal progress bars) during scanning and merging
- Detailed logging to a log file (and to console). Use --log to set the log file path; by default it writes to the same folder as the output file: output/merge_ifc.log

Usage examples:
  python merge_ifc.py detect --input-dir input --recursive --report output/duplicates.json --log output/scan.log
  python merge_ifc.py merge --input-dir input --base input/my_base.ifc -o output/merged.ifc --remap --recursive --log output/merge.log

Requires:
    pip install ifcopenshell

"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Optional
import logging
import sys
import time

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.guid


# --- Utility: progress bar (simple, no external deps) ---

def print_progress(prefix: str, current: int, total: int, bar_len: int = 30) -> None:
    if total <= 0:
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


# --- Logging setup ---

def setup_logger(log_path: Optional[str], verbose_console: bool = True) -> logging.Logger:
    logger = logging.getLogger("merge_ifc")
    logger.setLevel(logging.DEBUG)
    # avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose_console else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    # file handler if provided
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        logger.debug(f"Logging to file: {log_file}")

    return logger


# --- File discovery ---

def find_ifc_files_from_dir(dirpath: str, recursive: bool = False) -> List[str]:
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Input directory not found: {dirpath}")
    if recursive:
        files = sorted([str(f) for f in p.rglob("*.ifc")])
    else:
        files = sorted([str(f) for f in p.glob("*.ifc")])
    return files


def gather_input_files(inputs: Optional[List[str]], input_dir: Optional[str], recursive: bool = False) -> List[str]:
    files: List[str] = []
    if input_dir:
        files.extend(find_ifc_files_from_dir(input_dir, recursive=recursive))
    if inputs:
        files.extend([str(Path(p)) for p in inputs])
    # deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


# --- GUID collection & detection ---

def collect_guids_from_file(path: str, logger: Optional[logging.Logger] = None) -> dict:
    model = ifcopenshell.open(path)
    guids = defaultdict(list)
    count = 0
    for e in model.by_type("IfcRoot"):
        count += 1
        try:
            gid = getattr(e, "GlobalId", None)
        except Exception:
            gid = None
        if gid:
            guids[gid].append((e, e.is_a()))
    if logger:
        logger.debug(f"Collected {count} IfcRoot entities from {path}")
    return guids


def detect_duplicates(input_paths: List[str], logger: Optional[logging.Logger] = None) -> dict:
    occurrences = defaultdict(list)
    total = len(input_paths)
    for idx, p in enumerate(input_paths, start=1):
        if logger:
            logger.info(f"Scanning file {idx}/{total}: {p}")
        print_progress("Scanning files", idx, total)
        pstr = str(p)
        guids = collect_guids_from_file(pstr, logger=logger)
        for gid, entries in guids.items():
            for ent, ifc_type in entries:
                occurrences[gid].append({"file": pstr, "type": ifc_type})
    duplicates = {g: occ for g, occ in occurrences.items() if len(occ) > 1}
    if logger:
        logger.info(f"Duplicate GUIDs found: {len(duplicates)}")
    return duplicates


# --- Reports ---

def write_json_report(report: dict, outpath: str):
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_csv_report(report: dict, outpath: str):
    rows = []
    for gid, occ in report.items():
        for entry in occ:
            rows.append({"GlobalId": gid, "file": entry["file"], "ifc_type": entry["type"]})
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["GlobalId", "file", "ifc_type"])
        writer.writeheader()
        writer.writerows(rows)


# --- Merge operation ---

def merge_ifcs(output_path: str, input_paths: List[str], remap_guids: bool = False, dry_run: bool = False, logger: Optional[logging.Logger] = None, verbose: bool = True) -> List[dict]:
    if len(input_paths) < 1:
        raise ValueError("At least one input IFC is required to merge")

    base_path = input_paths[0]
    if verbose:
        msg = f"Opening base IFC: {base_path}"
        print(msg)
    if logger:
        logger.info(f"Opening base IFC: {base_path}")
    base = ifcopenshell.open(base_path)

    # build set of existing GlobalIds in base
    base_guids = set()
    for e in base.by_type("IfcRoot"):
        try:
            gid = getattr(e, "GlobalId", None)
        except Exception:
            gid = None
        if gid:
            base_guids.add(gid)

    # find main project in base (for re-wiring)
    base_projects = base.by_type("IfcProject")
    if not base_projects:
        raise RuntimeError("Base IFC has no IfcProject element")
    original_project = base_projects[0]

    remap_actions: List[dict] = []
    total_files = len(input_paths)

    # iterate source files (skip the base which is input_paths[0])
    for idx, src_path in enumerate(input_paths[1:], start=2):
        if verbose:
            print_progress("Merging files", idx, total_files)
        if logger:
            logger.info(f"Merging file {idx}/{total_files}: {src_path}")

        source = ifcopenshell.open(src_path)
        src_roots = source.by_type("IfcRoot")
        total_entities = len(src_roots)
        if logger:
            logger.debug(f"File {src_path} contains {total_entities} IfcRoot entities")

        added_entities = []
        for j, element in enumerate(src_roots, start=1):
            merged = base.add(element)
            added_entities.append(merged)

            # update per-entity progress occasionally
            if j % 200 == 0 or j == total_entities:
                print_progress(f"Entities in file {idx}/{total_files}", j, total_entities)

            # check GlobalId collisions on the merged entity
            try:
                merged_gid = getattr(merged, "GlobalId", None)
            except Exception:
                merged_gid = None
            if merged_gid:
                if merged_gid in base_guids:
                    if remap_guids:
                        new_gid = ifcopenshell.guid.new()
                        remap_actions.append({"file": src_path, "entity": merged.is_a(), "old": merged_gid, "new": new_gid})
                        if logger:
                            logger.debug(f"Collision {merged_gid} in {src_path} -> remap to {new_gid}")
                        try:
                            merged.GlobalId = new_gid
                        except Exception:
                            setattr(merged, "GlobalId", new_gid)
                        base_guids.add(new_gid)
                    else:
                        raise RuntimeError(f"GlobalId collision detected for {merged_gid} (entity {merged.is_a()}) — use --remap to auto-fix")
                else:
                    base_guids.add(merged_gid)

        # handle project merging: rewire references from the temporary project to the base project
        src_projects = [e for e in added_entities if e.is_a("IfcProject")]
        if src_projects:
            merged_project = src_projects[0]
            for inverse in base.get_inverse(merged_project):
                try:
                    ifcopenshell.util.element.replace_attribute(inverse, merged_project, original_project)
                except Exception:
                    for attr in dir(inverse):
                        try:
                            val = getattr(inverse, attr)
                            if val is merged_project:
                                try:
                                    setattr(inverse, attr, original_project)
                                except Exception:
                                    pass
                        except Exception:
                            pass
            try:
                base.remove(merged_project)
            except Exception:
                if logger:
                    logger.warning("Unable to remove temporary merged project entity; leaving it in the model.")

    if dry_run:
        if logger:
            logger.info("Dry-run complete. No output file written.")
            logger.info(f"Remap actions (sample 50): {remap_actions[:50]}")
        print("\nDry-run complete. No output file written. Summary of remap actions:")
        for a in remap_actions[:50]:
            print(a)
        if len(remap_actions) > 50:
            print(f"... and {len(remap_actions)-50} more remap actions")
        return remap_actions

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Writing merged IFC to: {outp}")
    if logger:
        logger.info(f"Writing merged IFC to: {outp}")
    base.write(str(outp))
    if verbose:
        print("Done.")
    if logger:
        logger.info("Merge complete.")
        logger.info(f"Total remap actions: {len(remap_actions)}")
    return remap_actions


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFC utilities: detect duplicate GlobalIds and merge with optional GUID remapping. Supports scanning input directories.")
    subparsers = parser.add_subparsers(dest="command")

    # detect
    p_detect = subparsers.add_parser("detect", help="Detect duplicate GlobalIds across IFC files")
    p_detect.add_argument("inputs", nargs="*", help="Optional explicit input IFC files")
    p_detect.add_argument("--input-dir", help="Directory to scan for IFC files")
    p_detect.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories")
    p_detect.add_argument("--report", help="Write JSON report of duplicates to this path")
    p_detect.add_argument("--csv", help="Write CSV report of duplicates to this path")
    p_detect.add_argument("--log", help="Optional log file path (writes detailed log)")

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge multiple IFC files (first is base).")
    p_merge.add_argument("inputs", nargs="*", help="Optional explicit input IFC files (they are appended after files discovered in --input-dir)")
    p_merge.add_argument("--input-dir", help="Directory to scan for IFC files")
    p_merge.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories")
    p_merge.add_argument("--base", "--base-file", dest="base", help="Optional: path to an IFC file to force as the base (will be placed first)")
    p_merge.add_argument("-o", "--output", required=True, help="Output IFC file path")
    p_merge.add_argument("--remap", action="store_true", help="Automatically remap duplicate GlobalIds during merge")
    p_merge.add_argument("--dry-run", action="store_true", help="Show what would happen (remap actions) but don't write output")
    p_merge.add_argument("--log", help="Optional log file path (writes detailed log)")

    args = parser.parse_args()

    if args.command == "detect":
        files = gather_input_files(args.inputs, args.input_dir, recursive=args.recursive)
        if not files:
            print("No IFC files found to scan. Provide input files or --input-dir with IFC files.")
            raise SystemExit(1)

        # determine log path if provided
        log_path = args.log
        logger = setup_logger(log_path, verbose_console=True)

        logger.info(f"Scanning {len(files)} file(s)")
        dup = detect_duplicates(files, logger=logger)
        if not dup:
            logger.info("No duplicate GlobalIds found.")
        else:
            logger.info(f"Found {len(dup)} duplicated GlobalId(s).")
        if args.report:
            write_json_report(dup, args.report)
            logger.info(f"JSON report written to {args.report}")
        if args.csv:
            write_csv_report(dup, args.csv)
            logger.info(f"CSV report written to {args.csv}")

    elif args.command == "merge":
        files = gather_input_files(args.inputs, args.input_dir, recursive=args.recursive)
        if not files:
            print("No IFC files found to merge. Provide input files or --input-dir with IFC files.")
            raise SystemExit(1)

        # If --base provided, ensure it exists and place it first in the list
        if args.base:
            base_path = str(Path(args.base))
            if not Path(base_path).exists():
                print(f"Base file specified but not found: {base_path}")
                raise SystemExit(1)
            files = [f for f in files if f != base_path]
            files.insert(0, base_path)

        # prepare logger: default log path next to output if not specified
        log_path = args.log if args.log else str(Path(args.output).parent / "merge_ifc.log")
        logger = setup_logger(log_path, verbose_console=True)

        logger.info(f"Merging {len(files)} file(s). Base: {files[0]}")
        print(f"Merging {len(files)} file(s). Note: the first file will be used as base: {files[0]}")
        try:
            remap_actions = merge_ifcs(args.output, files, remap_guids=args.remap, dry_run=args.dry_run, logger=logger)
            if remap_actions is not None and len(remap_actions) > 0:
                logger.info(f"Total remap actions: {len(remap_actions)}")
                print(f"Total remap actions: {len(remap_actions)}")
        except Exception as e:
            logger.exception("Merge failed")
            print("Merge failed:", e)
            raise
    else:
        parser.print_help()
