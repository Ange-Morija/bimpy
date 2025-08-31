#!/usr/bin/env python3
# functions/preflight_ifc.py
"""
Preflight checker for IFC inputs before merge.
Usage:
  python functions/preflight_ifc.py --input-dir input --output output/preflight_report.json
  python functions/preflight_ifc.py --files input/a.ifc input/b.ifc --output out.json
Requires: pip install ifcopenshell
"""

import argparse, json, sys
from pathlib import Path
from collections import defaultdict, Counter
import math

try:
    import ifcopenshell
except Exception as e:
    print("ifcopenshell required: pip install ifcopenshell")
    raise

def inspect_file(path: Path):
    model = ifcopenshell.open(str(path))
    info = {"path": str(path), "size_bytes": path.stat().st_size, "schema": getattr(model, "schema", None)}
    # IfcProject presence
    info["project_count"] = len(model.by_type("IfcProject"))
    # Units
    units = []
    for ua in model.by_type("IfcUnitAssignment"):
        if getattr(ua, "Units", None):
            for u in ua.Units:
                units.append(u.is_a())
    info["units_sample"] = units[:20]
    # counts
    info["total_IfcRoot"] = len(model.by_type("IfcRoot"))
    # count IfcProduct (and those with actual Items)
    prod_total = 0
    prod_with_repr = 0
    prod_with_items = 0
    for e in model.by_type("IfcProduct"):
        prod_total += 1
        rep = getattr(e, "Representation", None)
        if rep and getattr(rep, "Representations", None):
            prod_with_repr += 1
            got_items = False
            for r in rep.Representations:
                items = getattr(r, "Items", None)
                if items and len(items) > 0:
                    got_items = True
                    break
            if got_items:
                prod_with_items += 1
    info["IfcProduct_total"] = prod_total
    info["IfcProduct_with_representation"] = prod_with_repr
    info["IfcProduct_with_items"] = prod_with_items
    # low-level geometry types
    info["IfcShapeRepresentation_count"] = len(model.by_type("IfcShapeRepresentation"))
    info["IfcRepresentationMap_count"] = len(model.by_type("IfcRepresentationMap"))
    # bbox via IfcCartesianPoint (fast heuristic)
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    count_pts = 0
    for p in model.by_type("IfcCartesianPoint"):
        coords = getattr(p, "Coordinates", None)
        if not coords:
            continue
        # normalize to 3
        x = float(coords[0]) if len(coords) > 0 else 0.0
        y = float(coords[1]) if len(coords) > 1 else 0.0
        z = float(coords[2]) if len(coords) > 2 else 0.0
        count_pts += 1
        mins[0] = min(mins[0], x)
        mins[1] = min(mins[1], y)
        mins[2] = min(mins[2], z)
        maxs[0] = max(maxs[0], x)
        maxs[1] = max(maxs[1], y)
        maxs[2] = max(maxs[2], z)
    info["cartesian_point_count"] = count_pts
    if count_pts:
        info["bbox_min"] = mins
        info["bbox_max"] = maxs
        info["bbox_span"] = [maxs[i] - mins[i] for i in range(3)]
    else:
        info["bbox_min"] = None
        info["bbox_max"] = None
        info["bbox_span"] = None
    # collect GlobalIds for duplicate detection (returns map of gid->count)
    gids = defaultdict(int)
    for e in model.by_type("IfcRoot"):
        g = getattr(e, "GlobalId", None)
        if g:
            gids[str(g)] += 1
    info["globalid_count_unique"] = len(gids)
    info["globalid_duplicates"] = sum(1 for v in gids.values() if v > 1)
    # basic top types
    type_counter = Counter()
    for e in model.by_type("IfcRoot"):
        type_counter[e.is_a()] += 1
    info["top_types"] = type_counter.most_common(20)
    return info

def preflight(paths):
    results = []
    # for cross-file duplicate detection
    global_gid_map = defaultdict(list)
    for p in paths:
        print("Inspecting", p)
        info = inspect_file(Path(p))
        results.append(info)
    # cross-file duplicates
    for r in results:
        # reopen to list GIDs (lightweight)
        model = ifcopenshell.open(r["path"])
        for e in model.by_type("IfcRoot"):
            g = getattr(e, "GlobalId", None)
            if g:
                global_gid_map[str(g)].append(r["path"])
    cross_dups = {g:files for g,files in global_gid_map.items() if len(files) > 1}
    # recommendations
    recommendations = []
    # sizes: sum sizes
    total_size = sum(r["size_bytes"] for r in results)
    if total_size > 1_000_000_000:
        recommendations.append({
            "code":"BIG_FILE",
            "message":"Total input size > 1GB: Autodesk Viewer may reject upload. Consider splitting/federation or use server-side processing."
        })
    # global duplicate recommendation
    if cross_dups:
        recommendations.append({
            "code":"GLOBALID_DUPLICATES",
            "message":f"Found {len(cross_dups)} GlobalId values present in multiple files. Consider remapping duplicates (ifcpatch RegenerateGlobalIds) or handle manually.",
            "sample": dict(list(cross_dups.items())[:10])
        })
    # bbox/coords check
    for r in results:
        span = r.get("bbox_span")
        if span:
            big = max(abs(v) for v in (r["bbox_min"] + r["bbox_max"]))
            if big > 1e6 or any(v > 1e6 for v in span):
                recommendations.append({"code":"LARGE_COORDS","file":r["path"], "message":"Coordinates or extents very large: georeferenced coordinates far from origin. Consider IfcMapConversion or recentre before merge."})
            if all(v < 1e-3 for v in span):
                recommendations.append({"code":"SMALL_EXTENT","file":r["path"], "message":"Model extent very small -> possible units mismatch (mm vs m). Consider unit check or rescale."})
    # geometry presence
    for r in results:
        if r.get("IfcProduct_with_items",0) == 0:
            recommendations.append({"code":"NO_GEOM","file":r["path"], "message":"No IfcProduct with representation items found — verify export settings of source."})
    # schema consistency
    schemas = set(r["schema"] for r in results)
    if len(schemas) > 1:
        recommendations.append({"code":"MIXED_SCHEMA","message":f"Multiple IFC schemas detected: {list(schemas)}. Consider standardizing to a single target schema before merge."})
    report = {"files":results, "cross_file_globalid_duplicates_count": len(cross_dups), "cross_file_globalid_sample": dict(list(cross_dups.items())[:20]), "recommendations": recommendations, "total_input_size_bytes": total_size}
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="directory with ifc files")
    parser.add_argument("--files", nargs="*", help="explicit ifc files")
    parser.add_argument("--output", default="output/preflight_report.json")
    args = parser.parse_args()
    paths = []
    if args.input_dir:
        p = Path(args.input_dir)
        paths += sorted([str(x) for x in p.glob("*.ifc")])
    if args.files:
        paths += args.files
    if not paths:
        print("No input files found")
        sys.exit(1)
    report = preflight(paths)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("Wrote report to", outp)
    # quick print recommendations
    print("\nRecommendations:")
    for r in report["recommendations"]:
        print(" -", r.get("code"), "-", r.get("message"))
    print("\nDone.")

if __name__ == "__main__":
    main()
