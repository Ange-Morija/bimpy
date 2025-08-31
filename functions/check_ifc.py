#!/usr/bin/env python3
"""
functions/check_output.py

But :
 - Ouvrir un fichier IFC (par défaut output/merged.ifc)
 - Vérifier présence de géométrie, compter IfcRoot, IfcProduct avec Representation.Items>0
 - Calculer bbox via IfcCartesianPoint
 - Lister top types (top 30)
 - Détecter duplicatas GlobalId (cross-file not needed here)
 - Afficher barres de progression (tqdm)
 - Écrire un log (output/check_output.log) et un rapport JSON (output/check_report.json)
 - Assurer que tout est JSON-serializable (transforme objets IFC en chaînes/dicts légers)

Usage:
  python functions/check_output.py --ifc output/merged.ifc --report output/check_report.json

Dépendances:
  pip install ifcopenshell tqdm
"""
import argparse
import json
import logging
import traceback
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

try:
    import ifcopenshell
except Exception as e:
    raise RuntimeError("ifcopenshell is required. Install with: pip install ifcopenshell") from e

# ---------------- helpers ----------------
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging to file: %s", log_path)

def make_serializable(obj):
    """
    Convertir obj en quelque chose JSON-serializable.
    - Types primitifs pass through
    - Lists/tuples/dicts récursivement
    - Tout objet IFC / non-serializable -> str(obj) + s'ils ont GlobalId/is_a on inclut info utile
    """
    # Primitifs
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Containers
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    # Try to detect Ifc entity-ish (duck typing)
    try:
        is_a = getattr(obj, "is_a", None)
        gid = getattr(obj, "GlobalId", None)
        if callable(is_a) or gid is not None:
            # Return a small informative dict
            try:
                t = obj.is_a() if callable(is_a) else str(type(obj).__name__)
            except Exception:
                t = str(type(obj).__name__)
            return {"__ifc_type": t, "GlobalId": make_serializable(gid), "repr": str(obj)}
    except Exception:
        pass
    # Fallback: try to convert to primitive (e.g., decimal), else str()
    try:
        return str(obj)
    except Exception:
        return None

# ---------------- checks ----------------
def inspect_ifc(ifc_path: Path, show_progress=True):
    logging.info("Opening IFC: %s", ifc_path)
    model = ifcopenshell.open(str(ifc_path))

    report = {}
    report["file"] = str(ifc_path)
    try:
        report["schema"] = getattr(model, "schema", None)
    except Exception:
        report["schema"] = None

    report["size_bytes"] = ifc_path.stat().st_size if ifc_path.exists() else None

    # Count IfcRoot total (we need total to drive progress)
    try:
        total_roots = len(model.by_type("IfcRoot"))
    except Exception:
        total_roots = None

    report["total_IfcRoot"] = total_roots

    # Iterate IfcRoot and compute:
    # - type counts
    # - count of products with representation/items > 0
    # - collect GlobalId duplicates (per file)
    type_counter = Counter()
    products_with_items = 0
    products_with_any_representation = 0
    gid_counter = defaultdict(int)

    # To avoid building huge lists we iterate and update progress bar
    roots = model.by_type("IfcRoot")
    iterator = roots
    if show_progress and total_roots:
        iterator = tqdm(roots, total=total_roots, desc="Entities", unit="ent")

    for e in iterator:
        try:
            t = e.is_a()
        except Exception:
            t = str(type(e).__name__)
        type_counter[t] += 1
        # GlobalId
        try:
            g = getattr(e, "GlobalId", None)
            if g:
                gid_counter[str(g)] += 1
        except Exception:
            pass
        # IfcProduct geometry check (we accept any subclass of IfcProduct)
        try:
            # Some IfcRoot are not IfcProduct; using is_a string is ok
            if t.startswith("IfcProduct"):
                rep = getattr(e, "Representation", None)
                if rep and getattr(rep, "Representations", None):
                    products_with_any_representation += 1
                    # check for items inside representations
                    items_found = False
                    for r in rep.Representations:
                        items = getattr(r, "Items", None)
                        if items and len(items) > 0:
                            items_found = True
                            break
                    if items_found:
                        products_with_items += 1
            else:
                # For many subtypes like IfcWall/IfcBeam - they are also IfcProduct subclasses
                # Some libraries return their exact type; we also check by membership in known product types
                if t in ("IfcWall","IfcBeam","IfcSlab","IfcColumn","IfcDoor","IfcWindow","IfcCovering","IfcFurniture","IfcPlate"):
                    rep = getattr(e, "Representation", None)
                    if rep and getattr(rep, "Representations", None):
                        products_with_any_representation += 1
                        items_found = False
                        for r in rep.Representations:
                            items = getattr(r, "Items", None)
                            if items and len(items) > 0:
                                items_found = True
                                break
                        if items_found:
                            products_with_items += 1
        except Exception:
            # ignore per-entity failures, just log debug
            logging.debug("Entity check failed for type %s: %s", t, traceback.format_exc())

    report["top_types"] = type_counter.most_common(30)
    report["IfcProduct_with_representation_count"] = products_with_any_representation
    report["IfcProduct_with_items_count"] = products_with_items
    # GlobalId duplicates in this file
    report["globalid_total_unique"] = len(gid_counter)
    report["globalid_duplicates_count"] = sum(1 for v in gid_counter.values() if v > 1)
    # Sample duplicates (max 50)
    dup_sample = [gid for gid, c in gid_counter.items() if c > 1][:50]
    report["globalid_duplicates_sample"] = dup_sample

    # Low-level geometry counts (fast)
    try:
        report["IfcShapeRepresentation_count"] = len(model.by_type("IfcShapeRepresentation"))
        report["IfcRepresentationMap_count"] = len(model.by_type("IfcRepresentationMap"))
        report["IfcGeometricRepresentationContext_count"] = len(model.by_type("IfcGeometricRepresentationContext"))
    except Exception:
        report["IfcShapeRepresentation_count"] = report["IfcRepresentationMap_count"] = report["IfcGeometricRepresentationContext_count"] = None

    # Compute bbox via IfcCartesianPoint (heuristic)
    mins = [None, None, None]
    maxs = [None, None, None]
    cp_count = 0
    try:
        cps = model.by_type("IfcCartesianPoint")
        # don't show a progress bar for points (may be many), but we iterate safely
        for p in tqdm(cps, desc="CartesianPoints", unit="pt", leave=False):
            coords = getattr(p, "Coordinates", None)
            if not coords:
                continue
            # pad coords to 3
            x = float(coords[0]) if len(coords) > 0 else 0.0
            y = float(coords[1]) if len(coords) > 1 else 0.0
            z = float(coords[2]) if len(coords) > 2 else 0.0
            cp_count += 1
            if mins[0] is None or x < mins[0]: mins[0] = x
            if mins[1] is None or y < mins[1]: mins[1] = y
            if mins[2] is None or z < mins[2]: mins[2] = z
            if maxs[0] is None or x > maxs[0]: maxs[0] = x
            if maxs[1] is None or y > maxs[1]: maxs[1] = y
            if maxs[2] is None or z > maxs[2]: maxs[2] = z
    except Exception:
        logging.debug("CartesianPoint bbox calc failed: %s", traceback.format_exc())

    report["cartesian_point_count"] = cp_count
    if cp_count:
        report["bbox_min"] = mins
        report["bbox_max"] = maxs
        report["bbox_span"] = [ (maxs[i] - mins[i]) if mins[i] is not None and maxs[i] is not None else None for i in range(3) ]
    else:
        report["bbox_min"] = report["bbox_max"] = report["bbox_span"] = None

    # Units / IfcProject
    try:
        units = []
        for ua in model.by_type("IfcUnitAssignment"):
            if getattr(ua, "Units", None):
                for u in ua.Units:
                    units.append(u.is_a())
        report["units_sample"] = units[:20]
    except Exception:
        report["units_sample"] = None

    try:
        projects = model.by_type("IfcProject")
        if projects:
            proj = projects[0]
            report["project_present"] = True
            report["project_name"] = getattr(proj, "Name", None)
        else:
            report["project_present"] = False
            report["project_name"] = None
    except Exception:
        report["project_present"] = None
        report["project_name"] = None

    return report

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Check IFC in output and produce a JSON report + log")
    parser.add_argument("--ifc", default="output/merged.ifc", help="Path to IFC to check (default: output/merged.ifc)")
    parser.add_argument("--report", default="output/check_report.json", help="Path to write JSON report")
    parser.add_argument("--log", default="output/check_output.log", help="Log file path")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    ifc_path = Path(args.ifc)
    report_path = Path(args.report)
    log_path = Path(args.log)

    setup_logging(log_path)
    logging.info("Starting check for: %s", ifc_path)

    try:
        if not ifc_path.exists():
            logging.error("IFC file not found: %s", ifc_path)
            print("IFC file not found:", ifc_path)
            return

        report = inspect_ifc(ifc_path, show_progress=not args.no_progress)

        # Add recommendations simple heuristics
        recs = []
        # if no geometry products -> warn
        if report.get("IfcProduct_with_items_count", 0) == 0:
            recs.append("Aucune entité produit avec Representation.Items>0 trouvée -> vérifier export source ou test subset.")
        # bbox checks
        span = report.get("bbox_span")
        if span and any(v is not None and v > 1e6 for v in span):
            recs.append("Extents très grands (>1e6). Le viewer peut perdre le modèle. Recentrer ou utiliser IfcMapConversion.")
        if span and all(v is not None and v < 1e-3 for v in span):
            recs.append("Extents très petits (<1e-3). Possibilité de mismatch d'unités (mm vs m).")
        # globalid duplicates
        if report.get("globalid_duplicates_count",0) > 0:
            recs.append(f"Duplicats GlobalId dans le fichier: {report['globalid_duplicates_count']}. Envisager remap.")

        report["recommendations"] = recs

        # Ensure JSON serializable by transforming values
        safe_report = make_serializable(report)

        # write report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(safe_report, fh, indent=2, ensure_ascii=False)

        logging.info("Wrote report: %s", report_path)
        print("Report written to:", report_path)
    except Exception as exc:
        logging.exception("Fatal error during check: %s", exc)
        # write partial error report
        err_report = {"error": str(exc), "traceback": traceback.format_exc()}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(err_report, fh, indent=2, ensure_ascii=False)
        print("Error occurred; see log for details:", log_path)

if __name__ == "__main__":
    main()
