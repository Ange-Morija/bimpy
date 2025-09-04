"""Inspector for a single IFC file.

Exposes a single function `inspect_ifc(path)` that returns a JSON-serializable
report summarizing counts, geometry presence, basic bbox heuristic and simple
recommendations.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json

try:
    import ifcopenshell
except Exception:  # pragma: no cover
    ifcopenshell = None


def inspect_ifc(ifc_path: str) -> dict:
    """Return a serializable summary for `ifc_path`.

    The function is defensive: it raises a RuntimeError if `ifcopenshell` is not
    available.
    """
    if ifcopenshell is None:
        raise RuntimeError("ifcopenshell is required")
    p = Path(ifc_path)
    model = ifcopenshell.open(str(p))
    report = {
        "file": str(p),
        "schema": getattr(model, "schema", None),
        "size_bytes": p.stat().st_size,
    }

    # counts
    report["total_IfcRoot"] = len(model.by_type("IfcRoot"))
    report["IfcShapeRepresentation_count"] = len(model.by_type("IfcShapeRepresentation"))
    report["IfcRepresentationMap_count"] = len(model.by_type("IfcRepresentationMap"))

    # products and geometry
    prod_total = 0
    prod_with_repr = 0
    prod_with_items = 0
    for e in model.by_type("IfcProduct"):
        prod_total += 1
        rep = getattr(e, "Representation", None)
        if rep and getattr(rep, "Representations", None):
            prod_with_repr += 1
            for r in rep.Representations:
                items = getattr(r, "Items", None)
                if items and len(items) > 0:
                    prod_with_items += 1
                    break

    report["IfcProduct_total"] = prod_total
    report["IfcProduct_with_representation"] = prod_with_repr
    report["IfcProduct_with_items"] = prod_with_items

    # bbox heuristic via IfcCartesianPoint (fast)
    xs = []
    ys = []
    zs = []
    for pt in model.by_type("IfcCartesianPoint")[:10000]:
        coords = getattr(pt, "Coordinates", None)
        if not coords:
            continue
        if len(coords) > 0:
            xs.append(float(coords[0]))
        if len(coords) > 1:
            ys.append(float(coords[1]))
        if len(coords) > 2:
            zs.append(float(coords[2]))
    if xs:
        report["bbox_min"] = [min(xs), min(ys) if ys else None, min(zs) if zs else None]
        report["bbox_max"] = [max(xs), max(ys) if ys else None, max(zs) if zs else None]
    else:
        report["bbox_min"] = report["bbox_max"] = None

    # GlobalId duplicates
    gid_map = defaultdict(int)
    for e in model.by_type("IfcRoot"):
        g = getattr(e, "GlobalId", None)
        if g:
            gid_map[str(g)] += 1
    report["globalid_total_unique"] = len(gid_map)
    report["globalid_duplicates_count"] = sum(1 for v in gid_map.values() if v > 1)

    # top types
    counter = Counter()
    for e in model.by_type("IfcRoot"):
        try:
            counter[e.is_a()] += 1
        except Exception:
            continue
    report["top_types"] = counter.most_common(20)

    # simple recommendations
    recs = []
    if report["IfcProduct_with_items"] == 0:
        recs.append({"code": "NO_GEOM", "message": "No product with geometry items found."})
    if report["globalid_duplicates_count"] > 0:
        recs.append({"code": "GLOBALID_DUP", "message": "Duplicate GlobalIds inside file."})
    report["recommendations"] = recs

    return report


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('ifc', nargs='?', default='output/merged.ifc')
    ap.add_argument('--out', dest='out', default=None)
    args = ap.parse_args()
    r = inspect_ifc(args.ifc)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)
        print('Wrote', args.out)
    else:
        print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
