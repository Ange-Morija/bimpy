# functions/diagnose_ifc.py
import sys
import json
from pathlib import Path
import ifcopenshell
from collections import Counter

def analyze(ifc_path: Path):
    f = ifcopenshell.open(str(ifc_path))
    out = {}
    out['schema'] = f.schema
    # counts
    out['IfcShapeRepresentation_count'] = len(f.by_type('IfcShapeRepresentation'))
    out['IfcProductDefinitionShape_count'] = len(f.by_type('IfcProductDefinitionShape'))
    out['IfcRepresentationMap_count'] = len(f.by_type('IfcRepresentationMap'))
    out['IfcGeometricRepresentationContext_count'] = len(f.by_type('IfcGeometricRepresentationContext'))
    # products: how many have representation items
    products = f.by_type('IfcProduct')
    with_repr = 0
    with_items = 0
    sample_no_repr = []
    for p in products:
        rep = getattr(p, 'Representation', None)
        if rep:
            with_repr += 1
            # try count items
            try:
                for sr in rep.Representations:
                    if getattr(sr, 'Items', None):
                        with_items += 1
                        break
            except Exception:
                pass
        else:
            if len(sample_no_repr) < 20:
                sample_no_repr.append({'type': p.is_a(), 'GlobalId': getattr(p, 'GlobalId', None), 'Name': getattr(p, 'Name', None)})
    out['total_IfcRoot'] = sum(1 for _ in f.by_type('IfcRoot'))
    out['products_total'] = len(products)
    out['products_with_representation'] = with_repr
    out['products_with_items_sample_no_repr'] = sample_no_repr
    # bbox via IfcCartesianPoint (quick & approximate)
    points = f.by_type('IfcCartesianPoint')
    coords = []
    for pt in points[:10000]:
        try:
            coords.append(list(map(float, pt.Coordinates)))
        except Exception:
            pass
    if coords:
        xs = [c[0] for c in coords if len(c)>0]
        ys = [c[1] for c in coords if len(c)>1]
        zs = [c[2] for c in coords if len(c)>2]
        if xs:
            out['bbox_sample_min'] = [min(xs), min(ys) if ys else None, min(zs) if zs else None]
            out['bbox_sample_max'] = [max(xs), max(ys) if ys else None, max(zs) if zs else None]
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_ifc.py path/to/file.ifc [out.json]")
        sys.exit(1)
    path = Path(sys.argv[1])
    report = analyze(path)
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print("Wrote report to", sys.argv[2])
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
