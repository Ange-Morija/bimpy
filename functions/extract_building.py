"""Extract a spatial zone (IfcSpace) and its related products into a new IFC file.

This version is a compact, well-documented implementation that tries to copy
only the necessary geometry contexts, the space, and products contained in it.
"""
from pathlib import Path
from typing import List
try:
    import ifcopenshell
    import ifcopenshell.util.element as element_util
    import ifcopenshell.guid
except Exception:  # pragma: no cover - dependency
    ifcopenshell = None


def find_space(model, query: str):
    q = query.lower().strip()
    for s in model.by_type('IfcSpace'):
        name = (getattr(s, 'LongName', None) or getattr(s, 'Name', None) or '').lower()
        if q in name:
            return s
    # fallback: building/storey
    for t in ('IfcBuildingStorey', 'IfcBuilding'):
        for e in model.by_type(t):
            if getattr(e, 'Name', None) and q in str(e.Name).lower():
                return e
    return None


def collect_products_in_space(model, space) -> List:
    prods = set()
    for inv in model.get_inverse(space):
        if inv.is_a('IfcRelContainedInSpatialStructure'):
            for p in getattr(inv, 'RelatedElements', []) or []:
                prods.add(p)
    return list(prods)


def extract_zone_to_file(ifc_input: Path, query: str, out_ifc: Path) -> Path:
    if ifcopenshell is None:
        raise RuntimeError('ifcopenshell required')
    src = ifcopenshell.open(str(ifc_input))
    target = find_space(src, query)
    if not target:
        raise RuntimeError('Zone not found')
    products = collect_products_in_space(src, target)

    new = ifcopenshell.file(schema=src.schema)
    # copy minimal header/project
    projects = src.by_type('IfcProject')
    if projects:
        new.add(projects[0])
    # copy geometric contexts and maps
    for c in src.by_type('IfcGeometricRepresentationContext'):
        new.add(c)
    for m in src.by_type('IfcRepresentationMap'):
        new.add(m)

    # add the spatial container and related inverses
    new.add(target)
    for inv in src.get_inverse(target):
        new.add(inv)

    for p in products:
        try:
            new.add(p)
            # ensure representation items are present
            rep = getattr(p, 'Representation', None)
            if rep:
                for r in getattr(rep, 'Representations', []) or []:
                    new.add(r)
                    for it in getattr(r, 'Items', []) or []:
                        new.add(it)
            for inv in src.get_inverse(p):
                new.add(inv)
        except Exception:
            # best-effort copy
            continue

    out_ifc.parent.mkdir(parents=True, exist_ok=True)
    new.write(str(out_ifc))
    return out_ifc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ifc', default='input/merged.ifc')
    ap.add_argument('--query', required=True)
    ap.add_argument('--out', default='output/zone_extract.ifc')
    args = ap.parse_args()
    p = extract_zone_to_file(Path(args.ifc), args.query, Path(args.out))
    print('Wrote', p)


if __name__ == '__main__':
    main()
