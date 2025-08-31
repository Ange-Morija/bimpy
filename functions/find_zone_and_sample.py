# find_zone_and_sample.py
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.guid
from pathlib import Path
import logging
import random
from tqdm import tqdm
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def open_ifc(path):
    logger.info(f"Opening IFC: {path}")
    return ifcopenshell.open(str(path))

def find_zone(model, query):
    q = query.lower().strip()
    # 1) try GUID if the query looks like a GUID (22 chars base64 IFC)
    if len(q) == 22:
        try:
            ent = model.by_guid(q)
            logger.info(f"Found by GUID: {ent}")
            return ent
        except Exception:
            pass
    # 2) search IfcZone
    for z in model.by_type('IfcZone'):
        for attr in ('Name', 'LongName', 'Tag', 'Description', 'ObjectType'):
            val = getattr(z, attr, None)
            if val and q in str(val).lower():
                logger.info(f"Found IfcZone by {attr}: {val}")
                return z
    # 3) search IfcSpace
    for s in model.by_type('IfcSpace'):
        for attr in ('Name', 'LongName', 'Tag', 'Description', 'ObjectType'):
            val = getattr(s, attr, None)
            if val and q in str(val).lower():
                logger.info(f"Found IfcSpace by {attr}: {val}")
                return s
    # 4) fallback: search any spatial element (BuildingStorey/Building) by name substring
    for t in ('IfcBuildingStorey', 'IfcBuilding', 'IfcSite', 'IfcProject'):
        for e in model.by_type(t):
            val = getattr(e, 'Name', None)
            if val and q in str(val).lower():
                logger.info(f"Found {t} by Name: {val}")
                return e
    logger.warning("Zone not found by heuristics.")
    return None

def collect_products_in_zone(model, zone_entity):
    """Collect IfcProducts that are contained/assigned to the zone via IfcRelContainedInSpatialStructure / inverses."""
    products = set()
    # iterate all products and check their inverse relations
    all_products = model.by_type('IfcProduct')
    for p in tqdm(all_products, desc="Scanning products", unit="prod"):
        try:
            inverses = model.get_inverse(p)
        except Exception:
            inverses = []
        for inv in inverses:
            if inv.is_a('IfcRelContainedInSpatialStructure') and getattr(inv, 'RelatingStructure', None) is zone_entity:
                products.add(p)
    logger.info(f"Collected {len(products)} products assigned to zone.")
    return list(products)

def compute_bbox_of_products(products):
    """Naive bbox: iterate products, request geometry with ifcopenshell.geom.create_shape (world coords),
       collect vertices min/max. This can be slow for many entities."""
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    minx = miny = minz = float('inf')
    maxx = maxy = maxz = float('-inf')
    count = 0
    for p in tqdm(products, desc="Tessellating products", unit="ent"):
        try:
            shape = ifcopenshell.geom.create_shape(settings, p)
            verts = shape.geometry.verts  # flat list [x0,y0,z0, x1,y1,z1, ...]
            for i in range(0, len(verts), 3):
                x,y,z = verts[i], verts[i+1], verts[i+2]
                minx, miny, minz = min(minx,x), min(miny,y), min(minz,z)
                maxx, maxy, maxz = max(maxx,x), max(maxy,y), max(maxz,z)
            count += 1
        except Exception as e:
            # ignore entities failing to tessellate
            continue
    if count == 0:
        raise RuntimeError("No geometry available to compute bbox.")
    return (minx, miny, minz), (maxx, maxy, maxz)

def sample_positions_in_bbox(bbox_min, bbox_max, n=10):
    minx,miny,minz = bbox_min
    maxx,maxy,maxz = bbox_max
    positions = []
    for _ in range(n):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        z = random.uniform(minz, maxz)
        positions.append((x,y,z))
    return positions

def main(ifc_path, query, n_samples=10):
    model = open_ifc(ifc_path)
    zone = find_zone(model, query)
    if zone is None:
        logger.error("Zone not found — abort.")
        return
    products = collect_products_in_zone(model, zone)
    if not products:
        logger.warning("No products linked to zone — falling back to geometry of the zone entity itself (if any).")
        products = [zone]
    bbox_min, bbox_max = compute_bbox_of_products(products)
    logger.info(f"Zone bbox_min={bbox_min} bbox_max={bbox_max}")
    positions = sample_positions_in_bbox(bbox_min, bbox_max, n=n_samples)
    for i,p in enumerate(positions, 1):
        logger.info(f"Sample {i}: {p}")
    return {
        "zone": getattr(zone, 'Name', str(zone)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "positions": positions
    }

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ifc', default='input/merged.ifc')
    ap.add_argument('--query', required=True, help='mot clé pour la zone (ex: "ZONA DE EMBARQUE")')
    ap.add_argument('--n', type=int, default=20)
    args = ap.parse_args()
    res = main(args.ifc, args.query, n_samples=args.n)
    print(res)
