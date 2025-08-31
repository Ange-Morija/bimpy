# functions/extract_zone_fixed.py
import sys
from pathlib import Path
import ifcopenshell
import ifcopenshell.util.element as element_util
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def find_space(model, query):
    q = query.lower()
    for s in model.by_type('IfcSpace'):
        name = (getattr(s, 'Name', '') or '').lower()
        longname = (getattr(s, 'LongName', '') or '').lower()
        if q in name or q in longname:
            return s
    # fallback: search IfcBuilding name
    for b in model.by_type('IfcBuilding'):
        if q in (getattr(b, 'Name', '') or '').lower():
            return b
    return None

def create_new_file_like(src_model):
    # create new file with same schema version
    try:
        new = ifcopenshell.file(schema_version=src_model.schema)
    except Exception:
        # fallback: use schema_version attribute
        new = ifcopenshell.file(schema_version=src_model.schema_version)
    return new

def minimal_spatial_setup(new_model, source_space):
    # create minimal spatial elements (IfcProject/IfcSite/IfcBuilding/IfcBuildingStorey/IfcSpace copy)
    # we will copy the space entity itself using copy_deep so relationships stay consistent.
    # create project, site, building, storey in the new file
    project = new_model.create_entity('IfcProject', GlobalId=ifcopenshell.guid.new(), Name='Extracted project')
    site = new_model.create_entity('IfcSite', GlobalId=ifcopenshell.guid.new(), Name='Extracted site')
    building = new_model.create_entity('IfcBuilding', GlobalId=ifcopenshell.guid.new(), Name='Extracted building')
    storey = new_model.create_entity('IfcBuildingStorey', GlobalId=ifcopenshell.guid.new(), Name='Extracted storey')
    # set up aggregation relationships
    new_model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
    new_model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])
    new_model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=[storey])
    # copy the space (deep copy) and attach to storey using IfcRelContainedInSpatialStructure
    copied_space = element_util.copy_deep(new_model, source_space)
    new_model.create_entity('IfcRelContainedInSpatialStructure', GlobalId=ifcopenshell.guid.new(), RelatingStructure=storey, RelatedElements=[copied_space])
    return project, site, building, storey, copied_space

def collect_products_for_space(space):
    products = set()
    # search inverses of space
    for inv in space.get_inverse():
        # many inverse types; check related elements attributes
        for attr in dir(inv):
            try:
                val = getattr(inv, attr)
                if isinstance(val, list):
                    for v in val:
                        if hasattr(v, 'is_a') and v.is_a() and v.is_a().startswith('Ifc') and v.is_a() != 'IfcSpace':
                            products.add(v)
                else:
                    if hasattr(val, 'is_a') and val.is_a() and val.is_a().startswith('Ifc') and val.is_a() != 'IfcSpace':
                        products.add(val)
            except Exception:
                continue
    # fallback: if no inverses found, search for IfcRelContainedInSpatialStructure referencing this space by globalid
    # but above loop usually finds them.
    return list(products)

def extract_zone_copy_deep(ifc_in: Path, query: str, ifc_out: Path):
    src = ifcopenshell.open(str(ifc_in))
    target = find_space(src, query)
    if not target:
        raise RuntimeError(f"Target not found for query: {query}")
    logging.info(f"Found target: type={target.is_a()}, GlobalId={target.GlobalId}, Name={getattr(target,'Name',None)}, LongName={getattr(target,'LongName',None)}")

    products = collect_products_for_space(target)
    logging.info(f"Found {len(products)} products assigned to the space (via inverses).")

    new = create_new_file_like(src)
    # minimal header + spatial structure and copy space
    project, site, building, storey, copied_space = minimal_spatial_setup(new, target)
    # copy each product deep into new model and attach to copied_space or storey (depending original)
    for i, p in enumerate(products, start=1):
        logging.info(f"Copying product {i}/{len(products)}: {p.is_a()} {getattr(p,'GlobalId',None)}")
        try:
            cp = element_util.copy_deep(new, p)
            # attach product to storey (or to the space)
            try:
                new.create_entity('IfcRelContainedInSpatialStructure', GlobalId=ifcopenshell.guid.new(), RelatingStructure=storey, RelatedElements=[cp])
            except Exception:
                pass
        except Exception as e:
            logging.warning("copy_deep failed for product: %s - %s", getattr(p,'GlobalId',None), e)

    # write
    ifc_out.parent.mkdir(parents=True, exist_ok=True)
    new.write(str(ifc_out))
    logging.info("Wrote extracted file to %s", str(ifc_out))

def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_zone_fixed.py input.ifc \"QUERY\" output.ifc")
        sys.exit(1)
    extract_zone_copy_deep(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))

if __name__ == '__main__':
    main()
