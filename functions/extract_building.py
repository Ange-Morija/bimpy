# functions/extract_zone_with_geom.py
import logging
from pathlib import Path
import argparse
import ifcopenshell
from tqdm import tqdm
import sys

LOG_PATH = Path("output") / "extract_zone.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


def open_model(ifc_path: Path):
    logging.info(f"Opening IFC: {ifc_path}")
    if not ifc_path.exists():
        raise FileNotFoundError(ifc_path)
    return ifcopenshell.open(str(ifc_path))


def find_space(model, query: str):
    q = query.lower()
    logging.info(f"Searching IfcSpace for query: {query}")
    for s in model.by_type("IfcSpace"):
        name = (s.LongName or s.Name or "").lower()
        if q in name:
            logging.info(f"Found IfcSpace: GlobalId={s.GlobalId} Name={s.Name} LongName={s.LongName}")
            return s
    return None


def collect_products_in_space(model, space):
    # collect products belonging to space by inverse IfcRelContainedInSpatialStructure
    products = set()
    for inv in model.get_inverse(space):
        if inv.is_a("IfcRelContainedInSpatialStructure"):
            for p in getattr(inv, "RelatedElements", []) or []:
                products.add(p)
    logging.info(f"Found {len(products)} products in the space")
    return list(products)


def ensure_contexts_copied(orig, new):
    # copy geometric contexts and representation maps first
    for ctx in orig.by_type("IfcGeometricRepresentationContext"):
        new.add(ctx)
    for rmap in orig.by_type("IfcRepresentationMap"):
        new.add(rmap)


def add_product_and_geometry(orig, new, product):
    """
    Add product + its direct representation items and inverses.
    We explicitly add:
      - product (new.add(product))
      - product.Representation -> each IfcShapeRepresentation -> add and add Items
      - inverse relationships (e.g. relationships that reference the product)
    """
    # add product (this will attempt to copy related entities recursively)
    new.add(product)

    # explicit: ensure product's representation items are present
    prod_rep = getattr(product, "Representation", None)
    if prod_rep:
        reps = getattr(prod_rep, "Representations", []) or []
        for rep in reps:
            new.add(rep)
            items = getattr(rep, "Items", []) or []
            for it in items:
                new.add(it)

    # copy inverse relationships (helpful to keep relations like IfcRelAggregates/Associates)
    for inv in orig.get_inverse(product):
        new.add(inv)


def extract_zone_to_file(ifc_input: Path, query: str, out_ifc: Path):
    model = open_model(ifc_input)
    space = find_space(model, query)
    if not space:
        logging.error("No IfcSpace found for query")
        raise SystemExit(1)

    products = collect_products_in_space(model, space)
    if len(products) == 0:
        logging.warning("No products in zone; file may appear empty in viewer.")

    # create new target file with same schema
    new_model = ifcopenshell.file(schema=model.schema)
    # copy header basics
    try:
        new_model.assign_header_from(model)
    except Exception:
        logging.debug("assign_header_from failed or not available; continuing")

    # copy project (header) minimal
    project = model.by_type("IfcProject")
    if project:
        new_model.add(project[0])

    # copy geometic contexts & maps first (important)
    ensure_contexts_copied(model, new_model)

    # add the space itself (so spatial container exists)
    new_model.add(space)

    # progress bar for products
    logging.info("Collecte des produits assignés à la zone (analyse des relations inverses)...")
    for p in tqdm(products, desc="Copying products", unit="prod"):
        try:
            add_product_and_geometry(model, new_model, p)
        except Exception as e:
            logging.exception(f"Failed to copy product {getattr(p,'GlobalId',str(p))}: {e}")

    # also copy any spatial containment inverses (storey, building) so tree exists
    for inv in model.get_inverse(space):
        new_model.add(inv)  # e.g. IfcRelContainedInSpatialStructure

    out_ifc.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing extracted IFC to: {out_ifc}")
    new_model.write(str(out_ifc))
    logging.info("Done.")
    return out_ifc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", default="input/merged.ifc", help="Input merged IFC")
    parser.add_argument("--query", required=True, help="Zone query (partial LongName/Name)")
    parser.add_argument("--out", default="output/zone_extract.ifc", help="Output small IFC path")
    args = parser.parse_args()

    try:
        outp = extract_zone_to_file(Path(args.ifc), args.query, Path(args.out))
        logging.info(f"Extraction done -> {outp}")
    except Exception as e:
        logging.exception("Extraction échouée:")
        raise


if __name__ == "__main__":
    main()
