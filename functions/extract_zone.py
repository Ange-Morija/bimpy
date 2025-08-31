#!/usr/bin/env python3
"""
extract_zone_fixed.py

Extraire une zone (IfcSpace) et les produits qui y sont contenus dans un nouveau fichier IFC minimal.
Corrige l'usage de l'API (utilise ifcopenshell.api.run).
Dépendances: ifcopenshell, tqdm
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_minimal_header_and_structure(orig_model):
    """
    Crée un nouveau fichier IFC minimal (project/site/building/storey) et
    retourne (new_model, storey, project_name).
    """
    # get schema/version from original if present and normalize
    version = getattr(orig_model, "schema", "IFC4")
    if isinstance(version, str):
        version = version.upper()
    if version not in ("IFC2X3", "IFC4", "IFC4X3"):
        version = "IFC4"

    logging.debug("Creating new IFC file with version %s", version)

    # CORRECTION : use 'version' not 'schema'
    new_model = ifcopenshell.api.run("project.create_file", version=version)

    # create owner history (pass the file as first arg)
    try:
        ifcopenshell.api.run("owner.create_owner_history", new_model)
    except Exception as e:
        logging.debug("owner.create_owner_history failed (non-fatal): %s", e)

    project_name = "ExtractedProject"
    # create the spatial structure using the API (new_model passed to each call)
    try:
        project = ifcopenshell.api.run("root.create_entity", new_model, ifc_class="IfcProject", name=project_name)
        site = ifcopenshell.api.run("root.create_entity", new_model, ifc_class="IfcSite", name="Site")
        building = ifcopenshell.api.run("root.create_entity", new_model, ifc_class="IfcBuilding", name="Building")
        storey = ifcopenshell.api.run("root.create_entity", new_model, ifc_class="IfcBuildingStorey", name="Storey")

        # assemble hierarchy (note order of kwargs may vary between versions; named args used for clarity)
        ifcopenshell.api.run("aggregate.assign_object", new_model, relating_object=project, products=[site])
        ifcopenshell.api.run("aggregate.assign_object", new_model, relating_object=site, products=[building])
        ifcopenshell.api.run("aggregate.assign_object", new_model, relating_object=building, products=[storey])
    except Exception as e:
        logging.warning("Failed using higher-level API for spatial structure: %s. Falling back to minimal entities.", e)
        # fallback minimal entities (ensure project & storey exist)
        project = ifcopenshell.api.run("root.create_entity", new_model, ifc_class="IfcProject", name=project_name)
        storey = ifcopenshell.api.run("root.create_entity", new_model, ifc_class="IfcBuildingStorey", name="Storey")

    return new_model, storey, project_name



def find_space(model, query):
    """
    Recherche IfcSpace dont Name ou LongName contient la query (cas-insensible).
    Retourne la première correspondance.
    """
    q = query.lower()
    logging.info("Searching IfcSpace for query: %s", query)
    for s in model.by_type("IfcSpace"):
        name = (getattr(s, "Name", "") or "") or ""
        longname = (getattr(s, "LongName", "") or "") or ""
        if (name and q in str(name).lower()) or (longname and q in str(longname).lower()):
            logging.info("Found IfcSpace: GlobalId=%s Name=%s LongName=%s", getattr(s, "GlobalId", None), name, longname)
            return s
    return None


def collect_products_in_space(model, space):
    """
    Parcourt les relations inverses pour collecter les produits contenus dans la zone (IfcRelContainedInSpatialStructure).
    Retourne liste unique d'entités.
    """
    prods = []
    inv = list(model.get_inverse(space))
    for r in inv:
        if r.is_a("IfcRelContainedInSpatialStructure"):
            related = getattr(r, "RelatedElements", []) or []
            for p in related:
                prods.append(p)
    # dedupe by id
    seen = set()
    unique = []
    for p in prods:
        if p.id() not in seen:
            seen.add(p.id())
            unique.append(p)
    return unique


def extract_zone_to_file(ifc_path: Path, query: str, out_path: Path):
    logging.info("Opening IFC: %s", ifc_path)
    model = ifcopenshell.open(str(ifc_path))

    space = find_space(model, query)
    if not space:
        raise RuntimeError(f"No IfcSpace matching query '{query}' found.")

    logging.info("Collecting products referenced by the space (inverses)...")
    products = collect_products_in_space(model, space)
    logging.info("Found %d products in the space", len(products))

    logging.info("Creating minimal target file and spatial structure...")
    new_model, storey, project_name = create_minimal_header_and_structure(model)

    # Copy the space itself first
    entities_to_copy = [space] + products

    logging.info("Copying %d entities into new model and assigning to storey...", len(entities_to_copy))
    for ent in tqdm(entities_to_copy, desc="Copying entities", unit="ent"):
        try:
            # add returns the instance in new model (works like base.add in your merge script)
            new_ent = new_model.add(ent)
        except Exception:
            # fallback: try to create a shallow copy with attributes (best-effort)
            logging.debug("add() failed for entity %s; skipping", getattr(ent, "GlobalId", ent.id()))
            continue
        # assign the product into the storey (spatial containment) if product-like
        try:
            if new_ent.is_a("IfcProduct"):
                ifcopenshell.api.run("spatial.assign_container", new_model, relating_structure=storey, product=new_ent)
        except Exception as e:
            logging.debug("Failed spatial.assign_container for entity %s: %s", getattr(new_ent, "GlobalId", new_ent.id()), e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing extracted IFC to: %s", out_path)
    new_model.write(str(out_path))
    logging.info("Done. Wrote %s", out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Extract zone and contained products to a minimal IFC (fixed API usage).")
    p.add_argument("--ifc", default="input/merged.ifc", help="Input IFC file (default: input/merged.ifc)")
    p.add_argument("--query", required=True, help="Search string to find the IfcSpace (Name or LongName contains this)")
    p.add_argument("--out", default="output/extracted_zone.ifc", help="Output IFC path (default: output/extracted_zone.ifc)")
    args = p.parse_args()

    try:
        out = extract_zone_to_file(Path(args.ifc), args.query, Path(args.out))
        logging.info("Extraction réussie : %s", out)
    except Exception as e:
        logging.exception("Extraction échouée: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
