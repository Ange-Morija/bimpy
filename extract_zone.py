# extract_zone_fixed.py
import logging
from pathlib import Path
import ifcopenshell
import ifcpatch
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def find_space_or_building(model, query: str):
    q = (query or "").lower()
    # search IfcSpace by LongName / Name
    for s in model.by_type("IfcSpace"):
        longname = getattr(s, "LongName", None) or ""
        name = getattr(s, "Name", None) or ""
        if q in longname.lower() or q in name.lower():
            return ("IfcSpace", s)
    # fallback: IfcBuilding
    for b in model.by_type("IfcBuilding"):
        name = getattr(b, "Name", "") or ""
        if q in name.lower():
            return ("IfcBuilding", b)
    return (None, None)

def extract_zone_to_file(ifc_in: Path, query: str, out_file: Path, log_file: Path | None = None):
    logging.info("Opening IFC: %s", ifc_in)
    model = ifcopenshell.open(str(ifc_in))

    kind, target = find_space_or_building(model, query)
    if target is None:
        raise RuntimeError(f"No IfcSpace or IfcBuilding matching '{query}' found.")

    gid = getattr(target, "GlobalId", None)
    logging.info("Found target: type=%s, GlobalId=%s, Name=%s, LongName=%s",
                 kind, gid, getattr(target, "Name", ""), getattr(target, "LongName", ""))

    # build selector string for ifcpatch
    selector = f"{kind}[GlobalId='{gid}']"
    logging.info("Using selector for extraction: %s", selector)

    params = {
        "input": str(ifc_in),
        "file": model,                      # <--- important: pass the opened model object
        "recipe": "ExtractElements",
        "arguments": [selector],
        "log": str(log_file) if log_file else None,
        "output": str(out_file)              # optional, but we still call ifcpatch.write below
    }

    logging.info("Executing ifcpatch ExtractElements recipe (this preserves geometry and related items)...")
    try:
        output = ifcpatch.execute(params)
        # write the output to disk
        ifcpatch.write(output, str(out_file))
        logging.info("Extraction written to: %s", out_file)
    except Exception as e:
        logging.exception("Extraction échouée:")
        raise

    # quick validation: reopen written file and check IfcShapeRepresentation counts
    out_model = ifcopenshell.open(str(out_file))
    shape_count = len(out_model.by_type("IfcShapeRepresentation"))
    logging.info("Post-check: IfcShapeRepresentation count in output: %d", shape_count)
    if shape_count == 0:
        logging.warning("Aucune IfcShapeRepresentation trouvée dans l'export -> Autodesk Viewer pourra indiquer 'modèle vide'.")

    return out_file

def main():
    import argparse
    p = argparse.ArgumentParser(description="Extract zone/building to a new IFC using ifcpatch ExtractElements (fixed).")
    p.add_argument("-i", "--ifc", required=True, help="Input IFC file (path)")
    p.add_argument("-q", "--query", required=True, help="Recherche (ex: 'ZONA DE EMBARQUE')")
    p.add_argument("-o", "--out", required=True, help="IFC de sortie")
    p.add_argument("-l", "--log", help="log file (optionnel)")
    args = p.parse_args()

    try:
        extract_zone_to_file(Path(args.ifc), args.query, Path(args.out), Path(args.log) if args.log else None)
        logging.info("Extraction terminée.")
    except Exception as e:
        logging.error("Erreur: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
