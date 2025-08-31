#!/usr/bin/env python3
"""
Recenter merged IFC by subtracting centroid (min+max)/2 from all IfcCartesianPoint coordinates.

Usage:
  python functions/recenter_merged.py --input output/merged.ifc --output output/merged_recentred.ifc

Dependencies:
  pip install ifcopenshell tqdm
"""
import argparse, logging, traceback
from pathlib import Path
import math
from tqdm import tqdm

try:
    import ifcopenshell
except Exception as e:
    raise RuntimeError("ifcopenshell required: pip install ifcopenshell") from e

def setup_logging(logfile: Path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(str(logfile), mode="w", encoding="utf-8"),
                  logging.StreamHandler()]
    )
    logging.info("Logging to %s", logfile)

def compute_bbox(model):
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    count = 0
    for p in model.by_type("IfcCartesianPoint"):
        coords = getattr(p, "Coordinates", None)
        if not coords:
            continue
        # pad to 3
        x = float(coords[0]) if len(coords) > 0 else 0.0
        y = float(coords[1]) if len(coords) > 1 else 0.0
        z = float(coords[2]) if len(coords) > 2 else 0.0
        count += 1
        mins[0] = min(mins[0], x); mins[1] = min(mins[1], y); mins[2] = min(mins[2], z)
        maxs[0] = max(maxs[0], x); maxs[1] = max(maxs[1], y); maxs[2] = max(maxs[2], z)
    if count == 0:
        return None
    return {"min": mins, "max": maxs, "count": count}

def recenter(input_path: Path, output_path: Path, logfile: Path):
    setup_logging(logfile)
    logging.info("Opening IFC: %s", input_path)
    model = ifcopenshell.open(str(input_path))
    bbox = compute_bbox(model)
    if not bbox:
        logging.error("No IfcCartesianPoint found. Aborting recenter.")
        return False
    mins = bbox["min"]; maxs = bbox["max"]
    cx = (mins[0] + maxs[0]) / 2.0
    cy = (mins[1] + maxs[1]) / 2.0
    cz = (mins[2] + maxs[2]) / 2.0
    logging.info("BBox min=%s max=%s pts=%d", mins, maxs, bbox["count"])
    logging.info("Centroid computed: (%.3f, %.3f, %.3f). Subtracting from all IfcCartesianPoint.", cx, cy, cz)

    changed = 0
    pts = list(model.by_type("IfcCartesianPoint"))
    for p in tqdm(pts, desc="Recentring CartesianPoints", unit="pt"):
        try:
            coords = list(getattr(p, "Coordinates", []))
            if not coords:
                continue
            # ensure three components
            while len(coords) < 3:
                coords.append(0.0)
            newx = float(coords[0]) - cx
            newy = float(coords[1]) - cy
            newz = float(coords[2]) - cz
            # set back (tuple)
            try:
                p.Coordinates = (newx, newy, newz)
            except Exception:
                try:
                    setattr(p, "Coordinates", (newx, newy, newz))
                except Exception:
                    logging.debug("Failed to write coords for point: %s", p)
                    continue
            changed += 1
        except Exception:
            logging.debug("Exception adjusting point: %s", traceback.format_exc())

    logging.info("Adjusted %d CartesianPoints", changed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(output_path))
    logging.info("Wrote recentered IFC to: %s", output_path)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/merged.ifc")
    parser.add_argument("--output", default="output/merged_recentred.ifc")
    parser.add_argument("--log", default="output/recenter.log")
    args = parser.parse_args()
    inp = Path(args.input); outp = Path(args.output); logp = Path(args.log)
    if not inp.exists():
        print("Input IFC not found:", inp); return
    # backup prompt: we don't overwrite original
    print("Backup recommended: cp", inp, str(inp)+".bak")
    ok = recenter(inp, outp, logp)
    if ok:
        print("Done. Upload", outp, "to viewer to test.")

if __name__ == "__main__":
    main()
