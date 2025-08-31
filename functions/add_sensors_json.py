#!/usr/bin/env python3
"""
add_sensors_json.py

Injecte des IfcSensor dans output/merged.ifc à partir d'un fichier sensors.json.
Crée un PSet basique "Pset_SensorInfo", optionnellement rattache à IfcSpace via IfcRelContainedInSpatialStructure.
Génère un log (output/sensor_injection.log) et un rapport JSON (output/sensor_report.json).
"""

import json
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import ifcopenshell
import ifcopenshell.guid
from datetime import datetime

# ---- Config par défaut (modifiable) ----
IFC_INPUT = Path("output/malaga_try.ifc")
IFC_OUTPUT = Path("output/malaga_with_sensors.ifc")
SENSORS_JSON_DEFAULT = Path("sensors.json")
LOG_PATH = Path("output/sensor_injection.log")
REPORT_PATH = Path("output/sensor_report.json")
# ----------------------------------------

# Setup logging (file + console)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def new_guid():
    return ifcopenshell.guid.new()


def find_owner_history(model):
    oh = model.by_type("IfcOwnerHistory")
    return oh[0] if oh else None


def create_local_placement(model, x, y, z):
    """
    Crée un IfcLocalPlacement simple basé sur IfcAxis2Placement3D
    Retourne l'entité IfcLocalPlacement
    """
    # create point and placement (Axis2Placement3D + LocalPlacement)
    pt = model.create_entity("IfcCartesianPoint", [float(x), float(y), float(z)])
    ap = model.create_entity("IfcAxis2Placement3D", pt)
    lp = model.create_entity("IfcLocalPlacement", None, ap)
    return lp


def create_pset_for_sensor(model, owner_history, sensor_obj, deviceId, stype=None, manufacturer=None, installation_height=None, orientation=None, notes=None):
    """
    Crée un IfcPropertySet Pset_SensorInfo et la relie au sensor via IfcRelDefinesByProperties
    """
    props = []
    def mk_label(val):
        # IfcLabel entity
        return model.create_entity("IfcLabel", str(val))

    def mk_double(val):
        return model.create_entity("IfcReal", float(val))

    if deviceId:
        props.append(model.create_entity("IfcPropertySingleValue", "DeviceId", None, mk_label(deviceId)))
    if stype:
        props.append(model.create_entity("IfcPropertySingleValue", "SensorType", None, mk_label(stype)))
    if manufacturer:
        props.append(model.create_entity("IfcPropertySingleValue", "Manufacturer", None, mk_label(manufacturer)))
    if installation_height is not None:
        props.append(model.create_entity("IfcPropertySingleValue", "InstallationHeight", None, mk_double(installation_height)))
    if orientation is not None:
        props.append(model.create_entity("IfcPropertySingleValue", "Orientation", None, mk_double(orientation)))
    if notes:
        props.append(model.create_entity("IfcPropertySingleValue", "Notes", None, mk_label(notes)))

    if not props:
        return None

    pset = model.create_entity("IfcPropertySet", None, None, None, props)
    rel = model.create_entity("IfcRelDefinesByProperties", new_guid(), owner_history, None, None, [sensor_obj], pset)
    return rel


def attach_sensor_to_space(model, owner_history, sensor_obj, space_gid):
    """
    Crée IfcRelContainedInSpatialStructure si l'espace existe (recherche par GlobalId).
    Retourne l'objet relation ou None.
    """
    if not space_gid:
        return None
    for s in model.by_type("IfcSpace"):
        try:
            if getattr(s, "GlobalId", "") == space_gid:
                rel = model.create_entity("IfcRelContainedInSpatialStructure", new_guid(), owner_history, None, None, [sensor_obj], s)
                return rel
        except Exception:
            continue
    return None


def sensor_exists_by_deviceid(model, device_id):
    """
    Cherche un IfcSensor existant lié à ce DeviceId dans d'éventuels Psets (simple heuristique).
    """
    # parcours des IfcSensor existants : on regarde les IfcRelDefinesByProperties inverses
    for s in model.by_type("IfcSensor"):
        # check Name first
        try:
            if getattr(s, "Name", "") == device_id:
                return True
        except Exception:
            pass
    # fallback: cherche DeviceId dans PropertySets
    for rel in model.by_type("IfcRelDefinesByProperties"):
        try:
            pset = rel.RelatingPropertyDefinition
            if pset and pset.is_a("IfcPropertySet"):
                for prop in pset.HasProperties:
                    try:
                        if prop.Name == "DeviceId" and str(getattr(prop, "NominalValue", "")) == device_id:
                            return True
                    except Exception:
                        pass
        except Exception:
            pass
    return False


def main(json_path: Path = None, ifc_in: Path = None, ifc_out: Path = None):
    json_path = Path(json_path) if json_path else SENSORS_JSON_DEFAULT
    ifc_in = Path(ifc_in) if ifc_in else IFC_INPUT
    ifc_out = Path(ifc_out) if ifc_out else IFC_OUTPUT

    if not ifc_in.exists():
        logging.error(f"Input IFC not found: {ifc_in}")
        sys.exit(1)
    if not json_path.exists():
        logging.error(f"Sensors JSON not found: {json_path}")
        sys.exit(1)

    logging.info(f"Opening IFC: {ifc_in}")
    model = ifcopenshell.open(str(ifc_in))

    owner_hist = find_owner_history(model)
    if not owner_hist:
        logging.warning("No IfcOwnerHistory found — sensors will be created without owner history (may break some validators).")

    # read JSON
    with open(json_path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
            if not isinstance(data, list):
                logging.error("Expected JSON top-level to be a list of sensor objects.")
                sys.exit(1)
        except json.JSONDecodeError as e:
            logging.exception("Invalid JSON file")
            sys.exit(1)

    total = len(data)
    logging.info(f"Found {total} sensor(s) in JSON. Processing...")
    created = 0
    skipped = 0
    errors = 0
    report = {"timestamp": datetime.utcnow().isoformat() + "Z", "input_ifc": str(ifc_in), "output_ifc": str(ifc_out), "total_requested": total, "created": [], "skipped": [], "errors": []}

    for item in tqdm(data, desc="Sensors", unit="sensor"):
        try:
            deviceId = item.get("deviceId")
            x = item.get("x")
            y = item.get("y")
            z = item.get("z")
            space_gid = item.get("space_globalid") or item.get("spaceGlobalId") or item.get("space_gid")
            stype = item.get("stype") or item.get("type")
            manufacturer = item.get("manufacturer")
            installation_height = item.get("installation_height")
            orientation = item.get("orientation")
            notes = item.get("notes")

            if not deviceId:
                logging.warning(f"Skipping sensor (missing deviceId): {item}")
                skipped += 1
                report["skipped"].append({"reason": "missing_deviceId", "item": item})
                continue

            # avoid duplicating sensors with same deviceId (heuristic)
            if sensor_exists_by_deviceid(model, deviceId):
                logging.info(f"Sensor with deviceId '{deviceId}' seems to already exist: skipping.")
                skipped += 1
                report["skipped"].append({"reason": "already_exists", "deviceId": deviceId})
                continue

            gid = new_guid()
            placement = None
            if (x is not None) and (y is not None) and (z is not None):
                placement = create_local_placement(model, x, y, z)

            # create IfcSensor (OwnerHistory optional)
            # IfcSensor entity has attributes: GlobalId, OwnerHistory, Name, Description, ObjectType, ObjectPlacement, Representation, Tag, PredefinedType...
            sensor_kwargs = {}
            sensor_kwargs["GlobalId"] = gid
            if owner_hist:
                sensor_kwargs["OwnerHistory"] = owner_hist
            sensor_kwargs["Name"] = deviceId
            if placement:
                sensor_kwargs["ObjectPlacement"] = placement
            if stype:
                sensor_kwargs["PredefinedType"] = stype

            sensor = model.create_entity("IfcSensor", **sensor_kwargs)

            # create PSet and rel
            pset_rel = create_pset_for_sensor(model, owner_hist, sensor, deviceId, stype, manufacturer, installation_height, orientation, notes)

            # attach to space if provided (relation)
            rel = None
            if space_gid:
                rel = attach_sensor_to_space(model, owner_hist, sensor, space_gid)
                if not rel:
                    logging.warning(f"Space with GlobalId '{space_gid}' not found for sensor '{deviceId}'. Relation not created.")
            # record in report
            created += 1
            report["created"].append({"deviceId": deviceId, "GlobalId": gid, "has_placement": bool(placement), "attached_to_space": bool(rel)})
            logging.debug(f"Created sensor '{deviceId}' (GUID={gid}) placed={bool(placement)} space_link={bool(rel)}")
        except Exception as e:
            logging.exception(f"Failed to create sensor for item: {item}")
            errors += 1
            report["errors"].append({"item": item, "error": str(e)})

    # write out IFC and report
    logging.info(f"Created {created} sensor(s), skipped {skipped}, errors {errors}. Writing output IFC to {ifc_out} ...")
    ifc_out.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(ifc_out))
    logging.info("IFC write complete.")

    # write report
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logging.info(f"Report written to {REPORT_PATH}")

    logging.info("Done.")


if __name__ == "__main__":
    # optional CLI args: script.py [sensors.json] [input_ifc] [output_ifc]
    args = sys.argv[1:]
    js = args[0] if len(args) > 0 else None
    ifc_in = args[1] if len(args) > 1 else None
    ifc_out = args[2] if len(args) > 2 else None
    main(js, ifc_in, ifc_out)
