"""Inject sensors into an IFC from a simple JSON file.

Expected JSON format: a list of objects with keys: deviceId, x, y, z, space_globalid (optional), type...
"""
from pathlib import Path
import json
from typing import Optional

try:
    import ifcopenshell
    import ifcopenshell.guid
except Exception:  # pragma: no cover
    ifcopenshell = None


def _new_guid():
    return ifcopenshell.guid.new()


def sensor_exists(model, device_id: str) -> bool:
    for s in model.by_type('IfcSensor'):
        if getattr(s, 'Name', None) == device_id:
            return True
    return False


def attach_to_space(model, owner_hist, sensor, space_gid: Optional[str]):
    if not space_gid:
        return None
    for sp in model.by_type('IfcSpace'):
        if getattr(sp, 'GlobalId', None) == space_gid:
            rel = model.create_entity('IfcRelContainedInSpatialStructure', _new_guid(), owner_hist, None, None, [sensor], sp)
            return rel
    return None


def create_sensor(model, owner_hist, item: dict):
    gid = _new_guid()
    kwargs = {'GlobalId': gid}
    if owner_hist:
        kwargs['OwnerHistory'] = owner_hist
    name = item.get('deviceId') or item.get('name')
    if name:
        kwargs['Name'] = name
    x = item.get('x'); y = item.get('y'); z = item.get('z')
    if x is not None and y is not None and z is not None:
        pt = model.create_entity('IfcCartesianPoint', [float(x), float(y), float(z)])
        ap = model.create_entity('IfcAxis2Placement3D', pt)
        lp = model.create_entity('IfcLocalPlacement', None, ap)
        kwargs['ObjectPlacement'] = lp
    stype = item.get('type') or item.get('stype')
    if stype:
        kwargs['PredefinedType'] = stype
    sensor = model.create_entity('IfcSensor', **kwargs)
    return sensor


def inject_from_json(json_path: str, ifc_in: str, ifc_out: str) -> dict:
    if ifcopenshell is None:
        raise RuntimeError('ifcopenshell required')
    model = ifcopenshell.open(str(ifc_in))
    owner_hist = model.by_type('IfcOwnerHistory')
    owner_hist = owner_hist[0] if owner_hist else None

    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('JSON must be a list of sensor objects')
    report = {'created': [], 'skipped': [], 'errors': []}
    for item in data:
        deviceId = item.get('deviceId') or item.get('name')
        if not deviceId:
            report['skipped'].append({'reason': 'no_id', 'item': item})
            continue
        if sensor_exists(model, deviceId):
            report['skipped'].append({'reason': 'exists', 'deviceId': deviceId})
            continue
        try:
            sensor = create_sensor(model, owner_hist, item)
            attach_to_space(model, owner_hist, sensor, item.get('space_globalid') or item.get('space'))
            report['created'].append({'deviceId': deviceId, 'GlobalId': sensor.GlobalId})
        except Exception as e:
            report['errors'].append({'item': item, 'error': str(e)})

    Path(ifc_out).parent.mkdir(parents=True, exist_ok=True)
    model.write(ifc_out)
    return report


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('json', nargs='?', default='sensors.json')
    ap.add_argument('ifc_in', nargs='?', default='output/merged.ifc')
    ap.add_argument('ifc_out', nargs='?', default='output/with_sensors.ifc')
    args = ap.parse_args()
    r = inject_from_json(args.json, args.ifc_in, args.ifc_out)
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
