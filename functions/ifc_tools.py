from __future__ import annotations

# --- Imports standard ---------------------------------------------------------
import json
import re
import unicodedata
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Union, Tuple

from glob import glob
import pandas as pd

# --- IfcOpenShell -------------------------------------------------------------
try:
    import ifcopenshell
    from ifcopenshell.entity_instance import entity_instance
    from ifcopenshell import guid as ifc_guid
    try:
        # Copie profonde d’un sous-arbre (0.7+)
        from ifcopenshell.util.element import copy_deep as _copy_deep
    except Exception:
        _copy_deep = None
except Exception as e:
    raise RuntimeError("IfcOpenShell n'est pas installé ou ne se charge pas correctement.") from e

# --- tqdm (optionnel) ---------------------------------------------------------
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# --- Chemins par défaut -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is not None:
        return logger
    log = logging.getLogger("ifc_tools")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


# --- Ouverture & infos --------------------------------------------------------
def open_ifc(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return ifcopenshell.open(str(path))


def _safe_entity_count(model):
    try:
        return int(len(model))
    except Exception:
        pass
    try:
        return len(model.by_type("IfcRoot"))
    except Exception:
        try:
            return sum(1 for _ in model)
        except Exception:
            return None


def get_project_info(model) -> Dict[str, str]:
    proj_list = model.by_type("IfcProject")
    proj = proj_list[0] if proj_list else None
    schema_attr = getattr(model, "schema", None)
    schema = str(schema_attr() if callable(schema_attr) else schema_attr) if schema_attr else "Unknown"
    num = _safe_entity_count(model)

    return {
        "FileName": getattr(model, "path", "") or "",
        "Schema": schema,
        "ProjectName": getattr(proj, "Name", None) or "",
        "Description": getattr(proj, "Description", None) or "",
        "NumEntities": "" if num is None else str(num),
    }


def count_by_class(model) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ent in set(e.is_a() for e in model):
        try:
            out[ent] = len(model.by_type(ent))
        except Exception:
            out[ent] = sum(1 for e in model if e.is_a(ent))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


# --- Arbre spatial ------------------------------------------------------------
def _children(obj: entity_instance) -> Iterable[entity_instance]:
    for rel in getattr(obj, "IsDecomposedBy", []) or []:
        for child in getattr(rel, "RelatedObjects", []) or []:
            yield child


def _contained_elements(container: entity_instance) -> Iterable[entity_instance]:
    for rel in getattr(container, "ContainsElements", []) or []:
        for elem in getattr(rel, "RelatedElements", []) or []:
            yield elem


def get_spatial_tree(model) -> Dict:
    proj_list = model.by_type("IfcProject")
    if not proj_list:
        return {}
    proj = proj_list[0]

    def walk(node):
        data = {
            "type": node.is_a(),
            "name": getattr(node, "Name", None),
            "global_id": getattr(node, "GlobalId", None),
            "children": [],
            "elements": [],
        }
        for e in _contained_elements(node):
            data["elements"].append({
                "type": e.is_a(),
                "name": getattr(e, "Name", None),
                "global_id": getattr(e, "GlobalId", None),
            })
        for ch in _children(node):
            data["children"].append(walk(ch))
        return data

    return walk(proj)


# --- Extraction d'éléments ----------------------------------------------------
def _make_storey_index(model) -> Dict[int, str]:
    idx: Dict[int, str] = {}
    for storey in model.by_type("IfcBuildingStorey"):
        name = getattr(storey, "Name", None) or ""
        for elem in _contained_elements(storey):
            idx[elem.id()] = name
    return idx


def _get_pset_value(elem: entity_instance, pset: str, prop: str):
    for rel in getattr(elem, "IsDefinedBy", []) or []:
        p = getattr(rel, "RelatingPropertyDefinition", None)
        if p and p.is_a("IfcPropertySet") and getattr(p, "Name", None) == pset:
            for prop_inst in getattr(p, "HasProperties", []) or []:
                if getattr(prop_inst, "Name", None) == prop:
                    return getattr(prop_inst, "NominalValue", None) or getattr(prop_inst, "Description", None)
    return None


def extract_elements(
    model,
    classes: Optional[List[str]] = None,
    psets: Optional[List[str]] = None,
) -> pd.DataFrame:
    if classes:
        targets: List[entity_instance] = []
        for c in classes:
            targets.extend(model.by_type(c))
    else:
        default = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam",
                   "IfcDoor", "IfcWindow", "IfcStair", "IfcRoof"]
        targets = []
        for d in default:
            targets.extend(model.by_type(d))

    storey_idx = _make_storey_index(model)

    rows = []
    for e in targets:
        row = {
            "GlobalId": getattr(e, "GlobalId", None),
            "Name": getattr(e, "Name", None),
            "Type": e.is_a(),
            "Storey": storey_idx.get(e.id(), ""),
        }
        for attr in ("Tag", "PredefinedType", "ObjectType"):
            if hasattr(e, attr):
                row[attr] = getattr(e, attr)
        if psets:
            for spec in psets:
                if ":" in spec:
                    pset, prop = spec.split(":", 1)
                    row[f"{pset}.{prop}"] = _get_pset_value(e, pset, prop)
        rows.append(row)

    return pd.DataFrame(rows)


# --- Exports ------------------------------------------------------------------
def export_dataframe(df: pd.DataFrame, name: str, fmt: str = "csv") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUTS_DIR / f"{name}_{ts}.{fmt}"
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "json":
        df.to_json(out, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError("Format non supporté (utilise 'csv' ou 'json').")
    return out


# --- Batch utilities ----------------------------------------------------------
def find_ifc_files(path_like: Union[str, Path]) -> List[Path]:
    p = Path(path_like)
    candidates: List[Path] = []
    if p.is_dir():
        candidates.extend(sorted(p.rglob("*.ifc")))
        candidates.extend(sorted(p.rglob("*.ifczip")))
    elif p.exists():
        candidates = [p]
    else:
        for g in sorted(glob(str(p))):
            gp = Path(g)
            if gp.suffix.lower() in [".ifc", ".ifczip"]:
                candidates.append(gp)
    if not candidates:
        raise FileNotFoundError(f"Aucun fichier IFC trouvé pour: {path_like}")
    return candidates


def project_info_row(model, file_path: Path) -> Dict[str, str]:
    info = get_project_info(model)
    info["File"] = file_path.name
    return info


def counts_to_dataframe(model, file_path: Path) -> pd.DataFrame:
    counts = count_by_class(model)
    return pd.DataFrame(
        [{"File": file_path.name, "Class": k, "Count": v} for k, v in counts.items()]
    )


def extract_elements_for_file(
    model, file_path: Path, classes: Optional[List[str]] = None, psets: Optional[List[str]] = None
) -> pd.DataFrame:
    df = extract_elements(model, classes=classes, psets=psets)
    if not df.empty:
        df.insert(0, "File", file_path.name)
    return df


def batch_process(
    inputs: Path | str,
    classes: Optional[List[str]] = None,
    psets: Optional[List[str]] = None,
    write_trees: bool = False,
    out_prefix: str = "batch",
) -> Dict[str, Path]:
    files = find_ifc_files(inputs)

    all_infos, all_counts, all_elements = [], [], []
    tree_paths: List[Path] = []

    for f in tqdm(files, desc="Traitement IFC", unit="fichier"):
        model = open_ifc(f)
        all_infos.append(project_info_row(model, f))
        all_counts.append(counts_to_dataframe(model, f))
        all_elements.append(extract_elements_for_file(model, f, classes=classes, psets=psets))

        if write_trees:
            tree = get_spatial_tree(model)
            out_json = OUTPUTS_DIR / f"{f.stem}_tree.json"
            out_json.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
            tree_paths.append(out_json)

    info_df = pd.DataFrame(all_infos).sort_values("File")
    counts_df = pd.concat(all_counts, ignore_index=True).sort_values(["Class", "File"])
    elems_df = (
        pd.concat([df for df in all_elements if not df.empty], ignore_index=True)
        if any(len(df) for df in all_elements)
        else pd.DataFrame()
    )

    info_csv = export_dataframe(info_df, f"{out_prefix}_info")
    counts_csv = export_dataframe(counts_df, f"{out_prefix}_counts")
    elems_csv = export_dataframe(elems_df, f"{out_prefix}_elements") if not elems_df.empty else None

    return {
        "info_csv": info_csv,
        "counts_csv": counts_csv,
        "elements_csv": elems_csv or Path(""),
        "trees_dir": OUTPUTS_DIR,
    }


# --- Bâtiments ----------------------------------------------------------------
def extract_buildings(model, select_id: Optional[str] = None, select_name: Optional[str] = None) -> pd.DataFrame:
    buildings = model.by_type("IfcBuilding")
    rows: List[dict] = []

    def _addr_to_str(addr) -> str:
        if not addr:
            return ""
        parts = []
        if getattr(addr, "AddressLines", None):
            parts += [line for line in (addr.AddressLines or []) if line]
        for f in ("PostalBox", "Town", "Region", "PostalCode", "Country"):
            v = getattr(addr, f, None)
            if v:
                parts.append(v)
        return " ".join(parts)

    for b in buildings:
        gid = getattr(b, "GlobalId", "")
        name = getattr(b, "Name", "") or ""
        if select_id and gid != select_id:
            continue
        if select_name and select_name.lower() not in name.lower():
            continue

        storeys = []
        for rel in getattr(b, "IsDecomposedBy", []) or []:
            for obj in getattr(rel, "RelatedObjects", []) or []:
                if obj.is_a("IfcBuildingStorey"):
                    storeys.append(obj)

        rows.append({
            "GlobalId": gid,
            "Name": name,
            "LongName": getattr(b, "LongName", None),
            "CompositionType": getattr(b, "CompositionType", None),
            "ElevationOfRefHeight": getattr(b, "ElevationOfRefHeight", None),
            "ElevationOfTerrain": getattr(b, "ElevationOfTerrain", None),
            "NumStoreys": len(storeys),
            "StoreyNames": ", ".join([(getattr(s, "Name", None) or "") for s in storeys]),
            "Address": _addr_to_str(getattr(b, "BuildingAddress", None)),
        })

    return pd.DataFrame(rows)


# --- Helpers géométrie / contexts / placements -------------------------------
def _has_geometry(model) -> bool:
    try:
        return bool(model.by_type("IfcGeometricRepresentationContext")) and \
               bool(model.by_type("IfcProductDefinitionShape"))
    except Exception:
        return False


def _pick_geometry_base(files: List[Path]) -> Path:
    for f in files:
        try:
            m = ifcopenshell.open(str(f))
            if _has_geometry(m):
                return f
        except Exception:
            pass
    return files[0]


def _attach_under_project(base: "ifcopenshell.file", obj: entity_instance):
    """
    Attache 'obj' (Site/Building/…) sous l'IfcProject via un IfcRelAggregates.
    Gère les schémas où OwnerHistory est présent (6 args) ou omis (5 args).
    """
    dst_proj = (base.by_type("IfcProject") or [None])[0]
    if not dst_proj:
        return

    gid = ifc_guid.new()
    # Essai 1 : schémas avec OwnerHistory (6 paramètres)
    try:
        base.create_entity(
            "IfcRelAggregates",
            gid,
            None,          # OwnerHistory (IFC2x3 / certaines builds IFC4)
            None,          # Name
            None,          # Description
            dst_proj,      # RelatingObject
            [obj],         # RelatedObjects
        )
        return
    except TypeError:
        pass

    # Essai 2 : schémas sans OwnerHistory (5 paramètres)
    base.create_entity(
        "IfcRelAggregates",
        gid,
        None,          # Name
        None,          # Description
        dst_proj,      # RelatingObject
        [obj],         # RelatedObjects
    )


def _iter_root_containers(src: "ifcopenshell.file") -> List[entity_instance]:
    """Racines spatiales, bâtiments en priorité (évite les soucis copy_deep sur IfcSite)."""
    roots: List[entity_instance] = []
    proj = (src.by_type("IfcProject") or [None])[0]
    if proj:
        for rel in getattr(proj, "IsDecomposedBy", []) or []:
            roots.extend(getattr(rel, "RelatedObjects", []) or [])
    if not roots:
        roots = src.by_type("IfcSite") + src.by_type("IfcBuilding")

    buildings = [r for r in roots if hasattr(r, "is_a") and r.is_a("IfcBuilding")]
    sites = [r for r in roots if hasattr(r, "is_a") and r.is_a("IfcSite")]

    return buildings + sites


def _iter_children_of_type(node: entity_instance, type_name: str) -> Iterable[entity_instance]:
    for rel in getattr(node, "IsDecomposedBy", []) or []:
        for ch in getattr(rel, "RelatedObjects", []) or []:
            if ch.is_a(type_name):
                yield ch


def _attach_contexts_to_project(model: "ifcopenshell.file") -> None:
    proj = (model.by_type("IfcProject") or [None])[0]
    if not proj:
        return
    existing = list(proj.RepresentationContexts or [])
    all_ctx = list(model.by_type("IfcRepresentationContext"))
    to_add = [c for c in all_ctx if c not in existing]
    if to_add:
        proj.RepresentationContexts = tuple(existing + to_add)


def _ensure_minimal_context(model: "ifcopenshell.file") -> None:
    if model.by_type("IfcGeometricRepresentationContext"):
        return
    proj = (model.by_type("IfcProject") or [None])[0]
    if not proj:
        return
    p0 = model.create_entity("IfcCartesianPoint", (0.0, 0.0, 0.0))
    wcs = model.create_entity("IfcAxis2Placement3D", p0, None, None)
    ctx = model.create_entity(
        "IfcGeometricRepresentationContext",
        None, "Model", 3, 1e-5, wcs, None
    )
    proj.RepresentationContexts = tuple(list(proj.RepresentationContexts or []) + [ctx])


def _ensure_minimal_units(model: "ifcopenshell.file") -> None:
    proj = (model.by_type("IfcProject") or [None])[0]
    if not proj:
        return
    if getattr(proj, "UnitsInContext", None):
        return
    length = model.create_entity("IfcSIUnit", None, "LENGTHUNIT", None, "METRE")
    unit_assign = model.create_entity("IfcUnitAssignment", [length])
    proj.UnitsInContext = unit_assign


# --- Décalage global (rebase) ------------------------------------------------
def _get_or_create_local_placement(model: "ifcopenshell.file", ent: entity_instance):
    lp = getattr(ent, "ObjectPlacement", None)
    if lp and lp.is_a("IfcLocalPlacement"):
        return lp
    # créer un placement local à (0,0,0)
    p0 = model.create_entity("IfcCartesianPoint", (0.0, 0.0, 0.0))
    axis = model.create_entity("IfcAxis2Placement3D", p0, None, None)
    lp = model.create_entity("IfcLocalPlacement", None, None, axis)
    ent.ObjectPlacement = lp
    return lp


def _force_relative_axis_3d(model: "ifcopenshell.file", lp: entity_instance):
    axis = getattr(lp, "RelativePlacement", None)
    if axis and axis.is_a("IfcAxis2Placement3D"):
        return axis
    # convertir/forcer en 3D
    loc = None
    if axis and hasattr(axis, "Location"):
        loc = axis.Location
    if not loc:
        loc = model.create_entity("IfcCartesianPoint", (0.0, 0.0, 0.0))
    elif len(loc.Coordinates) == 2:
        x, y = loc.Coordinates
        loc = model.create_entity("IfcCartesianPoint", (float(x), float(y), 0.0))
    new_axis = model.create_entity("IfcAxis2Placement3D", loc, None, None)
    lp.RelativePlacement = new_axis
    return new_axis


def rebase_model(model: "ifcopenshell.file", dx: float, dy: float, dz: float, logger: Optional[logging.Logger] = None) -> int:
    """
    Translate toutes les racines (IfcSite/IfcBuilding) de (dx,dy,dz) en mètres.
    Retourne le nombre de racines affectées.
    """
    log = _get_logger(logger)
    proj = (model.by_type("IfcProject") or [None])[0]
    if not proj:
        return 0
    roots: List[entity_instance] = []
    for rel in getattr(proj, "IsDecomposedBy", []) or []:
        roots.extend(getattr(rel, "RelatedObjects", []) or [])
    if not roots:
        roots = model.by_type("IfcSite") + model.by_type("IfcBuilding")

    changed = 0
    for r in roots:
        lp = _get_or_create_local_placement(model, r)
        axis = _force_relative_axis_3d(model, lp)
        loc = axis.Location
        x, y, z = (list(loc.Coordinates) + [0.0, 0.0, 0.0])[:3]
        loc.Coordinates = (float(x) + dx, float(y) + dy, float(z) + dz)
        changed += 1
    log.info("Décalage appliqué aux racines: %d (dx=%.3f, dy=%.3f, dz=%.3f)", changed, dx, dy, dz)
    return changed


# --- Doctor / Repair ----------------------------------------------------------
def quick_health(model: "ifcopenshell.file") -> dict:
    """Retourne un diagnostic minimal pour un IFC."""
    def c(t: str) -> int:
        try:
            return len(model.by_type(t))
        except Exception:
            return 0

    proj = (model.by_type("IfcProject") or [None])[0]
    schema_attr = getattr(model, "schema", None)
    schema = str(schema_attr() if callable(schema_attr) else schema_attr) if schema_attr else "Unknown"

    report = {
        "schema": schema,
        "has_project": bool(proj),
        "contexts": c("IfcGeometricRepresentationContext"),
        "rep_contexts": c("IfcRepresentationContext"),
        "products": c("IfcProduct"),
        "shapes": c("IfcProductDefinitionShape"),
        "sites": c("IfcSite"),
        "buildings": c("IfcBuilding"),
        "storeys": c("IfcBuildingStorey"),
        "has_units": bool(getattr(proj, "UnitsInContext", None)) if proj else False,
        "issues": [],
    }

    # Vérifie des champs problématiques (ex: Description non string)
    try:
        for rel in model.by_type("IfcRelAggregates"):
            if not (rel.Name is None or isinstance(rel.Name, str)):
                report["issues"].append("IfcRelAggregates.Name non string")
            if not (rel.Description is None or isinstance(rel.Description, str)):
                report["issues"].append("IfcRelAggregates.Description non string")
    except Exception:
        pass

    # Détection simple: racines en 2D
    try:
        roots = model.by_type("IfcSite") + model.by_type("IfcBuilding")
        roots_2d = 0
        for r in roots:
            lp = getattr(r, "ObjectPlacement", None)
            rp = getattr(lp, "RelativePlacement", None) if lp else None
            if rp and hasattr(rp, "is_a") and rp.is_a("IfcAxis2Placement2D"):
                roots_2d += 1
        if roots_2d:
            report["issues"].append(f"{roots_2d} racine(s) avec Axis2Placement2D")
    except Exception:
        pass

    if report["contexts"] == 0:
        report["issues"].append("Aucun IfcGeometricRepresentationContext")
    if report["shapes"] == 0:
        report["issues"].append("Aucune IfcProductDefinitionShape (pas de géométrie)")
    if not report["has_units"]:
        report["issues"].append("Aucune UnitsInContext")

    return report


def autofix_minimal(
    model: "ifcopenshell.file",
    *,
    force_contexts: bool = True,
    force_units: bool = True,
    promote_root_axis3d: bool = True,
    nullify_bad_relaggregates: bool = True,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Corrige les problèmes fréquents bloquants pour APS.
    Retourne le nombre d'actions effectuées.
    """
    log = _get_logger(logger)
    actions = 0

    if nullify_bad_relaggregates:
        try:
            for rel in model.by_type("IfcRelAggregates"):
                if not (rel.Name is None or isinstance(rel.Name, str)):
                    rel.Name = None; actions += 1
                if not (rel.Description is None or isinstance(rel.Description, str)):
                    rel.Description = None; actions += 1
        except Exception:
            pass

    if force_contexts:
        before = len(model.by_type("IfcGeometricRepresentationContext"))
        _attach_contexts_to_project(model)
        _ensure_minimal_context(model)
        after = len(model.by_type("IfcGeometricRepresentationContext"))
        if after > before:
            actions += (after - before + 1)

    if force_units:
        proj = (model.by_type("IfcProject") or [None])[0]
        if proj and not getattr(proj, "UnitsInContext", None):
            _ensure_minimal_units(model)
            actions += 1

    if promote_root_axis3d:
        # “translation 0” qui force Axis2Placement3D sur les racines
        actions += rebase_model(model, 0.0, 0.0, 0.0, logger=log)

    return actions


# --- GUIDs & inputs -----------------------------------------------------------
def _fix_duplicate_guids(model: "ifcopenshell.file") -> int:
    seen = set()
    changed = 0
    for e in model:
        if hasattr(e, "GlobalId"):
            gid = getattr(e, "GlobalId", None)
            if not gid:
                continue
            if gid in seen:
                setattr(e, "GlobalId", ifc_guid.new())
                changed += 1
            else:
                seen.add(gid)
    return changed


def _normalize_input_files(inputs: Union[Path, str, Iterable[Union[Path, str]]]) -> List[Path]:
    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            k = str(Path(x).resolve()).lower()
            if k not in seen:
                seen.add(k)
                out.append(Path(x))
        return out

    if isinstance(inputs, (list, tuple, set)) or (hasattr(inputs, "__iter__") and not isinstance(inputs, (str, Path))):
        files: List[Path] = []
        for p in inputs:
            p = Path(p)
            if p.is_dir():
                files += sorted(p.rglob("*.ifc"))
                files += sorted(p.rglob("*.ifczip"))
            else:
                files.append(p)
        files = [f for f in files if f.suffix.lower() in (".ifc", ".ifczip") and f.exists()]
        return _dedup(files)

    files = find_ifc_files(inputs)
    return _dedup(files)


# --- Fusion robuste -----------------------------------------------------------
def merge_ifc_files(
    inputs: Union[Path, str, Iterable[Union[Path, str]]],
    out_path: Union[Path, str],
    *,
    fix_duplicate_guids: bool = True,
    translate: Optional[Tuple[float, float, float]] = None,  # <-- option
    force_contexts: bool = False,                             # <-- option
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Fusionne plusieurs IFC en un seul (robuste pour les viewers / APS).
    - choisit une base qui a déjà de la géométrie,
    - copie profondément les racines (bâtiments en priorité), avec fallback spécial IfcSite,
    - rattache/assure les contexts et unités,
    - option de décalage global (rebase),
    - dédup GUIDs, écriture et synthèse.
    """
    log = _get_logger(logger)
    if _copy_deep is None:
        raise RuntimeError(
            "La fusion robuste requiert ifcopenshell>=0.7 et util.element.copy_deep. "
            "Mets à jour IfcOpenShell (pip install -U ifcopenshell)."
        )

    out_path = Path(out_path)
    files: List[Path] = _normalize_input_files(inputs)
    if not files:
        raise FileNotFoundError(f"Aucun fichier IFC trouvé pour: {inputs}")

    log.info("Fusion de %d fichier(s) → %s", len(files), out_path)

    # 1) base avec géométrie si possible
    base_path = _pick_geometry_base(files)
    order = [base_path] + [f for f in files if f != base_path]
    base = ifcopenshell.open(str(order[0]))
    log.debug("Fichier base: %s", base_path.name)

    # 2) copier profondément depuis les autres fichiers
    for f in tqdm(order[1:], desc="Fusion IFC", unit="étape"):
        try:
            src = ifcopenshell.open(str(f))
        except Exception as e:
            log.warning("Lecture impossible (%s) → ignoré: %s", e, f.name)
            continue

        roots = _iter_root_containers(src)
        log.debug("  + %s : %d racine(s)", f.name, len(roots))

        for r in roots:
            try:
                new_obj = _copy_deep(base, r)
                if new_obj is not None and (new_obj.is_a("IfcSite") or new_obj.is_a("IfcBuilding")):
                    _attach_under_project(base, new_obj)
            except Exception as e:
                if r.is_a("IfcSite"):
                    # Fallback “intelligent” : copier les IfcBuilding enfants du site
                    log.warning(
                        "Problème lors du traitement d’un IfcSite après copy_deep → copie des IfcBuilding enfants. (%s)", e
                    )
                    for b in _iter_children_of_type(r, "IfcBuilding"):
                        try:
                            nb = _copy_deep(base, b)
                            _attach_under_project(base, nb)
                        except Exception:
                            pass
                else:
                    log.warning("Échec copy_deep sur %s (%s) → objet sauté.", r.is_a(), e)
                continue

    # 3) contexts & unités (toujours utile pour schémas exotiques)
    if force_contexts:
        _attach_contexts_to_project(base)
        _ensure_minimal_context(base)
        _ensure_minimal_units(base)
    else:
        # attach au minimum les contexts existants
        _attach_contexts_to_project(base)

    # 4) décalage global si demandé
    if translate is not None:
        dx, dy, dz = translate
        rebase_model(base, dx, dy, dz, logger=log)

    # 5) dédup GUIDs
    if fix_duplicate_guids:
        changed = _fix_duplicate_guids(base)
        if changed:
            log.info("GUIDs dupliqués corrigés : %d", changed)

    # 6) écriture + synthèse
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.write(str(out_path))
    log.info("Écrit : %s", out_path.resolve())

    if not _has_geometry(base):
        log.warning(
            "Le fichier fusionné ne contient aucune géométrie exploitable "
            "(pas de IfcProductDefinitionShape)."
        )
    else:
        shp = len(base.by_type("IfcProductDefinitionShape"))
        prod = len(base.by_type("IfcProduct"))
        ctx = len(base.by_type("IfcGeometricRepresentationContext"))
        log.info("Synthèse géométrie → Contexts: %d | Products: %d | Shapes: %d", ctx, prod, shp)

    return out_path


# --- Fusion par règles --------------------------------------------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return t.casefold()


def merge_ifc_by_rules(
    inputs: Path | str,
    out_path: Path | str,
    *,
    building_name: Optional[str] = None,
    building_id: Optional[str] = None,
    disciplines: Optional[List[str]] = None,
    filename_regex: Optional[str] = None,
    fix_duplicate_guids: bool = True,
    translate: Optional[Tuple[float, float, float]] = None,  # option
    force_contexts: bool = False,                             # option
    logger: Optional[logging.Logger] = None,
) -> Path:
    log = _get_logger(logger)
    files = find_ifc_files(inputs)
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour: {inputs}")

    want_name = _norm(building_name) if building_name else None
    want_gid = building_id
    selected: List[Path] = []

    for f in files:
        fname = Path(f).name

        if disciplines:
            up = fname.upper()
            if not any(d.upper() in up for d in disciplines):
                log.debug("Ignore (discipline): %s", fname)
                continue

        if filename_regex and not re.search(filename_regex, fname, flags=re.IGNORECASE):
            log.debug("Ignore (regex): %s", fname)
            continue

        if want_name or want_gid:
            try:
                m = open_ifc(f)
                bdf = extract_buildings(m)
                if bdf is None or bdf.empty:
                    log.debug("Aucun IfcBuilding dans: %s", fname)
                    continue

                ok = True
                if want_name and not any(want_name in _norm(n) for n in bdf["Name"].tolist()):
                    ok = False
                if ok and want_gid and not any(str(g) == want_gid for g in bdf["GlobalId"].tolist()):
                    ok = False

                if not ok:
                    log.debug("Ignore (building rule): %s", fname)
                    continue
            except Exception as e:
                log.warning("Lecture impossible (%s) → ignoré: %s", e, fname)
                continue

        selected.append(Path(f))

    if not selected:
        raise FileNotFoundError("Aucun fichier ne correspond aux règles fournies.")

    log.info("Fichiers sélectionnés (%d): %s", len(selected), ", ".join(p.name for p in selected))
    return merge_ifc_files(
        inputs=selected,
        out_path=out_path,
        fix_duplicate_guids=fix_duplicate_guids,
        translate=translate,
        force_contexts=force_contexts,
        logger=log,
    )


# --- Résumé -------------------------------------------------------------------
def summarize_inputs(inputs: Union[str, Path]) -> dict:
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "count_files": 0,
        "files": [],
    }
    files = find_ifc_files(inputs)
    payload["count_files"] = len(files)

    for f in tqdm(files, desc="Résumé IFC", unit="fichier"):
        f = Path(f)
        rec: dict = {"file": f.name, "path": str(f)}
        try:
            m = open_ifc(f)
            rec["info"] = get_project_info(m)
            rec["counts"] = count_by_class(m)
            bdf = extract_buildings(m)
            rec["buildings"] = bdf.to_dict(orient="records")
        except Exception as e:
            rec["error"] = str(e)
        payload["files"].append(rec)
    return payload


__all__ = [
    "open_ifc",
    "get_project_info",
    "count_by_class",
    "get_spatial_tree",
    "extract_elements",
    "export_dataframe",
    "find_ifc_files",
    "batch_process",
    "extract_buildings",
    "merge_ifc_files",
    "merge_ifc_by_rules",
    "rebase_model",
    "quick_health",
    "autofix_minimal",
    "summarize_inputs",
]
