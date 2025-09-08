from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd


try:
    import ifcopenshell
    from ifcopenshell.entity_instance import entity_instance
except Exception as e:  # message propre si IfcOpenShell manque
    raise RuntimeError(
        "IfcOpenShell n'est pas installé ou ne se charge pas correctement."
    ) from e


# --- Chemins par défaut -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)


# --- Ouverture & infos --------------------------------------------------------

def open_ifc(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return ifcopenshell.open(str(path))


def get_project_info(model) -> Dict[str, str]:
    proj = model.by_type("IfcProject")[0] if model.by_type("IfcProject") else None
    schema = getattr(model, "schema", None)
    return {
        "FileName": getattr(model, "path", "") or "",
        "Schema": str(schema) if schema else "Unknown",
        "ProjectName": getattr(proj, "Name", None) or "",
        "Description": getattr(proj, "Description", None) or "",
        "NumEntities": str(len(model))  # total brut
    }


def count_by_class(model) -> Dict[str, int]:
    out = {}
    for ent in set(e.is_a() for e in model):
        out[ent] = len(model.by_type(ent))
    # tri décroissant
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


# --- Arbre spatial (Project → Site → Building → Storey) -----------------------

def _children(obj: entity_instance) -> Iterable[entity_instance]:
    # via IsDecomposedBy (agrégation/spatial)
    for rel in getattr(obj, "IsDecomposedBy", []) or []:
        for child in getattr(rel, "RelatedObjects", []) or []:
            yield child

def _contained_elements(container: entity_instance) -> Iterable[entity_instance]:
    # ex: storey → elements via ContainsElements
    for rel in getattr(container, "ContainsElements", []) or []:
        for elem in getattr(rel, "RelatedElements", []) or []:
            yield elem

def get_spatial_tree(model) -> Dict:
    """Construit un dict JSON-like de l’arborescence spatiale + contenus."""
    proj = model.by_type("IfcProject")[0]
    def walk(node):
        data = {
            "type": node.is_a(),
            "name": getattr(node, "Name", None),
            "global_id": getattr(node, "GlobalId", None),
            "children": [],
            "elements": [],
        }
        # éléments contenus (souvent au niveau Storey / Zone)
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
    """
    Map element.id() -> storey name (si disponible).
    """
    idx: Dict[int, str] = {}
    for storey in model.by_type("IfcBuildingStorey"):
        name = getattr(storey, "Name", None) or ""
        for elem in _contained_elements(storey):
            idx[elem.id()] = name
    return idx

def _get_pset_value(elem: entity_instance, pset: str, prop: str):
    """Lecture rapide des Psets sans dépendances annexes."""
    # Parcourt IfcRelDefinesByProperties -> IfcPropertySet
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
    """
    Retourne un DataFrame avec colonnes de base + propriétés demandées.

    psets : liste du style ["Pset_WallCommon:FireRating", "Pset_DoorCommon:IsExternal"]
    """
    if classes:
        targets = []
        for c in classes:
            targets.extend(model.by_type(c))
    else:
        # par défaut quelques classes "bâtiment"
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
        # propriétés IFC "standard"
        for attr in ("Tag", "PredefinedType", "ObjectType"):
            if hasattr(e, attr):
                row[attr] = getattr(e, attr)
        # Propriétés depuis Psets
        if psets:
            for spec in psets:
                if ":" in spec:
                    pset, prop = spec.split(":", 1)
                    row[f"{pset}.{prop}"] = _get_pset_value(e, pset, prop)
                else:
                    # si on ne précise pas la propriété, on ignore
                    pass
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

from glob import glob

def find_ifc_files(path_like) -> List[Path]:
    """
    Accepte : dossier, fichier, ou motif glob ('inputs/*.ifc').
    Renvoie une liste triée de chemins .ifc (et .ifczip s'il y en a).
    """
    p = Path(path_like)
    candidates: List[Path] = []
    if p.is_dir():
        candidates.extend(sorted(p.rglob("*.ifc")))
        candidates.extend(sorted(p.rglob("*.ifczip")))
    elif p.exists():
        candidates = [p]
    else:
        # motif glob
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
    """
    Traite plusieurs IFC et écrit 3 CSV agrégés (+ JSON par fichier si demandé).
    Renvoie les chemins écrits.
    """
    files = find_ifc_files(inputs)

    all_infos, all_counts, all_elements = [], [], []
    tree_paths: List[Path] = []

    for f in files:
        model = open_ifc(f)
        all_infos.append(project_info_row(model, f))
        all_counts.append(counts_to_dataframe(model, f))
        all_elements.append(extract_elements_for_file(model, f, classes=classes, psets=psets))

        if write_trees:
            tree = get_spatial_tree(model)
            out_json = OUTPUTS_DIR / f"{f.stem}_tree.json"
            out_json.write_text(
                json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8"
            )
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




def extract_buildings(model, select_id: Optional[str] = None, select_name: Optional[str] = None) -> pd.DataFrame:
    """
    Retourne un tableau des IfcBuilding avec infos utiles.
    Filtrage possible par GlobalId exact (--id) ou nom contenant (--name).
    """
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

        # niveaux du bâtiment
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


# --- Merge utilities ----------------------------------------------------------
from typing import List, Optional
from pathlib import Path

import ifcopenshell
from ifcopenshell import guid as ifc_guid
try:
    # IfcPatch est livré avec IfcOpenShell
    import ifcpatch
except Exception as e:
    raise RuntimeError(
        "Le module 'ifcpatch' n'est pas disponible. Mets à jour IfcOpenShell (pip install -U ifcopenshell)."
    ) from e


def _fix_duplicate_guids(model: "ifcopenshell.file") -> int:
    """
    Détecte des GUIDs dupliqués (sur les entités IfcRoot) et en régénère.
    Retourne le nombre de GUIDs modifiés.
    """
    seen = set()
    changed = 0
    for e in model:
        # Seules les entités IfcRoot possèdent GlobalId
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


def merge_ifc_files(
    inputs: str | Path,
    out_path: str | Path,
    fix_duplicate_guids: bool = True,
) -> Path:
    """
    Fusionne plusieurs fichiers IFC en 1 modèle.

    - Sélectionne le **premier** fichier comme base et y fusionne les suivants.
    - Utilise la recette IfcPatch 'MergeProjects' (gère l'union des IfcProject
      et convertit automatiquement les unités de longueur des fichiers ajoutés).
    - Optionnellement corrige les GUIDs dupliqués.

    Params
    ------
    inputs : dossier/fichier/motif glob (ex: 'inputs/*.ifc' ou Path('inputs'))
    out_path : chemin du .ifc de sortie
    fix_duplicate_guids : régénère les GlobalId en cas de doublon

    Retour
    ------
    Path du fichier de sortie écrit.
    """
    files: List[Path] = find_ifc_files(inputs)
    if len(files) < 2:
        raise ValueError("Il faut au moins 2 fichiers IFC pour fusionner.")

    base = files[0]
    others = [str(p) for p in files[1:]]

    base_model = ifcopenshell.open(str(base))

    # Exécute la recette officielle "MergeProjects"
    merged_output = ifcpatch.execute({
        "input": str(base),
        "file": base_model,
        "recipe": "MergeProjects",
        "arguments": others,
    })  # retourne typiquement un ifcopenshell.file

    # L'API renvoie soit un modèle, soit une chaîne – 'write' gère les deux.
    out_path = Path(out_path)
    if not out_path.suffix.lower():
        out_path = out_path.with_suffix(".ifc")

    # Corrige d'éventuels GUIDs dupliqués (optionnel mais prudent)
    try:
        model_obj = merged_output if isinstance(merged_output, ifcopenshell.file) else base_model
        if fix_duplicate_guids:
            _fix_duplicate_guids(model_obj)
    except Exception:
        # Si on ne peut pas caster proprement, on laisse IfcPatch écrire tel quel
        pass

    ifcpatch.write(merged_output, str(out_path))
    return out_path
