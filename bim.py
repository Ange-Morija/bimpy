#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

# tqdm (progress bars) avec fallback silencieux si non installé
try:
    from tqdm.auto import tqdm
except Exception:  # fallback no-op
    class _DummyTqdm:
        def __init__(self, *a, **k): pass
        def __iter__(self): return iter([])
        def update(self, *a, **k): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
    def tqdm(iterable=None, *a, **k):
        if iterable is None:
            return _DummyTqdm()
        return iterable



from functions import (
    open_ifc,
    get_project_info,
    count_by_class,
    get_spatial_tree,
    extract_elements,
    export_dataframe,
    batch_process,       # conservé pour compat, non utilisé dans la voie "progress"
    extract_buildings,   # NEW
    merge_ifc_files,     # NEW
)

# On importe quelques utilitaires internes pour pouvoir montrer une vraie progression par fichier
# (ils ont été ajoutés précédemment dans functions/ifc_tools.py)
try:
    from functions.ifc_tools import (
        find_ifc_files,
        project_info_row,
        counts_to_dataframe,
        extract_elements_for_file,
        OUTPUTS_DIR,
    )
except Exception:
    # Si ces helpers ne sont pas exportés, on continue sans barre détaillée pour 'batch'
    find_ifc_files = None
    project_info_row = None
    counts_to_dataframe = None
    extract_elements_for_file = None
    OUTPUTS_DIR = Path("outputs")


def main():
    parser = argparse.ArgumentParser(prog="bimpy", description="Outils IFC minimalistes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- COMMANDES SUR UN SEUL FICHIER ---
    p_info = sub.add_parser("info", help="Infos de base sur le fichier")
    p_info.add_argument("ifc", type=Path)

    p_counts = sub.add_parser("counts", help="Comptage par classe IFC")
    p_counts.add_argument("ifc", type=Path)

    p_tree = sub.add_parser("tree", help="Arbre spatial (JSON)")
    p_tree.add_argument("ifc", type=Path)
    p_tree.add_argument("--out", choices=["json", "print"], default="print")

    p_extract = sub.add_parser("extract", help="Extraction d'éléments en tableau")
    p_extract.add_argument("ifc", type=Path)
    p_extract.add_argument("--classes", help="Liste séparée par des virgules (ex: IfcWall,IfcDoor)")
    p_extract.add_argument("--pset", action="append",
                           help="Répéter l'option. Ex: --pset Pset_WallCommon:FireRating")

    # --- NOUVELLE COMMANDE : BUILDINGS ---
    p_build = sub.add_parser("buildings", help="Extraire les IfcBuilding (bâtiments)")
    p_build.add_argument("ifc", type=Path)
    p_build.add_argument("--id", help="Filtrer par GlobalId exact")
    p_build.add_argument("--name", help="Filtrer par nom (contient, insensible à la casse)")
    p_build.add_argument("--out", choices=["csv", "json", "print"], default="csv",
                         help="Format de sortie (csv par défaut)")

    # --- COMMANDE BATCH ---
    p_batch = sub.add_parser("batch", help="Traiter un dossier ou motif de fichiers IFC")
    p_batch.add_argument("input", type=Path, help="Dossier, fichier, ou motif glob (ex: inputs/*.ifc)")
    p_batch.add_argument("--classes", help="Liste séparée par des virgules (ex: IfcWall,IfcDoor)")
    p_batch.add_argument("--pset", action="append",
                         help="Répéter l'option. Ex: --pset Pset_WallCommon:FireRating")
    p_batch.add_argument("--trees", action="store_true",
                         help="Écrire un JSON d'arbre spatial par fichier")
    p_batch.add_argument("--out-prefix", default="batch",
                         help="Préfixe des fichiers de sortie")

    # --- COMMANDE : MERGE ---
    p_merge = sub.add_parser("merge", help="Fusionner plusieurs IFC en un seul")
    p_merge.add_argument("input", type=Path, help="Dossier, fichier, ou motif glob (ex: inputs/*.ifc)")
    p_merge.add_argument("output", type=Path, help="Chemin du fichier IFC de sortie (ex: outputs/merged.ifc)")
    p_merge.add_argument("--no-guid-fix", action="store_true",
                         help="Ne pas régénérer les GUIDs en cas de doublon")

    args = parser.parse_args()

    # --- BRANCHES ---
    if args.cmd == "info":
        with tqdm(total=2, desc="Lecture & infos", unit="étape") as pbar:
            model = open_ifc(args.ifc); pbar.update(1)
            info = get_project_info(model); pbar.update(1)
        for k, v in info.items():
            print(f"{k}: {v}")

    elif args.cmd == "counts":
        with tqdm(total=2, desc="Lecture & comptage", unit="étape") as pbar:
            model = open_ifc(args.ifc); pbar.update(1)
            counts = count_by_class(model); pbar.update(1)
        width = max(len(k) for k in counts.keys())
        for k, v in counts.items():
            print(f"{k:<{width}}  {v}")

    elif args.cmd == "tree":
        with tqdm(total=3, desc="Arbre spatial", unit="étape") as pbar:
            model = open_ifc(args.ifc); pbar.update(1)
            data = get_spatial_tree(model); pbar.update(1)
            if args.out == "print":
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                out = Path("outputs") / "spatial_tree.json"
                out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Écrit: {out.resolve()}")
            pbar.update(1)

    elif args.cmd == "extract":
        classes = args.classes.split(",") if args.classes else None
        psets = args.pset if args.pset else None
        with tqdm(total=3, desc="Extraction d'éléments", unit="étape") as pbar:
            model = open_ifc(args.ifc); pbar.update(1)
            df = extract_elements(model, classes=classes, psets=psets); pbar.update(1)
            out = export_dataframe(df, "elements"); pbar.update(1)
        print(f"Écrit: {out.resolve()}")

    elif args.cmd == "buildings":
        with tqdm(total=3, desc="Extraction des bâtiments", unit="étape") as pbar:
            model = open_ifc(args.ifc); pbar.update(1)
            df = extract_buildings(model, select_id=args.id, select_name=args.name); pbar.update(1)
            if args.out == "print":
                print(df.to_string(index=False))
                pbar.update(1)
            else:
                fmt = args.out  # 'csv' ou 'json'
                out = export_dataframe(df, "buildings", fmt=fmt); pbar.update(1)
                print(f"Écrit: {out.resolve()}")

    elif args.cmd == "batch":
        classes = args.classes.split(",") if args.classes else None
        psets = args.pset if args.pset else None

        # Si les helpers existent, on fait un batch AVEC barre fine par fichier
        if find_ifc_files and project_info_row and counts_to_dataframe and extract_elements_for_file:
            files = find_ifc_files(args.input)
            if not files:
                raise FileNotFoundError(f"Aucun fichier trouvé pour: {args.input}")

            info_rows, counts_frames, elements_frames = [], [], []

            for f in tqdm(files, desc="Traitement des fichiers IFC", unit="fichier"):
                model = open_ifc(f)
                info_rows.append(project_info_row(model, f))
                counts_frames.append(counts_to_dataframe(model, f))
                df_e = extract_elements_for_file(model, f, classes=classes, psets=psets)
                if df_e is not None and not df_e.empty:
                    elements_frames.append(df_e)

                if args.trees:
                    # JSON d'arbre par fichier
                    tree = get_spatial_tree(model)
                    out_json = OUTPUTS_DIR / f"{Path(f).stem}_tree.json"
                    out_json.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")

            # Agrégation + exports
            info_df = pd.DataFrame(info_rows).sort_values("File")
            counts_df = pd.concat(counts_frames, ignore_index=True).sort_values(["Class", "File"])
            elems_df = pd.concat(elements_frames, ignore_index=True) if elements_frames else pd.DataFrame()

            info_csv = export_dataframe(info_df, f"{args.out_prefix}_info")
            counts_csv = export_dataframe(counts_df, f"{args.out_prefix}_counts")
            elems_csv = export_dataframe(elems_df, f"{args.out_prefix}_elements") if not elems_df.empty else None

            print(f"Écrit : {info_csv}")
            print(f"Écrit : {counts_csv}")
            if elems_df is not None and not elems_df.empty:
                print(f"Écrit : {elems_csv}")
            if args.trees:
                print(f"Arbres spatiaux JSON dans : {OUTPUTS_DIR.resolve()}")

        else:
            # Fallback : on garde la fonction batch existante (progress global)
            with tqdm(total=1, desc="Traitement batch", unit="lot") as pbar:
                results = batch_process(
                    inputs=args.input,
                    classes=classes,
                    psets=psets,
                    write_trees=args.trees,
                    out_prefix=args.out_prefix,
                )
                pbar.update(1)
            print(f"Écrit : {results['info_csv']}")
            print(f"Écrit : {results['counts_csv']}")
            elements_csv = results.get("elements_csv")
            if elements_csv and getattr(elements_csv, "name", ""):
                print(f"Écrit : {elements_csv}")
            if args.trees:
                print(f"Arbres spatiaux JSON dans : {results['trees_dir'].resolve()}")

    elif args.cmd == "merge":
        # On montre une petite progression "étapes" autour de la fusion
        # (IfcPatch fait le gros du travail à l'intérieur)
        with tqdm(total=2, desc="Fusion IFC", unit="étape") as pbar:
            # Étape 1 : vérification liste de fichiers (si dispo)
            try:
                if find_ifc_files:
                    _ = find_ifc_files(args.input)
            finally:
                pbar.update(1)

            # Étape 2 : fusion
            out = merge_ifc_files(
                inputs=args.input,
                out_path=args.output,
                fix_duplicate_guids=not args.no_guid_fix,
            )
            pbar.update(1)
        print(f"Écrit : {out.resolve()}")

if __name__ == "__main__":
    main()
