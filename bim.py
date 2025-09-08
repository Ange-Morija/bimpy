#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

# >>> change ici : on importe le module entier
import functions as F

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _setup_logging(verbosity: int, log_file: Path | None) -> logging.Logger:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger("bimpy")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="bimpy", description="Outils IFC minimalistes")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Augmente la verbosité (-v=INFO, -vv=DEBUG)")
    parser.add_argument("--log-file", type=Path, help="Chemin d'un fichier log (ex: outputs/merge.log)")

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

    # --- BUILDINGS ---
    p_build = sub.add_parser("buildings", help="Extraire les IfcBuilding (bâtiments)")
    p_build.add_argument("ifc", type=Path)
    p_build.add_argument("--id", help="Filtrer par GlobalId exact")
    p_build.add_argument("--name", help="Filtrer par nom (contient, insensible à la casse)")
    p_build.add_argument("--out", choices=["csv", "json", "print"], default="csv",
                         help="Format de sortie (csv par défaut)")

    # --- BATCH ---
    p_batch = sub.add_parser("batch", help="Traiter un dossier ou motif de fichiers IFC")
    p_batch.add_argument("input", type=Path, help="Dossier, fichier, ou motif glob (ex: inputs/*.ifc)")
    p_batch.add_argument("--classes", help="Liste séparée par des virgules (ex: IfcWall,IfcDoor)")
    p_batch.add_argument("--pset", action="append",
                         help="Répéter l'option. Ex: --pset Pset_WallCommon:FireRating")
    p_batch.add_argument("--trees", action="store_true",
                         help="Écrire un JSON d'arbre spatial par fichier")
    p_batch.add_argument("--out-prefix", default="batch",
                         help="Préfixe des fichiers de sortie")

    # --- MERGE ---
    p_merge = sub.add_parser("merge", help="Fusionner plusieurs IFC (avec règles de sélection)")
    p_merge.add_argument("input", type=str, help="Dossier/fichier/motif glob (ex: inputs/*.ifc)")
    p_merge.add_argument("output", type=Path, help="Chemin du fichier IFC résultant")
    p_merge.add_argument("--building-name", dest="building_name",
                         help="Inclure les fichiers qui contiennent un IfcBuilding portant ce nom (match partiel)")
    p_merge.add_argument("--building-id", dest="building_id",
                         help="Inclure les fichiers qui contiennent un IfcBuilding avec ce GlobalId exact")
    p_merge.add_argument("--discipline", action="append",
                         help="Filtre sur fragments de nom de fichier (ex: --discipline -ARQ-)")
    p_merge.add_argument("--filename-regex", dest="filename_regex",
                         help="Filtre regex sur le nom de fichier")
    p_merge.add_argument("--no-guid-fix", dest="no_guid_fix", action="store_true",
                         help="Ne pas dédupliquer les GUIDs après fusion")
    p_merge.add_argument("--translate", nargs=3, type=float, metavar=("DX","DY","DZ"),
                         help="Décalage global (m) appliqué après fusion")
    p_merge.add_argument("--force-contexts", action="store_true",
                         help="Force contexte/units minima dans le fichier final")

    # --- SUMMARY ---
    p_summary = sub.add_parser("summary", help="Résumé JSON multi-fichiers (infos, counts, buildings)")
    p_summary.add_argument("input", type=str, help="Dossier/fichier/motif glob (ex: inputs/*.ifc)")
    p_summary.add_argument("--out", type=Path, help="Fichier de sortie JSON (ex: outputs/summary.json)")
    p_summary.add_argument("--pretty", action="store_true", help="Afficher le JSON joliment dans la console")

    # --- DOCTOR ---
    p_doctor = sub.add_parser("doctor", help="Vérifie/fixe les IFC (contexts/units/placements/relations)")
    p_doctor.add_argument("input", type=str, help="Dossier/fichier/motif glob (ex: inputs/*.ifc)")
    p_doctor.add_argument("--fix", action="store_true", help="Écrire des copies corrigées")
    p_doctor.add_argument("--out-dir", type=Path, default=Path("outputs/fixed"), help="Dossier de sortie")
    p_doctor.add_argument("--force-contexts", action="store_true", help="Ajouter un contexte minimal si absent")
    p_doctor.add_argument("--force-units", action="store_true", help="Ajouter des unités minimales si absentes")
    p_doctor.add_argument("--promote-root-axis3d", action="store_true",
                          help="Forcer Axis2Placement3D sur les placements racine")
    p_doctor.add_argument("--nullify-bad-relaggregates", action="store_true",
                          help="Réinitialiser Name/Description invalides sur IfcRelAggregates")
    p_doctor.add_argument("--report", type=Path, help="Écrire un rapport JSON")

    args = parser.parse_args()
    log = _setup_logging(args.verbose, getattr(args, "log_file", None))

    # --- BRANCHES ---
    if args.cmd == "info":
        model = F.open_ifc(args.ifc)
        info = F.get_project_info(model)
        for k, v in info.items():
            print(f"{k}: {v}")

    elif args.cmd == "counts":
        model = F.open_ifc(args.ifc)
        counts = F.count_by_class(model)
        width = max(len(k) for k in counts.keys())
        for k, v in counts.items():
            print(f"{k:<{width}}  {v}")

    elif args.cmd == "tree":
        model = F.open_ifc(args.ifc)
        data = F.get_spatial_tree(model)
        if args.out == "print":
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            out = Path("outputs") / "spatial_tree.json"
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Écrit: {out.resolve()}")

    elif args.cmd == "extract":
        model = F.open_ifc(args.ifc)
        classes = args.classes.split(",") if args.classes else None
        psets = args.pset if args.pset else None
        df = F.extract_elements(model, classes=classes, psets=psets)
        out = F.export_dataframe(df, "elements")
        print(f"Écrit: {out.resolve()}")

    elif args.cmd == "buildings":
        model = F.open_ifc(args.ifc)
        df = F.extract_buildings(model, select_id=args.id, select_name=args.name)
        if args.out == "print":
            print(df.to_string(index=False))
        else:
            fmt = args.out  # 'csv' ou 'json'
            out = F.export_dataframe(df, "buildings", fmt=fmt)
            print(f"Écrit: {out.resolve()}")

    elif args.cmd == "batch":
        classes = args.classes.split(",") if args.classes else None
        psets = args.pset if args.pset else None
        log.info("Batch: input=%s classes=%s psets=%s trees=%s prefix=%s",
                 args.input, classes, psets, args.trees, args.out_prefix)
        results = F.batch_process(
            inputs=args.input,
            classes=classes,
            psets=psets,
            write_trees=args.trees,
            out_prefix=args.out_prefix,
        )
        print(f"Écrit : {results['info_csv']}")
        print(f"Écrit : {results['counts_csv']}")
        elements_csv = results.get("elements_csv")
        if elements_csv and str(elements_csv):
            print(f"Écrit : {elements_csv}")
        if args.trees:
            print(f"Arbres spatiaux JSON dans : {results['trees_dir'].resolve()}")

    elif args.cmd == "merge":
        fix_guids = not args.no_guid_fix
        translate = tuple(args.translate) if args.translate else None
        log.info(
            "Merge: input=%s output=%s building_name=%s building_id=%s disciplines=%s regex=%s "
            "fix_guids=%s translate=%s force_contexts=%s ",
            args.input, args.output, args.building_name, args.building_id,
            args.discipline, args.filename_regex, fix_guids, translate, args.force_contexts,
        )
        out = F.merge_ifc_by_rules(
            inputs=args.input,
            out_path=args.output,
            building_name=args.building_name,
            building_id=args.building_id,
            disciplines=args.discipline,
            filename_regex=args.filename_regex,
            fix_duplicate_guids=fix_guids,
            translate=translate,
            force_contexts=args.force_contexts,
            logger=log,
        )
        out_path = Path(out) if out else Path(args.output)
        print(f"Écrit : {out_path.resolve()}")

    elif args.cmd == "summary":
        files = F.find_ifc_files(args.input)
        log.info("Summary sur %d fichier(s)", len(files))

        payload = {
            "generated_at": Path.cwd().as_posix(),
            "count_files": len(files),
            "files": [],
        }

        for f in files:
            f = Path(f)
            rec: dict = {"file": f.name, "path": str(f)}
            try:
                m = F.open_ifc(f)
                rec["info"] = F.get_project_info(m)
                rec["counts"] = F.count_by_class(m)
                bdf = F.extract_buildings(m)
                rec["buildings"] = bdf.to_dict(orient="records")
            except Exception as e:
                rec["error"] = str(e)
            payload["files"].append(rec)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Écrit : {out_path.resolve()}")

        if args.pretty:
            print(json.dumps(payload, ensure_ascii=False, indent=2))

    elif args.cmd == "doctor":
        # On réutilise les helpers exposés par ifc_tools via le module functions (F)
        files = F.find_ifc_files(args.input)
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        report: list[dict] = []
        from tqdm.auto import tqdm
        for f in tqdm(files, desc="Doctor IFC", unit="fichier"):
            f = Path(f)
            rec: dict = {"file": f.name, "path": str(f)}
            m = F.open_ifc(f)

            before = {
                "schema": getattr(m, "schema", lambda: "Unknown")(),
                "has_project": bool(m.by_type("IfcProject")),
                "contexts": len(m.by_type("IfcGeometricRepresentationContext")),
                "rep_contexts": len(m.by_type("IfcRepresentationContext")),
                "products": len(m.by_type("IfcProduct")),
                "shapes": len(m.by_type("IfcProductDefinitionShape")),
                "sites": len(m.by_type("IfcSite")),
                "buildings": len(m.by_type("IfcBuilding")),
                "storeys": len(m.by_type("IfcBuildingStorey")),
                "has_units": bool((m.by_type("IfcProject") or [None])[0]
                                  and getattr((m.by_type("IfcProject") or [None])[0], "UnitsInContext", None)),
                "issues": [],
            }

            fix_actions = 0
            if args.force_contexts:
                try:
                    F._attach_contexts_to_project(m)  # type: ignore[attr-defined]
                    F._ensure_minimal_context(m)      # type: ignore[attr-defined]
                    fix_actions += 1
                except Exception:
                    pass
            if args.force_units:
                try:
                    F._ensure_minimal_units(m)        # type: ignore[attr-defined]
                    fix_actions += 1
                except Exception:
                    pass
            if args.promote_root_axis3d:
                try:
                    F.rebase_model(m, 0.0, 0.0, 0.0)  # assure Axis2Placement3D à la racine
                    fix_actions += 1
                except Exception:
                    pass
            if args.nullify_bad_relaggregates:
                try:
                    for rel in m.by_type("IfcRelAggregates"):
                        # sécurité : Name/Description doivent être STRING, pas des entités
                        if hasattr(rel, "Name") and not isinstance(getattr(rel, "Name"), (str, type(None))):
                            rel.Name = None
                        if hasattr(rel, "Description") and not isinstance(getattr(rel, "Description"), (str, type(None))):
                            rel.Description = None
                    fix_actions += 1
                except Exception:
                    pass

            fixed_path = None
            if args.fix:
                fixed_path = out_dir / f"{f.stem}_fixed.ifc"
                m.write(str(fixed_path))

            after = {
                "schema": getattr(m, "schema", lambda: "Unknown")(),
                "has_project": bool(m.by_type("IfcProject")),
                "contexts": len(m.by_type("IfcGeometricRepresentationContext")),
                "rep_contexts": len(m.by_type("IfcRepresentationContext")),
                "products": len(m.by_type("IfcProduct")),
                "shapes": len(m.by_type("IfcProductDefinitionShape")),
                "sites": len(m.by_type("IfcSite")),
                "buildings": len(m.by_type("IfcBuilding")),
                "storeys": len(m.by_type("IfcBuildingStorey")),
                "has_units": bool((m.by_type("IfcProject") or [None])[0]
                                  and getattr((m.by_type("IfcProject") or [None])[0], "UnitsInContext", None)),
                "issues": [],
            }

            rec.update({
                "before": before,
                "fix_actions": fix_actions,
                "fixed_path": str(fixed_path) if fixed_path else None,
                "after": after,
            })
            report.append(rec)

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Écrit : {args.report.resolve()}")
        else:
            for r in report:
                fp = r.get("fixed_path") or "-"
                print(f"- {r['file']}: shapes={r['before']['shapes']} contexts={r['before']['contexts']} "
                      f"issues={len(r['before']['issues'])} | FIX={r['fix_actions']} | OUT={fp}")

    else:
        parser.error("Commande inconnue")

if __name__ == "__main__":
    main()
