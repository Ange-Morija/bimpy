#!/usr/bin/env python3
"""CLI wrapper to expose project functionalities as subcommands.

Usage examples:
  python cli.py merge --input-dir input -o output/merged.ifc --remap --recursive
  python cli.py detect --input-dir input --report output/duplicates.json
  python cli.py check --ifc output/merged.ifc --report output/check_report.json
  python cli.py extract --ifc input/merged.ifc --query "zone" --out output/zone.ifc
  python cli.py recenter --input output/merged.ifc --output output/merged_recentred.ifc
  python cli.py add-sensors sensors.json input/merged.ifc output/with_sensors.ifc
  python cli.py diagnose input/merged.ifc
  python cli.py find-zone --ifc input/merged.ifc --query "zone"
"""
import argparse
from pathlib import Path
import sys

from bimpy.functions import merge_ifc, check_output, extract_building, recenter_merged, add_sensors_json


def main():
    parser = argparse.ArgumentParser(description="BIMpy unified CLI")
    sub = parser.add_subparsers(dest='cmd')

    # merge/detect -> reuse merge_ifc's CLI logic by delegating
    p_merge = sub.add_parser('merge', help='Merge IFC files (delegates to functions.merge_ifc)')
    p_merge.add_argument('--input-dir')
    p_merge.add_argument('--recursive', action='store_true')
    p_merge.add_argument('--base')
    p_merge.add_argument('-o', '--output', required=True)
    p_merge.add_argument('--remap', action='store_true')
    p_merge.add_argument('--dry-run', action='store_true')
    p_merge.add_argument('--log')

    p_detect = sub.add_parser('detect', help='Detect duplicate GlobalIds (delegates to functions.merge_ifc detect)')
    p_detect.add_argument('--input-dir')
    p_detect.add_argument('--recursive', action='store_true')
    p_detect.add_argument('--report')
    p_detect.add_argument('--csv')
    p_detect.add_argument('--log')

    p_check = sub.add_parser('check', help='Inspect a single IFC and write a report')
    p_check.add_argument('--ifc', default='output/merged.ifc')
    p_check.add_argument('--report', default='output/check_report.json')
    p_check.add_argument('--log', default='output/check_output.log')
    p_check.add_argument('--no-progress', action='store_true')

    p_extract = sub.add_parser('extract', help='Extract zone from IFC')
    p_extract.add_argument('--ifc', default='input/merged.ifc')
    p_extract.add_argument('--query', required=True)
    p_extract.add_argument('--out', default='output/zone_extract.ifc')

    p_recenter = sub.add_parser('recenter', help='Recenter merged IFC')
    p_recenter.add_argument('--input', default='output/merged.ifc')
    p_recenter.add_argument('--output', default='output/merged_recentred.ifc')
    p_recenter.add_argument('--log', default='output/recenter.log')

    p_sensors = sub.add_parser('add-sensors', help='Inject sensors from JSON into IFC')
    p_sensors.add_argument('json', nargs='?', default='sensors.json')
    p_sensors.add_argument('ifc_in', nargs='?', default='output/malaga_try.ifc')
    p_sensors.add_argument('ifc_out', nargs='?', default='output/malaga_with_sensors.ifc')

    # note: reduced set of commands (merge, detect, check, extract, recenter, add-sensors)

    args = parser.parse_args()

    if args.cmd == 'merge':
        # build files list from input dir
        files = merge_ifc.gather_input_files([], args.input_dir, recursive=args.recursive)
        if args.base:
            base_path = str(Path(args.base))
            files = [f for f in files if f != base_path]
            files.insert(0, base_path)
        log = args.log if args.log else None
        merge_ifc.setup_logger(log, verbose_console=True)
        remaps = merge_ifc.merge_ifcs(args.output, files, remap_guids=args.remap, dry_run=args.dry_run, logger=None)
        print('Done. Remaps sample:', remaps[:10])

    elif args.cmd == 'detect':
        files = merge_ifc.gather_input_files([], args.input_dir, recursive=args.recursive)
        merge_ifc.setup_logger(args.log, verbose_console=True)
        dup = merge_ifc.detect_duplicates(files, logger=None)
        if args.report:
            merge_ifc.write_json_report(dup, args.report)
        if args.csv:
            merge_ifc.write_csv_report(dup, args.csv)
        print('Duplicates found:', len(dup))

    elif args.cmd == 'check':
        check_output.main()

    # 'check-all' command removed; use 'check' for single-file inspection

    elif args.cmd == 'extract':
        try:
            extract_building.extract_zone_to_file(Path(args.ifc), args.query, Path(args.out))
        except Exception as e:
            print('Extraction failed:', e); raise

    elif args.cmd == 'recenter':
        recenter_merged.recenter(Path(args.input), Path(args.output), Path(args.log))

    elif args.cmd == 'add-sensors':
        add_sensors_json.main(args.json, args.ifc_in, args.ifc_out)

    # removed: diagnose, find-zone

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
