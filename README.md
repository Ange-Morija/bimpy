# BIMpy IFC Merge Utility

This project provides command-line tools to:

- **Detect duplicate GlobalIds** in IFC files.
- **Merge multiple IFC files** into one, with optional automatic GUID remapping in case of collisions.
- **Scan a folder** (recursively or not) to process all IFC files found.

## Main Features

- **Duplicate Detection**: Generate JSON or CSV reports of duplicate GlobalIds.
- **Advanced Merging**: Merge multiple IFCs, specifying a base file, with automatic handling of GlobalId collisions.
- **Flexible Options**: Supports input directories, recursion, dry-run, and base file selection.

## Installation

Install the required dependencies:

```bash
pip install ifcopenshell
```

## Usage

### CLI

Un nouveau point d'entrée `cli.py` expose l'ensemble des fonctionnalités via des sous-commandes.

Exemples rapides (depuis la racine du projet):

```bash
python cli.py merge --input-dir input -o output/merged.ifc --remap --recursive
python cli.py detect --input-dir input --recursive --report output/duplicates.json --csv output/duplicates.csv
python cli.py check --ifc output/merged.ifc --report output/check_report.json
python cli.py check-all
python cli.py extract --ifc input/merged.ifc --query "Ma zone" --out output/zone.ifc
python cli.py recenter --input output/merged.ifc --output output/merged_recentred.ifc
python cli.py add-sensors sensors.json input/merged.ifc output/with_sensors.ifc
python cli.py diagnose input/merged.ifc output/diagnose.json
python cli.py find-zone --ifc input/merged.ifc --query "zone_name"
```

### Options

- `--input-dir`: Folder containing IFC files to process.
- `--recursive`: Recursively search subfolders.
- `--base`: IFC file to use as the merge base.
- `--remap`: Automatically remap GlobalIds in case of collision.
- `--dry-run`: Show actions without writing the output file.

## Best Practices

- Always keep a backup of your original files.
- Check the order of files when merging: the first file is used as the base.

## Project Structure

- `functions/merge_ifc.py`: Main script for IFC detection and merging.
- `input/`: Place your IFC files to process here.
- `output/`: Merged files will be generated here.

## License

This project is licensed under the MIT.