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

### Detect Duplicates

```bash
python merge_ifc.py detect --input-dir input --recursive --report duplicates.json --csv duplicates.csv
```

### Merge IFC Files

```bash
python merge_ifc.py merge --input-dir input -o output/merged.ifc --remap --recursive
```

To force a specific base file:

```bash
python merge_ifc.py merge --input-dir input --base input/my_base.ifc -o output/merged.ifc --remap --recursive
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