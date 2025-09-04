# appeler depuis la racine du projet
from bimpy.functions.merge_ifc import merge_ifcs, gather_input_files, setup_logger

# préparer fichiers
files = gather_input_files(inputs=None, input_dir="input", recursive=True)

# préparer logger (optionnel)
logger = setup_logger("output/merge.log", verbose_console=True)

# exécuter dry-run
remaps = merge_ifcs(output_path="output/merged.ifc", input_paths=files, remap_guids=True, dry_run=True, logger=logger)

print("Remaps (sample):", remaps[:10])
# si ok, relancer sans dry_run
merge_ifcs(output_path="output/merged.ifc", input_paths=files, remap_guids=True, dry_run=False, logger=logger)
