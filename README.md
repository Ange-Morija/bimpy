# bimpy

Outils CLI minimalistes pour travailler avec des fichiers IFC.

Principales fonctionnalités
- Ouvrir et lire des fichiers IFC : [`functions.open_ifc`](functions/ifc_tools.py)
- Récupérer les méta-données projet : [`functions.get_project_info`](functions/ifc_tools.py)
- Compter les entités par classe : [`functions.count_by_class`](functions/ifc_tools.py)
- Exporter l'arbre spatial en JSON : [`functions.get_spatial_tree`](functions/ifc_tools.py)
- Extraire éléments / propriétés en DataFrame : [`functions.extract_elements`](functions/ifc_tools.py), [`functions.export_dataframe`](functions/ifc_tools.py)
- Traitement par lot : [`functions.batch_process`](functions/ifc_tools.py)
- Extraire les bâtiments : [`functions.extract_buildings`](functions/ifc_tools.py)
- Fusionner plusieurs IFC avec règles : [`functions.merge_ifc_by_rules`](functions/ifc_tools.py)
- Rebaser (déplacer) les racines du modèle : [`functions.rebase_model`](functions/ifc_tools.py)
- Trouver les fichiers IFC dans un motif/dossier : [`functions.find_ifc_files`](functions/ifc_tools.py)

Fichiers importants
- CLI principal : [bim.py](bim.py)
- Fonctions utilitaires IFC : [functions/ifc_tools.py](functions/ifc_tools.py)
- Dépendances : [requirements.txt](requirements.txt)
- Entrées/Sorties exemple : [inputs/](inputs/), [outputs/](outputs/)

Installation rapide
1. (Optionnel) Créez et activez un environnement virtuel Python.
2. Installez les dépendances :
   pip install -r requirements.txt

Usage (exemples)
- Infos basiques :
  python bim.py info inputs/mon_fichier.ifc

- Compte par classe :
  python bim.py counts inputs/mon_fichier.ifc

- Arbre spatial (JSON, ou affichage) :
  python bim.py tree inputs/mon_fichier.ifc --out json
  python bim.py tree inputs/mon_fichier.ifc --out print

- Extraire éléments en tableau (CSV par défaut) :
  python bim.py extract inputs/mon_fichier.ifc --classes IfcWall,IfcDoor
  Pour inclure des property sets : --pset Pset_WallCommon:FireRating (répéter l'option)

- Extraire bâtiments :
  python bim.py buildings inputs/mon_fichier.ifc --out csv
  Filtrer : --id <GlobalId> ou --name <fragment_nom>

- Traitement par lot (glob ou dossier) :
  python bim.py batch "inputs/*.ifc" --classes IfcWall --trees --out-prefix results

- Fusionner plusieurs IFC :
  python bim.py merge "inputs/*.ifc" outputs/merged.ifc --building-name "NomBat" --translate 0 0 0
  Options utiles : --no-guid-fix, --force-contexts, --discipline, --filename-regex

- Résumé multi-fichiers :
  python bim.py summary "inputs/*.ifc" --out outputs/summary.json --pretty

- Vérifier / corriger (doctor) :
  python bim.py doctor "inputs/*.ifc" --fix --out-dir outputs/fixed --force-contexts --force-units

Notes techniques rapides
- Le point d'entrée CLI est [bim.py](bim.py).
- Les opérations IFC (ouverture, fusion, rebase, extraction) sont implémentées dans [functions/ifc_tools.py](functions/ifc_tools.py) — voir les symboles listés ci‑dessus.
- Les sorties sont écrites dans le dossier [outputs/](outputs/) par défaut.

Contribuer
- Ouvrez une issue ou proposez une PR. Respectez le style du projet et ajoutez des tests si pertinent.

Licence
- MIT