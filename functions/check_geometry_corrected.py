#!/usr/bin/env python3
# functions/check_geometry_corrected.py
import ifcopenshell
from collections import Counter, defaultdict
from pathlib import Path

p = Path("output/merged.ifc")
m = ifcopenshell.open(str(p))
print("Fichier:", p, "taille (Mo):", p.stat().st_size/1024/1024)

# Count entities that have a Representation with Items (>0), for any subtype
count_with_repr = 0
count_with_items = 0
sample = []
type_counter = Counter()

for e in m.by_type("IfcRoot"):
    t = e.is_a()
    type_counter[t] += 1
    rep = getattr(e, "Representation", None)
    if rep and getattr(rep, "Representations", None):
        count_with_repr += 1
        items_total = 0
        for r in rep.Representations:
            items = getattr(r, "Items", None)
            if items:
                items_total += len(items)
        if items_total > 0:
            count_with_items += 1
            if len(sample) < 50:
                sample.append((t, getattr(e, "GlobalId", None), getattr(e, "Name", None), items_total))

print("Total IfcRoot:", len(list(m.by_type("IfcRoot"))))
print("Entités (any Representation):", count_with_repr)
print("Entités (Representation avec Items>0):", count_with_items)
print("Exemples (max 50) d'entités avec Items>0 :")
for s in sample:
    print(" ", s)
print("\nTop 20 types (extrait):")
for t,c in type_counter.most_common(20):
    print(" ", t, c)
