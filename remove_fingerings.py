import sys
from pathlib import Path
import xml.etree.ElementTree as ET

def strip(tag):
    return tag.split("}",1)[1] if "}" in tag else tag

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

tree = ET.parse(in_path)
root = tree.getroot()

removed = 0
for note in root.iter():
    if strip(note.tag) != "note":
        continue
    for notations in list(note):
        if strip(notations.tag)!="notations":
            continue
        for tech in list(notations):
            if strip(tech.tag)!="technical":
                continue
            for fing in list(tech):
                if strip(fing.tag)=="fingering":
                    tech.remove(fing)
                    removed += 1

tree.write(out_path, encoding="utf-8", xml_declaration=True)
print("Removed", removed, "fingerings")
