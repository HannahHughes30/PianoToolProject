import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

def strip(tag):
    return tag.split("}",1)[1] if "}" in tag else tag

def extract(path):
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for part in root.iter():
        if strip(part.tag) != "part":
            continue
        pid = part.attrib.get("id","P1")

        for measure in list(part):
            if strip(measure.tag) != "measure":
                continue
            mnum = measure.attrib.get("number")

            for note in list(measure):
                if strip(note.tag) != "note":
                    continue

                if any(strip(c.tag)=="rest" for c in list(note)):
                    continue

                pitch_el = next((c for c in list(note) if strip(c.tag)=="pitch"), None)
                if pitch_el is None:
                    continue

                step = next((c.text for c in pitch_el if strip(c.tag)=="step"), "")
                octave = next((c.text for c in pitch_el if strip(c.tag)=="octave"), "")
                pitch = f"{step}{octave}"

                staff_el = next((c for c in list(note) if strip(c.tag)=="staff"), None)
                voice_el = next((c for c in list(note) if strip(c.tag)=="voice"), None)

                staff = int(staff_el.text) if staff_el is not None else 1
                voice = int(voice_el.text) if voice_el is not None else 1

                fing = None
                for notations in list(note):
                    if strip(notations.tag)!="notations":
                        continue
                    for tech in list(notations):
                        if strip(tech.tag)!="technical":
                            continue
                        for f in list(tech):
                            if strip(f.tag)=="fingering":
                                fing = int(f.text)

                if fing is None:
                    continue

                rows.append({
                    "part": pid,
                    "measure": int(mnum),
                    "pitch": pitch,
                    "staff": staff,
                    "voice": voice,
                    "ground_truth": fing
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2])
    df = extract(infile)
    df.to_csv(outfile, index=False)
    print("Saved:", outfile, "rows:", len(df))
