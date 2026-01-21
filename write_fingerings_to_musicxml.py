"""
Write predicted fingerings back into a MusicXML/MXL file.

Usage:
  python write_fingerings_to_musicxml.py <input.(mxl|xml|musicxml)> <predictions.csv> <output.musicxml>
"""

import sys
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def iter_find(elem, want: str):
    for e in elem.iter():
        if strip_ns(e.tag) == want:
            yield e


def find_first(elem, want: str):
    return next(iter_find(elem, want), None)


def read_musicxml_bytes(path: Path) -> bytes:
    suf = path.suffix.lower()
    if suf in {".xml", ".musicxml"}:
        return path.read_bytes()
    if suf != ".mxl":
        raise ValueError("Input must be .mxl, .xml, or .musicxml")

    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()

        container = None
        for n in names:
            if n.lower() == "meta-inf/container.xml":
                container = n
                break

        if container:
            c_root = ET.fromstring(z.read(container))
            rootfile_path = None
            for rf in iter_find(c_root, "rootfile"):
                fp = rf.attrib.get("full-path")
                if fp:
                    rootfile_path = fp
                    break
            if rootfile_path and rootfile_path in names:
                return z.read(rootfile_path)

        xml_candidates = [n for n in names if n.lower().endswith((".xml", ".musicxml")) and "meta-inf" not in n.lower()]
        if not xml_candidates:
            raise RuntimeError(f"No XML found inside {path.name}")
        return z.read(xml_candidates[0])


def ensure_child(parent, tag):
    child = find_first(parent, tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def add_fingering_to_note(note_elem, finger: int):
    notations = ensure_child(note_elem, "notations")
    technical = ensure_child(notations, "technical")
    fing_el = find_first(technical, "fingering")
    if fing_el is None:
        fing_el = ET.SubElement(technical, "fingering")
    fing_el.text = str(int(finger))


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    in_path = Path(sys.argv[1])
    pred_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    if not in_path.exists():
        print(f"❌ Missing input: {in_path}")
        sys.exit(1)
    if not pred_path.exists():
        print(f"❌ Missing predictions: {pred_path}")
        sys.exit(1)

    df = pd.read_csv(pred_path)

    by_part = {}
    for pid, g in df.groupby("part", sort=False):
        by_part[pid] = g.reset_index(drop=True)

    idx_per_part = {pid: 0 for pid in by_part.keys()}

    xml_bytes = read_musicxml_bytes(in_path)
    root = ET.fromstring(xml_bytes)

    assigned = 0
    skipped_rests = 0

    for part in iter_find(root, "part"):
        pid = part.attrib.get("id", "P1")
        if pid not in by_part:
            continue

        part_df = by_part[pid]
        i = idx_per_part[pid]

        for measure in iter_find(part, "measure"):
            for note in iter_find(measure, "note"):
                if find_first(note, "rest") is not None:
                    skipped_rests += 1
                    continue
                if i >= len(part_df):
                    continue

                finger = part_df.loc[i, "predicted_fingering"]
                add_fingering_to_note(note, finger)

                i += 1
                assigned += 1

        idx_per_part[pid] = i

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml_out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    out_path.write_bytes(xml_out)

    print("============================================================")
    print("✓ WROTE FINGERED MUSICXML")
    print(f"Assigned fingerings: {assigned}")
    print(f"Skipped rests: {skipped_rests}")
    print(f"Output: {out_path}")
    print("============================================================")


if __name__ == "__main__":
    main()

