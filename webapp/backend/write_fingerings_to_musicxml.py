"""
Write predicted fingerings back into a MusicXML/MXL file by SEQUENCE per (part, staff, voice).

✅ Works with chords automatically (because chord tones are still <note> elements in order)
✅ Works with left/right hands (staff/voice separation)
✅ No fragile chord_id matching

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


def ns_of(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}")[0].strip("{")
    return ""


def q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}" if ns else tag


def find_first_child(elem, want: str):
    for c in list(elem):
        if strip_ns(c.tag) == want:
            return c
    return None


def ensure_child(parent, ns: str, tag: str):
    child = find_first_child(parent, tag)
    if child is None:
        child = ET.SubElement(parent, q(ns, tag))
    return child


def read_musicxml_bytes(path: Path) -> bytes:
    suf = path.suffix.lower()
    if suf in {".xml", ".musicxml"}:
        return path.read_bytes()
    if suf != ".mxl":
        raise ValueError("Input must be .mxl, .xml, or .musicxml")

    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()

        container = next((n for n in names if n.lower() == "meta-inf/container.xml"), None)
        if container:
            c_root = ET.fromstring(z.read(container))
            rootfile_path = None
            for rf in c_root.iter():
                if strip_ns(rf.tag) == "rootfile":
                    fp = rf.attrib.get("full-path")
                    if fp:
                        rootfile_path = fp
                        break
            if rootfile_path and rootfile_path in names:
                return z.read(rootfile_path)

        xml_candidates = [
            n for n in names
            if n.lower().endswith((".xml", ".musicxml")) and "meta-inf" not in n.lower()
        ]
        if not xml_candidates:
            raise RuntimeError(f"No XML found inside {path.name}")
        return z.read(xml_candidates[0])


def add_fingering_to_note(note_elem, ns: str, finger: int, placement: str = "above"):
    notations = ensure_child(note_elem, ns, "notations")
    technical = ensure_child(notations, ns, "technical")

    fing_el = find_first_child(technical, "fingering")
    if fing_el is None:
        fing_el = ET.SubElement(technical, q(ns, "fingering"))

    fing_el.text = str(int(finger))
    if placement:
        fing_el.set("placement", placement)


def collect_xml_notes(root):
    """
    Returns dict: (part, staff, voice) -> list[note_elem] in document order
    Skips rests (matches your predictor).
    """
    groups = {}

    for part in root.iter():
        if strip_ns(part.tag) != "part":
            continue
        pid = part.attrib.get("id", "P1")

        for measure in list(part):
            if strip_ns(measure.tag) != "measure":
                continue

            for note in list(measure):
                if strip_ns(note.tag) != "note":
                    continue
                if find_first_child(note, "rest") is not None:
                    continue

                staff_el = find_first_child(note, "staff")
                voice_el = find_first_child(note, "voice")
                staff = int(staff_el.text.strip()) if staff_el is not None else 1
                voice = int(voice_el.text.strip()) if voice_el is not None else 1

                key = (pid, staff, voice)
                groups.setdefault(key, []).append(note)

    return groups


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

    for col in ["predicted_fingering", "staff", "voice", "measure"]:
        if col not in df.columns:
            print(f"❌ predictions.csv missing column: {col}")
            print("   Found:", df.columns.tolist())
            sys.exit(1)

    has_part = "part" in df.columns
    df2 = df.copy()
    df2["part"] = df2["part"].astype(str) if has_part else "P1"
    df2["staff"] = df2["staff"].astype(int)
    df2["voice"] = df2["voice"].astype(int)
    df2["predicted_fingering"] = df2["predicted_fingering"].astype(int)

    xml_bytes = read_musicxml_bytes(in_path)
    root = ET.fromstring(xml_bytes)
    ns = ns_of(root.tag)

    xml_groups = collect_xml_notes(root)

    assigned = 0
    mismatched_groups = []

    for (pid, staff, voice), g in df2.groupby(["part", "staff", "voice"], sort=False):
        key = (pid, int(staff), int(voice))
        xml_notes = xml_groups.get(key, [])
        csv_rows = g.reset_index(drop=True)

        if len(xml_notes) != len(csv_rows):
            mismatched_groups.append((key, len(xml_notes), len(csv_rows)))

        n = min(len(xml_notes), len(csv_rows))
        for i in range(n):
            finger = int(csv_rows.loc[i, "predicted_fingering"])
            add_fingering_to_note(xml_notes[i], ns, finger, placement="above")
            assigned += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))

    print("============================================================")
    print("✓ WROTE FINGERED MUSICXML (SEQUENCE PER STAFF/VOICE)")
    print(f"Input:  {in_path}")
    print(f"Pred:   {pred_path}")
    print(f"Output: {out_path}")
    print("------------------------------------------------------------")
    print(f"Assigned fingerings: {assigned}")
    if mismatched_groups:
        print("⚠️  Count mismatches per (part,staff,voice):")
        for key, xml_n, csv_n in mismatched_groups:
            print(f"  {key}: XML notes={xml_n} vs CSV rows={csv_n}")
    else:
        print("✓ All groups matched in length.")
    print("============================================================")


if __name__ == "__main__":
    main()

