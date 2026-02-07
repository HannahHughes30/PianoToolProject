import os
import sys
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory, url_for

import pandas as pd

# ------------------------------------------------------------
# Paths / sys.path so imports work
# ------------------------------------------------------------
WEBAPP_DIR = Path(__file__).resolve().parent                 # .../webapp
PROJECT_ROOT = WEBAPP_DIR.parent                              # .../PianoToolProject
BACKEND_DIR = WEBAPP_DIR / "backend"

sys.path.insert(0, str(PROJECT_ROOT))  # so: import predict_from_mxl works
sys.path.insert(0, str(BACKEND_DIR))   # so: import write_fingerings_to_musicxml works

from predict_from_mxl import (
    extract_notes_from_musicxml,
    compute_left_hand_groups,
    predict_fingering_for_notes,
)

# import backend writer as a module (we'll call its functions)
import xml.etree.ElementTree as ET
import zipfile

# ------------------------------------------------------------
# Flask app
# ------------------------------------------------------------
app = Flask(__name__)

UPLOAD_DIR = WEBAPP_DIR / "uploads"
PRED_DIR = WEBAPP_DIR / "data" / "predictions"
ANNOTATED_DIR = WEBAPP_DIR / "annotated"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".xml", ".musicxml", ".mxl"}


# ------------------------------------------------------------
# Minimal MusicXML writer (adapted from your write_fingerings_to_musicxml.py)
# ------------------------------------------------------------
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _ns_of(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}")[0].strip("{")
    return ""


def _q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}" if ns else tag


def _find_first_child(elem, want: str):
    for c in list(elem):
        if _strip_ns(c.tag) == want:
            return c
    return None


def _ensure_child(parent, ns: str, tag: str):
    child = _find_first_child(parent, tag)
    if child is None:
        child = ET.SubElement(parent, _q(ns, tag))
    return child


def _read_musicxml_bytes(path: Path) -> bytes:
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
            for rf in c_root.iter():
                if _strip_ns(rf.tag) == "rootfile":
                    fp = rf.attrib.get("full-path")
                    if fp and fp in names:
                        return z.read(fp)

        xml_candidates = [
            n for n in names
            if n.lower().endswith((".xml", ".musicxml")) and "meta-inf" not in n.lower()
        ]
        if not xml_candidates:
            raise RuntimeError(f"No XML found inside {path.name}")
        return z.read(xml_candidates[0])


def _add_fingering_to_note(note_elem, ns: str, finger: int, placement: str = "above"):
    notations = _ensure_child(note_elem, ns, "notations")
    technical = _ensure_child(notations, ns, "technical")

    fing_el = _find_first_child(technical, "fingering")
    if fing_el is None:
        fing_el = ET.SubElement(technical, _q(ns, "fingering"))

    fing_el.text = str(int(finger))
    if placement:
        fing_el.set("placement", placement)


def _collect_xml_notes(root):
    """
    Returns dict: (part, staff, voice) -> list[note_elem] in document order
    Skips rests.
    """
    groups = {}

    for part in root.iter():
        if _strip_ns(part.tag) != "part":
            continue
        pid = part.attrib.get("id", "P1")

        for measure in list(part):
            if _strip_ns(measure.tag) != "measure":
                continue

            for note in list(measure):
                if _strip_ns(note.tag) != "note":
                    continue
                if _find_first_child(note, "rest") is not None:
                    continue

                staff_el = _find_first_child(note, "staff")
                voice_el = _find_first_child(note, "voice")
                staff = int(staff_el.text.strip()) if staff_el is not None else 1
                voice = int(voice_el.text.strip()) if voice_el is not None else 1

                key = (pid, staff, voice)
                groups.setdefault(key, []).append(note)

    return groups


def write_fingerings_to_musicxml(input_path: Path, preds_df: pd.DataFrame, output_path: Path):
    """
    Writes predicted_fingering into MusicXML and saves to output_path.
    """
    needed = {"predicted_fingering", "staff", "voice"}
    missing = [c for c in needed if c not in preds_df.columns]
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")

    df = preds_df.copy()
    if "part" not in df.columns:
        df["part"] = "P1"

    df["part"] = df["part"].astype(str)
    df["staff"] = df["staff"].astype(int)
    df["voice"] = df["voice"].astype(int)
    df["predicted_fingering"] = df["predicted_fingering"].astype(int)

    xml_bytes = _read_musicxml_bytes(input_path)
    root = ET.fromstring(xml_bytes)
    ns = _ns_of(root.tag)

    xml_groups = _collect_xml_notes(root)

    for (pid, staff, voice), g in df.groupby(["part", "staff", "voice"], sort=False):
        key = (pid, int(staff), int(voice))
        xml_notes = xml_groups.get(key, [])
        csv_rows = g.reset_index(drop=True)

        n = min(len(xml_notes), len(csv_rows))
        for i in range(n):
            finger = int(csv_rows.loc[i, "predicted_fingering"])
            _add_fingering_to_note(xml_notes[i], ns, finger, placement="above")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            result = {"error": "No file selected."}
            return render_template("index.html", result=result)

        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            result = {"error": f"Unsupported file type: {ext}. Upload .xml/.musicxml/.mxl"}
            return render_template("index.html", result=result)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = Path(f.filename).name.replace(" ", "_")
        upload_name = f"{stamp}_{safe_name}"
        upload_path = UPLOAD_DIR / upload_name
        f.save(upload_path)

        # 1) predict
        notes = extract_notes_from_musicxml(upload_path)
        left_groups = compute_left_hand_groups(notes)
        preds = predict_fingering_for_notes(notes, left_hand_groups=left_groups)

        # 2) save csv
        pred_csv = PRED_DIR / f"{Path(upload_name).stem}_predictions.csv"
        preds.to_csv(pred_csv, index=False)

        # 3) write annotated MusicXML
        out_musicxml = ANNOTATED_DIR / f"{Path(upload_name).stem}_fingered.musicxml"
        write_fingerings_to_musicxml(upload_path, preds, out_musicxml)

        # counts for UI
        rh = int((preds["hand"] == "RH").sum()) if "hand" in preds.columns else 0
        lh = int((preds["hand"] == "LH").sum()) if "hand" in preds.columns else 0

        left_groups_str = [f"{p}:{s}" for (p, s) in sorted(left_groups)]

        result = {
            "ok": True,
            "uploaded": upload_name,
            "rows": int(len(preds)),
            "rh": rh,
            "lh": lh,
            "left_groups": left_groups_str,
            "musicxml_url": url_for("serve_musicxml", filename=out_musicxml.name),
            "csv_url": url_for("download_csv", filename=pred_csv.name),
        }

    return render_template("index.html", result=result)


@app.route("/annotated/<path:filename>")
def serve_musicxml(filename):
    return send_from_directory(
        ANNOTATED_DIR,
        filename,
        mimetype="application/vnd.recordare.musicxml+xml",
    )


@app.route("/predictions/<path:filename>")
def download_csv(filename):
    return send_from_directory(PRED_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # use 5001 to avoid macOS ControlCenter on 5000
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=True)

