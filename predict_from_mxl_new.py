"""
Predict fingering from existing MusicXML (.xml/.musicxml) or compressed MusicXML (.mxl)

Usage:
  python predict_from_mxl.py <path_to_musicxml_or_mxl_file>

Examples:
  python predict_from_mxl.py data/omr_output/piano4_fourpieces_page5/temp_piano4_fourpieces_page5.mxl
  python predict_from_mxl.py data/test_pieces/mozart_sonfacile.xml
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile


NOTE_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_find(elem, want: str):
    for e in elem.iter():
        if _strip_ns(e.tag) == want:
            yield e


def _find_first(elem, want: str):
    return next(_iter_find(elem, want), None)


def midi_from_pitch(step: str, octave: int, alter: int = 0) -> int:
    base = (octave + 1) * 12 + NOTE_MAP[step]
    return base + alter


def pitch_string(step: str, octave: int, alter: int = 0) -> str:
    if alter == 1:
        return f"{step}#{octave}"
    if alter == -1:
        return f"{step}b{octave}"
    if alter == 2:
        return f"{step}##{octave}"
    if alter == -2:
        return f"{step}bb{octave}"
    return f"{step}{octave}"


def read_musicxml_bytes(path: Path) -> bytes:
    suffix = path.suffix.lower()

    if suffix in {".xml", ".musicxml"}:
        return path.read_bytes()

    if suffix != ".mxl":
        raise ValueError(f"Unsupported file type: {path.suffix} (use .mxl, .xml, or .musicxml)")

    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()

        # Prefer META-INF/container.xml (best practice for MXL)
        container_name = None
        for n in names:
            if n.lower() == "meta-inf/container.xml":
                container_name = n
                break

        if container_name:
            container_xml = z.read(container_name)
            c_root = ET.fromstring(container_xml)

            rootfile_path = None
            for rf in _iter_find(c_root, "rootfile"):
                fp = rf.attrib.get("full-path")
                if fp:
                    rootfile_path = fp
                    break

            if rootfile_path and rootfile_path in names:
                return z.read(rootfile_path)

        # Fallback: choose first plausible xml/musicxml not in META-INF
        xml_candidates = [
            n for n in names
            if n.lower().endswith((".xml", ".musicxml")) and "meta-inf" not in n.lower()
        ]
        if not xml_candidates:
            raise RuntimeError(f"No XML found inside {path.name}. Archive contents: {names}")

        preferred = None
        for n in xml_candidates:
            low = n.lower()
            if "score" in low or "musicxml" in low:
                preferred = n
                break

        return z.read(preferred or xml_candidates[0])


def extract_notes_from_musicxml(path: Path) -> pd.DataFrame:
    content = read_musicxml_bytes(path)
    root = ET.fromstring(content)

    notes = []
    for part in _iter_find(root, "part"):
        part_id = part.attrib.get("id", "P1")

        for measure in _iter_find(part, "measure"):
            measure_num = measure.attrib.get("number", "0")

            for note_elem in _iter_find(measure, "note"):
                if _find_first(note_elem, "rest") is not None:
                    continue

                pitch_elem = _find_first(note_elem, "pitch")
                if pitch_elem is None:
                    continue

                step_el = _find_first(pitch_elem, "step")
                octave_el = _find_first(pitch_elem, "octave")
                if step_el is None or octave_el is None:
                    continue

                step = (step_el.text or "").strip()
                octave = int((octave_el.text or "0").strip())

                alter_el = _find_first(pitch_elem, "alter")
                alter = int((alter_el.text or "0").strip()) if alter_el is not None else 0

                midi = midi_from_pitch(step, octave, alter)
                pstr = pitch_string(step, octave, alter)

                notes.append(
                    {"measure": str(measure_num), "pitch": pstr, "midi": midi, "part": part_id}
                )

    return pd.DataFrame(notes)


def _compute_features(part_df: pd.DataFrame, part_id: str) -> pd.DataFrame:
    df = part_df.copy().reset_index(drop=True)

    df["interval"] = df["midi"].diff().fillna(0)
    df["abs_interval"] = df["interval"].abs()

    df["measure_position"] = df.groupby("measure").cumcount()
    df["measure_length"] = df.groupby("measure")["measure_position"].transform("max") + 1
    df["measure_position_norm"] = df["measure_position"] / df["measure_length"]

    # Keep your original assumption: P2 = left hand
    df["hand_encoding"] = 1 if part_id == "P2" else 0

    df["is_black_key"] = df["pitch"].str.contains(r"#|b").astype(int)
    df["is_step_motion"] = (df["abs_interval"] <= 2).astype(int)
    df["is_large_jump"] = (df["abs_interval"] > 7).astype(int)
    df["is_high_register"] = (df["midi"] > 72).astype(int)
    df["is_low_register"] = (df["midi"] < 60).astype(int)
    df["is_ascending"] = (df["interval"] > 0).astype(int)
    df["is_descending"] = (df["interval"] < 0).astype(int)

    return df


def predict_sequence_with_prev_next(model, df: pd.DataFrame, next_hint=None, reverse=False):
    """
    Predict a fingering sequence left-to-right (default) or right-to-left (reverse=True).

    next_hint: list/array same length as df, providing an estimated next fingering for each note.
              If None, uses 0 for next_fingering.
    """
    n = len(df)
    preds = [0] * n

    indices = range(n - 1, -1, -1) if reverse else range(n)

    for idx in indices:
        row = df.iloc[idx]

        # prev finger depends on direction
        if reverse:
            prev_f = preds[idx + 1] if idx < n - 1 else 0
        else:
            prev_f = preds[idx - 1] if idx > 0 else 0

        # next finger from hint or 0
        next_f = int(next_hint[idx]) if next_hint is not None else 0

        feature_vector = [
            row["midi"],
            row["interval"],
            row["abs_interval"],
            row["measure_position_norm"],
            row["hand_encoding"],
            prev_f,
            next_f,
            row["is_black_key"],
            row["is_step_motion"],
            row["is_large_jump"],
            row["is_high_register"],
            row["is_low_register"],
            row["is_ascending"],
            row["is_descending"],
        ]

        preds[idx] = int(model.predict([feature_vector])[0])

    return preds


def predict_fingering_for_notes(notes_df: pd.DataFrame, model):
    """
    Predict fingering using a bidirectional (past+future) context strategy:

    1) Backward pass predicts a plausible fingering "future plan"
    2) Convert that into next_hint for each note
    3) Final forward pass uses both prev_f and next_hint
    """
    if notes_df.empty:
        return notes_df

    results = []

    for part_id in notes_df["part"].unique():
        part_df = notes_df[notes_df["part"] == part_id].copy().reset_index(drop=True)

        # Compute engineered features
        part_df = _compute_features(part_df, part_id)

        # PASS A: backward (creates a future-plan sequence)
        backward_preds = predict_sequence_with_prev_next(model, part_df, next_hint=None, reverse=True)

        # Build next_hint: for note i, next_hint[i] = predicted fingering at i+1 (from backward pass)
        next_hint = [0] * len(part_df)
        for i in range(len(part_df) - 1):
            next_hint[i] = backward_preds[i + 1]
        next_hint[-1] = 0

        # PASS B: final forward using next_hint
        final_preds = predict_sequence_with_prev_next(model, part_df, next_hint=next_hint, reverse=False)

        part_df["predicted_fingering"] = final_preds
        results.append(part_df)

    return pd.concat(results, ignore_index=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_from_mxl.py <musicxml_or_mxl_file>")
        print("Supports: .mxl, .xml, .musicxml")
        sys.exit(1)

    in_path = Path(sys.argv[1])

    if not in_path.exists():
        print(f"❌ File not found: {in_path}")
        sys.exit(1)

    if in_path.suffix.lower() not in {".mxl", ".xml", ".musicxml"}:
        print(f"❌ Unsupported file type: {in_path.suffix}")
        print("Use a .mxl, .xml, or .musicxml file.")
        sys.exit(1)

    print("=" * 60)
    print("FINGERING PREDICTION")
    print("=" * 60)
    print(f"\nInput: {in_path.name}")

    # Extract notes
    print("\nExtracting notes...")
    notes_df = extract_notes_from_musicxml(in_path)
    print(f"✓ Found {len(notes_df)} notes")

    if notes_df.empty:
        print("⚠️ No pitched notes found (all rests / no pitch elements).")
        sys.exit(0)

    # Load model
    print("\nLoading model...")
    model_path = Path("models/fingering_model.pkl")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as f:
        saved = pickle.load(f)
        model = saved["model"]
    print("✓ Model loaded")

    # Predict
    print("\nPredicting fingering (bidirectional context)...")
    predictions = predict_fingering_for_notes(notes_df, model)

    # Show results
    print(f"\n{'=' * 60}")
    print("PREDICTIONS")
    print(f"{'=' * 60}")

    for part_id in predictions["part"].unique():
        part_data = predictions[predictions["part"] == part_id]
        hand = "LEFT HAND" if part_id == "P2" else "RIGHT HAND"

        print(f"\n{hand} ({len(part_data)} notes):")
        print(f"{'Measure':<10} {'Note':<8} {'MIDI':<6} {'Fingering'}")
        print("-" * 40)

        for _, row in part_data.iterrows():
            print(f"{row['measure']:<10} {row['pitch']:<8} {row['midi']:<6} {row['predicted_fingering']}")

    # Save
    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{in_path.stem}_predictions.csv"
    predictions.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"✓ Saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

