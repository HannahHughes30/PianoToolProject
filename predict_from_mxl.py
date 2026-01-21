#!/usr/bin/env python3
"""
Human-like fingering prediction from MXL / MusicXML.

Key fix:
- chord_id is now GLOBAL across the whole score (no longer resets per measure)
  so chord grouping doesn't incorrectly connect notes across measures.
- HARD stepwise constraints apply correctly afterward.

Usage:
  python predict_from_mxl.py <path_to_mxl_or_xml>
"""

import sys
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile

NOTE_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
FINGERS = [1, 2, 3, 4, 5]
FINGER_OFFSET = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7}


# ---------------- XML helpers ----------------
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def _iter_find(elem, want: str):
    for e in elem.iter():
        if _strip_ns(e.tag) == want:
            yield e

def _find_first(elem, want: str):
    return next(_iter_find(elem, want), None)

def _has_child(elem, want: str) -> bool:
    return _find_first(elem, want) is not None


def read_musicxml_bytes(path: Path) -> bytes:
    suf = path.suffix.lower()
    if suf in (".xml", ".musicxml"):
        return path.read_bytes()
    if suf != ".mxl":
        raise ValueError(f"Unsupported file type: {suf}")

    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()

        container = next((n for n in names if n.lower() == "meta-inf/container.xml"), None)
        if container:
            root = ET.fromstring(z.read(container))
            for rf in _iter_find(root, "rootfile"):
                fp = rf.attrib.get("full-path")
                if fp and fp in names:
                    return z.read(fp)

        xmls = [n for n in names if n.lower().endswith((".xml", ".musicxml")) and "meta-inf" not in n.lower()]
        if not xmls:
            raise RuntimeError(f"No XML found inside: {path}")
        return z.read(xmls[0])


def midi_from_pitch(step: str, octave: int, alter: int = 0) -> int:
    return (octave + 1) * 12 + NOTE_MAP[step] + alter

def pitch_string(step: str, octave: int, alter: int) -> str:
    if alter == 1:
        return f"{step}#{octave}"
    if alter == -1:
        return f"{step}b{octave}"
    return f"{step}{octave}"


# ---------------- Extraction (FIXED chord_id) ----------------
def extract_notes_from_musicxml(path: Path) -> pd.DataFrame:
    root = ET.fromstring(read_musicxml_bytes(path))
    rows = []
    idx = 0

    for part in _iter_find(root, "part"):
        part_id = part.attrib.get("id", "P1")

        # ✅ GLOBAL chord counter for the whole part (NOT per measure)
        global_chord_counter = 0
        last_chord_id = None

        for measure in _iter_find(part, "measure"):
            mnum_raw = measure.attrib.get("number", "0")
            try:
                mnum = int(mnum_raw)
            except Exception:
                mnum = 0

            for note in _iter_find(measure, "note"):
                if _has_child(note, "rest"):
                    continue

                pitch = _find_first(note, "pitch")
                if pitch is None:
                    continue

                step_el = _find_first(pitch, "step")
                oct_el = _find_first(pitch, "octave")
                if step_el is None or oct_el is None:
                    continue

                step = (step_el.text or "").strip()
                octave = int((oct_el.text or "0").strip())

                alter_el = _find_first(pitch, "alter")
                alter = int((alter_el.text or "0").strip()) if alter_el is not None else 0

                staff_el = _find_first(note, "staff")
                staff = int((staff_el.text or "1").strip()) if staff_el is not None else 1

                voice_el = _find_first(note, "voice")
                voice = int((voice_el.text or "1").strip()) if voice_el is not None else 1

                is_chord_tone = _has_child(note, "chord")

                # ✅ chord_id now unique across the whole piece
                if not is_chord_tone:
                    global_chord_counter += 1
                    last_chord_id = global_chord_counter
                chord_id = last_chord_id

                rows.append({
                    "idx": idx,
                    "measure": mnum,
                    "pitch": pitch_string(step, octave, alter),
                    "midi": midi_from_pitch(step, octave, alter),
                    "part": part_id,
                    "staff": staff,
                    "voice": voice,
                    "chord_id": chord_id,
                })
                idx += 1

    return pd.DataFrame(rows)


# ---------------- fallback ----------------
def greedy_stepwise_fallback(midi_seq):
    if not midi_seq:
        return []
    f = 3
    out = [f]
    for i in range(1, len(midi_seq)):
        d = midi_seq[i] - midi_seq[i - 1]
        if d == 0:
            out.append(f)
            continue
        direction = 1 if d > 0 else -1
        abs_semi = abs(d)

        if abs_semi <= 2:
            f = max(1, min(5, f + direction))
        elif abs_semi <= 4:
            f = max(1, min(5, f + 2 * direction))
        elif abs_semi >= 10:
            f = 5 if direction > 0 else 1
        else:
            f = max(1, min(5, f + direction))
        out.append(f)
    return out


# ---------------- chord fingering ----------------
def assign_chord_fingers(chord_midis, is_left: bool):
    mids = sorted(chord_midis)
    span = mids[-1] - mids[0]

    if len(mids) == 1:
        return {mids[0]: 3}

    if len(mids) == 2:
        if span >= 11:
            fingers = (1, 5) if not is_left else (5, 1)
        elif span >= 6:
            fingers = (1, 4) if not is_left else (4, 1)
        else:
            fingers = (1, 3) if not is_left else (3, 1)
        return {mids[0]: fingers[0], mids[1]: fingers[1]}

    if len(mids) == 3:
        fingers = (1, 3, 5) if not is_left else (5, 3, 1)
        return {mids[i]: fingers[i] for i in range(3)}

    if not is_left:
        finger_list = [1, 2, 4, 5]
    else:
        finger_list = [5, 4, 2, 1]
    finger_list = (finger_list * 10)[:len(mids)]
    return {mids[i]: finger_list[i] for i in range(len(mids))}


# ---------------- HARD stepwise DP ----------------
def decode_position_dp(midi_seq, measure_seq, is_left: bool):
    n = len(midi_seq)
    if n == 0:
        return []
    if n == 1:
        return [3]

    def bases_for(m):
        return list(set(m - FINGER_OFFSET[f] for f in FINGERS))

    dp = [{} for _ in range(n)]

    m0 = midi_seq[0]
    for b in bases_for(m0):
        best = None
        for f in FINGERS:
            err = abs(m0 - (b + FINGER_OFFSET[f]))
            cost = 6.0 * err + (0.0 if f in (2, 3, 4) else 1.5)
            if best is None or cost < best[0]:
                best = (cost, None, f)
        dp[0][b] = best

    for i in range(1, n):
        mi = midi_seq[i]
        pm = midi_seq[i - 1]
        delta = mi - pm
        abs_semi = abs(delta)
        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
        at_measure_start = (measure_seq[i] != measure_seq[i - 1])

        row = {}
        for b in bases_for(mi):
            best_cost = None
            best_prev_b = None
            best_f = None

            for pb, (pcost, _, pf) in dp[i - 1].items():
                for f in FINGERS:
                    # HARD rules
                    if abs_semi == 0 and f != pf:
                        continue

                    if abs_semi <= 2 and direction != 0:
                        if f != pf + direction:
                            continue

                    cost = pcost
                    cost += 2.5 * abs(b - pb)

                    err = abs(mi - (b + FINGER_OFFSET[f]))
                    cost += 7.0 * err

                    if abs_semi <= 4 and abs(f - pf) > 2:
                        cost += 120.0

                    if delta >= 5 and f == 1:
                        cost += 90.0

                    if delta <= -5:
                        if pf >= 4 and f <= 2:
                            cost += 160.0
                        if abs(f - pf) >= 3:
                            cost += 70.0

                    if at_measure_start and (not is_left) and f == 1:
                        if i + 1 < n and midi_seq[i + 1] < mi:
                            cost += 120.0

                    cost += 0.0 if f in (2, 3, 4) else 1.0

                    if best_cost is None or cost < best_cost:
                        best_cost, best_prev_b, best_f = cost, pb, f

            if best_cost is not None:
                row[b] = (best_cost, best_prev_b, best_f)

        dp[i] = row
        if not dp[i]:
            return greedy_stepwise_fallback(midi_seq)

    end_base = min(dp[-1].items(), key=lambda kv: kv[1][0])[0]
    fingers = []
    b = end_base
    for i in range(n - 1, -1, -1):
        _, prev_b, f = dp[i][b]
        fingers.append(int(f))
        b = prev_b if prev_b is not None else b
    fingers.reverse()
    return fingers


def predict_fingering_for_notes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = []
    for (part, staff, voice), grp in df.groupby(["part", "staff", "voice"], sort=False):
        grp = grp.sort_values("idx").reset_index(drop=True).copy()
        is_left = (int(staff) == 2)

        grp["predicted_fingering"] = 0

        # chord detection is now correct because chord_id is unique
        chord_sizes = grp.groupby("chord_id")["midi"].transform("count")
        chord_mask = chord_sizes >= 2

        for chord_id, cg in grp[chord_mask].groupby("chord_id", sort=False):
            mids = cg["midi"].astype(int).tolist()
            chord_map = assign_chord_fingers(mids, is_left=is_left)
            for row_i in cg.index:
                m = int(grp.loc[row_i, "midi"])
                grp.loc[row_i, "predicted_fingering"] = chord_map.get(m, 3)

        single = grp[~chord_mask].copy()
        if not single.empty:
            midi_seq = single["midi"].astype(int).tolist()
            measure_seq = single["measure"].astype(int).tolist()
            fingers = decode_position_dp(midi_seq, measure_seq, is_left=is_left)
            grp.loc[single.index, "predicted_fingering"] = fingers

        grp["predicted_fingering"] = grp["predicted_fingering"].replace(0, 3).astype(int)
        out.append(grp)

    return pd.concat(out, ignore_index=True).sort_values("idx").reset_index(drop=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_from_mxl.py <file.mxl|xml|musicxml>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print("❌ File not found:", path)
        sys.exit(1)

    notes = extract_notes_from_musicxml(path)
    print(f"Extracted notes: {len(notes)}")

    preds = predict_fingering_for_notes(notes)

    out_path = Path("data/predictions") / f"{path.stem}_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()

