#!/usr/bin/env python3
"""
Human-like fingering prediction from MXL / MusicXML.

Combined approach:
- DP for single-note passages (with local lookahead window for hand position stability)
- Voice-leading chord fingering selection (prevents nonsensical rolling/jumping)
- Dyad "laddering" for common repeated 2-note chord progressions (e.g., 13 24 35)
- Stronger prevention of repeating the same finger on different consecutive notes
- NEW: discourage overusing outer fingers (5/1) away from range edges + penalize awkward 5->3 (RH) / 1->3 (LH)
"""

import sys
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile

NOTE_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
FINGERS = [1, 2, 3, 4, 5]
FINGER_OFFSET = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7}


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

        xmls = [
            n for n in names
            if n.lower().endswith((".xml", ".musicxml")) and "meta-inf" not in n.lower()
        ]
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


def _parse_clefs_from_attributes(measure: ET.Element) -> dict:
    out = {}
    attrs = _find_first(measure, "attributes")
    if attrs is None:
        return out
    for clef in _iter_find(attrs, "clef"):
        num_attr = clef.attrib.get("number")
        try:
            staff_num = int(num_attr) if num_attr is not None else 1
        except Exception:
            staff_num = 1
        sign = _find_first(clef, "sign")
        line = _find_first(clef, "line")
        if sign is None or sign.text is None:
            continue
        s = (sign.text or "").strip()
        l = (line.text or "").strip() if line is not None and line.text else ""
        out[staff_num] = f"{s}{l}" if l else s
    return out


def extract_notes_from_musicxml(path: Path) -> pd.DataFrame:
    root = ET.fromstring(read_musicxml_bytes(path))
    rows = []
    idx = 0

    for part in _iter_find(root, "part"):
        part_id = part.attrib.get("id", "P1")
        global_chord_counter = 0
        last_chord_id = None
        current_clef_by_staff = {}

        for measure in _iter_find(part, "measure"):
            clefs_update = _parse_clefs_from_attributes(measure)
            if clefs_update:
                current_clef_by_staff.update(clefs_update)

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
                if not is_chord_tone:
                    global_chord_counter += 1
                    last_chord_id = global_chord_counter
                chord_id = last_chord_id

                clef_for_staff = current_clef_by_staff.get(staff, "")

                rows.append({
                    "idx": idx,
                    "measure": mnum,
                    "pitch": pitch_string(step, octave, alter),
                    "midi": midi_from_pitch(step, octave, alter),
                    "part": str(part_id),
                    "staff": int(staff),
                    "voice": int(voice),
                    "clef": str(clef_for_staff),
                    "chord_id": int(chord_id) if chord_id is not None else 0,
                })
                idx += 1

    return pd.DataFrame(rows)


def compute_left_hand_groups(df: pd.DataFrame) -> set[tuple[str, int]]:
    left = set()
    if df.empty:
        return left

    for part, g in df.groupby("part", sort=False):
        staves = sorted(set(g["staff"].astype(int).unique()))
        if 2 in staves:
            left.add((str(part), 2))

    if left:
        return left

    parts = set(df["part"].astype(str).unique())
    if "P2" in parts:
        left.add(("P2", 1))
        return left

    if "clef" in df.columns:
        for (part, staff), g in df.groupby(["part", "staff"], sort=False):
            clefs = g["clef"].astype(str)
            f_share = clefs.str.startswith("F", na=False).sum() / max(1, len(clefs))
            if f_share >= 0.25:
                left.add((str(part), int(staff)))

    return left


def assign_chord_fingers(chord_midis, is_left: bool):
    mids = sorted(map(int, chord_midis))
    span = mids[-1] - mids[0]
    n = len(mids)

    if n == 1:
        return {mids[0]: 3}
    if n == 2:
        if span >= 11:
            base = [1, 5]
        elif span >= 6:
            base = [1, 4]
        else:
            base = [1, 3]
        if is_left:
            base = list(reversed(base))
        return {mids[0]: base[0], mids[1]: base[1]}
    if n == 3:
        base = [1, 3, 5]
        if is_left:
            base = list(reversed(base))
        return {mids[i]: base[i] for i in range(3)}

    fingers = []
    for i in range(n):
        f = 1 + round(i * (4 / (n - 1)))
        fingers.append(max(1, min(5, int(f))))
    if is_left:
        fingers = list(reversed(fingers))
    return {mids[i]: fingers[i] for i in range(n)}


# ──────────────────────────────────────────────────────────────────────────────
# Chords: stable voice-leading + dyad progression laddering
# ──────────────────────────────────────────────────────────────────────────────

def chord_fingering_candidates(chord_midis, is_left: bool):
    mids = sorted(map(int, chord_midis))
    span = mids[-1] - mids[0]
    n = len(mids)

    if n == 1:
        return [{mids[0]: 3}]
    if n >= 3 or span >= 11:
        return [assign_chord_fingers(mids, is_left)]

    if span >= 6:
        base_sets = [[1, 4], [2, 5]]
    else:
        base_sets = [[1, 3], [2, 4], [3, 5]]

    candidates = []
    for fs in base_sets:
        use = list(reversed(fs)) if is_left else fs
        candidates.append({mids[0]: use[0], mids[1]: use[1]})
    return candidates


def score_chord_choice(prev_map, curr_map, prev_mids, curr_mids):
    cost = 0.0

    fingers = list(curr_map.values())
    if len(set(fingers)) != len(fingers):
        cost += 10_000.0

    if prev_map is not None:
        common = set(prev_map.keys()) & set(curr_map.keys())
        for m in common:
            if prev_map[m] != curr_map[m]:
                cost += 350.0

        for m in curr_mids:
            nearest = min(prev_mids, key=lambda pm: abs(pm - m))
            prev_f = prev_map.get(nearest, None)
            if prev_f is not None:
                cost += 30.0 * abs(curr_map[m] - prev_f)

    if len(curr_mids) == 2:
        span = curr_mids[1] - curr_mids[0]
        fspan = abs(curr_map[curr_mids[1]] - curr_map[curr_mids[0]])
        cost += 10.0 * abs((span / 2.0) - (fspan * 2.0))

    return cost


def choose_chord_fingering(prev_map, chord_midis, is_left: bool):
    mids = sorted(map(int, chord_midis))
    candidates = chord_fingering_candidates(mids, is_left=is_left)

    if prev_map is None:
        def neutral_score(cm):
            return sum(1.0 if f in (2, 3, 4) else 2.0 for f in cm.values())
        return min(candidates, key=neutral_score)

    prev_mids = sorted(map(int, prev_map.keys()))
    best = None
    best_cost = None
    for cm in candidates:
        c = score_chord_choice(prev_map, cm, prev_mids, mids)
        if best_cost is None or c < best_cost:
            best_cost = c
            best = cm
    return best


def _dyad_pair_for_index(is_left: bool, idx: int):
    rh_pairs = [(1, 3), (2, 4), (3, 5)]
    a, b = rh_pairs[max(0, min(2, idx))]
    return (b, a) if is_left else (a, b)


def detect_and_build_dyad_ladder_maps(chord_midis_by_id: list[tuple[int, list[int]]], is_left: bool):
    out = {}
    dyads = [(cid, sorted(map(int, mids))) for cid, mids in chord_midis_by_id if len(mids) == 2]
    if len(dyads) < 3:
        return out

    i = 0
    while i < len(dyads) - 2:
        _, m0 = dyads[i]
        _, m1 = dyads[i + 1]
        _, m2 = dyads[i + 2]

        span0 = m0[1] - m0[0]
        span1 = m1[1] - m1[0]
        span2 = m2[1] - m2[0]
        if not (abs(span0 - span1) <= 1 and abs(span1 - span2) <= 1 and span0 <= 5 and span1 <= 5 and span2 <= 5):
            i += 1
            continue

        d1 = m1[0] - m0[0]
        d2 = m2[0] - m1[0]
        if d1 == 0 or d2 == 0:
            i += 1
            continue

        same_dir = (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0)
        if not same_dir:
            i += 1
            continue

        def stepwise(prev, curr):
            return abs(curr[0] - prev[0]) <= 2 and abs(curr[1] - prev[1]) <= 2

        if not (stepwise(m0, m1) and stepwise(m1, m2)):
            i += 1
            continue

        direction = 1 if d1 > 0 else -1

        run_start = i
        run_end = i + 2
        j = i + 3
        while j < len(dyads):
            _, prev = dyads[j - 1]
            cidj, curr = dyads[j]
            if not (abs((curr[1] - curr[0]) - span0) <= 1 and (curr[1] - curr[0]) <= 5 and stepwise(prev, curr)):
                break
            if (direction == 1 and curr[0] - prev[0] <= 0) or (direction == -1 and curr[0] - prev[0] >= 0):
                break
            run_end = j
            j += 1

        idx = 0 if direction == 1 else 2
        for k in range(run_start, run_end + 1):
            cid, mids = dyads[k]
            f_low, f_high = _dyad_pair_for_index(is_left=is_left, idx=idx)
            out[cid] = {mids[0]: f_low, mids[1]: f_high}
            idx = min(2, idx + 1) if direction == 1 else max(0, idx - 1)

        i = run_end + 1

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Single-note DP and smoothing
# ──────────────────────────────────────────────────────────────────────────────

def get_position_finger(midi_note, range_low, range_high, is_left: bool) -> int:
    if range_high == range_low:
        return 3
    pos = (midi_note - range_low) / (range_high - range_low)
    if is_left:
        if pos <= 0.2: return 5
        if pos <= 0.4: return 4
        if pos <= 0.6: return 3
        if pos <= 0.8: return 2
        return 1
    else:
        if pos <= 0.2: return 1
        if pos <= 0.4: return 2
        if pos <= 0.6: return 3
        if pos <= 0.8: return 4
        return 5


def _edge_penalty(mi: int, lo: int, hi: int, f: int, is_left: bool) -> float:
    """
    Discourage using outer fingers (5/1) unless we're near the local range edge.
    This directly fixes 'too many 5s' (RH) and 'too many 1s' (LH) in mid-range.
    """
    if hi <= lo:
        return 0.0
    # how close to the relevant edge (0 near edge, 1 far away)
    if not is_left:
        dist_top = (hi - mi) / max(1, hi - lo)  # 0 near top, 1 near bottom
        dist_bot = (mi - lo) / max(1, hi - lo)
        if f == 5:
            # prefer 5 only near top
            return 600.0 * max(0.0, dist_top - 0.25)
        if f == 1:
            # prefer 1 only near bottom
            return 350.0 * max(0.0, dist_bot - 0.30)
    else:
        dist_bot = (mi - lo) / max(1, hi - lo)  # 0 near bottom, 1 near top
        dist_top = (hi - mi) / max(1, hi - lo)
        if f == 1:
            # LH thumb (1) usually near top
            return 600.0 * max(0.0, dist_top - 0.25)
        if f == 5:
            # LH pinky (5) usually near bottom
            return 350.0 * max(0.0, dist_bot - 0.30)
    return 0.0


def _awkward_transition_penalty(pf: int, f: int, abs_semi: int, is_left: bool) -> float:
    """
    Penalize uncomfortable/unnatural transitions like RH 5->3 on small moves
    (and mirrored LH 1->3 on small moves).
    """
    if abs_semi >= 5:
        return 0.0

    # RH: 5->3 or 5->2 on stepwise-ish motion is usually avoidable
    if not is_left:
        if pf == 5 and f in (2, 3) and abs_semi <= 2:
            return 900.0
        if pf == 5 and f == 3 and abs_semi <= 4:
            return 450.0
        # also discourage 4->2 on tiny moves (often 4->3->2)
        if pf == 4 and f == 2 and abs_semi <= 2:
            return 250.0
    else:
        # LH mirror: 1->3 on tiny moves is usually awkward
        if pf == 1 and f == 3 and abs_semi <= 2:
            return 900.0
        if pf == 1 and f in (3, 4) and abs_semi <= 4:
            return 450.0
        if pf == 2 and f == 4 and abs_semi <= 2:
            return 250.0

    return 0.0


def decode_position_dp(midi_seq, measure_seq, is_left: bool):
    n = len(midi_seq)
    if n == 0:
        return []
    if n == 1:
        return [3]

    LOOKAHEAD = 12
    LOOKBACK = 4
    local_lo = []
    local_hi = []
    for i in range(n):
        a = max(0, i - LOOKBACK)
        b = min(n, i + LOOKAHEAD)
        window = midi_seq[a:b]
        local_lo.append(min(window))
        local_hi.append(max(window))

    def bases_for(m):
        return list(set(m - FINGER_OFFSET[f] for f in FINGERS))

    dp = [{} for _ in range(n)]

    m0 = midi_seq[0]
    for b in bases_for(m0):
        best = None
        for f in FINGERS:
            err = abs(m0 - (b + FINGER_OFFSET[f]))
            cost = 6.0 * err + (0.0 if f in (2, 3, 4) else 1.5)
            pos_finger = get_position_finger(m0, local_lo[0], local_hi[0], is_left)
            cost += 15.0 * abs(f - pos_finger)
            cost += _edge_penalty(m0, local_lo[0], local_hi[0], f, is_left)
            if best is None or cost < best[0]:
                best = (cost, None, f)
        dp[0][b] = best

    for i in range(1, n):
        mi = midi_seq[i]
        pm = midi_seq[i - 1]
        delta = mi - pm
        abs_semi = abs(delta)
        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)

        steps_ahead = 0
        for j in range(i, min(i + 6, n - 1)):
            d = midi_seq[j + 1] - midi_seq[j]
            if (direction > 0 and d in [1, 2]) or (direction < 0 and d in [-1, -2]):
                steps_ahead += 1
            else:
                break

        row = {}
        for b in bases_for(mi):
            best_cost = None
            best_prev_b = None
            best_f = None

            for pb, (pcost, _, pf) in dp[i - 1].items():
                for f in FINGERS:
                    if abs_semi == 0 and f != pf:
                        continue

                    cost = pcost
                    cost += 2.5 * abs(b - pb)
                    err = abs(mi - (b + FINGER_OFFSET[f]))
                    cost += 7.0 * err

                    # Position expectation
                    if abs_semi >= 5:
                        pos_finger = get_position_finger(mi, local_lo[i], local_hi[i], is_left)
                        cost += 50.0 * abs(f - pos_finger)
                    elif abs_semi <= 2:
                        if not is_left:
                            if direction > 0 and pf == 3 and f == 1 and steps_ahead >= 3:
                                cost -= 40.0
                            elif direction > 0 and pf in (3, 4) and f == 1:
                                pass
                            elif direction < 0 and pf == 1 and f in (3, 4):
                                pass
                            elif f != pf + direction:
                                cost += 50.0
                        else:
                            if direction > 0 and pf == 1 and f == 3 and steps_ahead >= 3:
                                cost -= 40.0
                            elif direction > 0 and pf == 1 and f == 3:
                                pass
                            elif direction < 0 and pf == 3 and f == 1:
                                pass
                            elif f != pf - direction:
                                cost += 50.0

                        pos_finger = get_position_finger(mi, local_lo[i], local_hi[i], is_left)
                        cost += 8.0 * abs(f - pos_finger)
                    else:
                        pos_finger = get_position_finger(mi, local_lo[i], local_hi[i], is_left)
                        cost += 25.0 * abs(f - pos_finger)
                        expected = (pf + direction) if not is_left else (pf - direction)
                        if f != expected:
                            cost += 15.0

                    # NEW: avoid overusing outer fingers away from local edges
                    cost += _edge_penalty(mi, local_lo[i], local_hi[i], f, is_left)

                    # NEW: punish awkward transitions like RH 5->3 (or LH 1->3) on small moves
                    cost += _awkward_transition_penalty(pf, f, abs_semi, is_left)

                    if abs(f - pf) > 2:
                        cost += 60.0

                    # STRONGER: avoid repeating same finger on different notes
                    if abs_semi > 0 and f == pf:
                        if abs_semi <= 2:
                            cost += 2000.0
                        else:
                            cost += 500.0

                    cost += 0.0 if f in (2, 3, 4) else 1.0

                    if best_cost is None or cost < best_cost:
                        best_cost, best_prev_b, best_f = cost, pb, f

            if best_cost is not None:
                row[b] = (best_cost, best_prev_b, best_f)

        dp[i] = row
        if not dp[i]:
            return [
                get_position_finger(midi_seq[k], local_lo[k], local_hi[k], is_left)
                for k in range(n)
            ]

    end_base = min(dp[-1].items(), key=lambda kv: kv[1][0])[0]
    fingers = []
    b = end_base
    for i in range(n - 1, -1, -1):
        _, prev_b, f = dp[i][b]
        fingers.append(int(f))
        b = prev_b if prev_b is not None else b
    fingers.reverse()
    return fingers


def pedagogy_smooth(fingers, midi_seq, is_left: bool):
    if not fingers:
        return fingers
    out = list(map(int, fingers))

    def clamp(x):
        return max(1, min(5, int(x)))

    for i in range(1, len(out)):
        d = midi_seq[i] - midi_seq[i - 1]
        if d == 0:
            out[i] = out[i - 1]
            continue

        # Force off repeated finger on stepwise motion
        if abs(d) <= 2 and out[i] == out[i - 1]:
            direction = 1 if d > 0 else -1
            pf = out[i - 1]
            out[i] = clamp(pf + direction) if not is_left else clamp(pf - direction)

        # Additional cleanup: avoid RH 5->3 and LH 1->3 when stepwise-ish
        if abs(d) <= 2:
            pf = out[i - 1]
            f = out[i]
            if not is_left and pf == 5 and f == 3:
                out[i] = 4  # usually 5->4 feels better
            if is_left and pf == 1 and f == 3:
                out[i] = 2  # usually 1->2 feels better

    return out


def predict_fingering_for_notes(df: pd.DataFrame, left_hand_groups: set[tuple[str, int]]):
    if df.empty:
        return df

    out = []

    for (part, staff, voice), grp in df.groupby(["part", "staff", "voice"], sort=False):
        grp = grp.sort_values("idx").reset_index(drop=True).copy()

        part_str = str(part)
        staff_int = int(staff)
        is_left = (part_str, staff_int) in left_hand_groups

        grp["hand"] = "LH" if is_left else "RH"
        grp["predicted_fingering"] = 0

        chord_sizes = grp.groupby("chord_id")["midi"].transform("count")
        chord_mask = chord_sizes >= 2

        # ── Chords ─────────────────────────────────────────────────────────────
        if chord_mask.any():
            chord_rows = grp[chord_mask].copy()
            chord_order = (chord_rows.groupby("chord_id")["idx"]
                           .min().sort_values().index.tolist())

            chord_midis_by_id = []
            for cid in chord_order:
                cg = chord_rows[chord_rows["chord_id"] == cid]
                mids = sorted(cg["midi"].astype(int).tolist())
                chord_midis_by_id.append((int(cid), mids))

            ladder_maps = detect_and_build_dyad_ladder_maps(chord_midis_by_id, is_left=is_left)

            prev_chord_map = None
            prev_chord_midis = None

            for cid, mids in chord_midis_by_id:
                if cid in ladder_maps:
                    chord_map = ladder_maps[cid]
                else:
                    if prev_chord_midis == mids and prev_chord_map is not None:
                        chord_map = prev_chord_map
                    else:
                        chord_map = choose_chord_fingering(prev_chord_map, mids, is_left=is_left)

                cg = chord_rows[chord_rows["chord_id"] == cid]
                for row_i in cg.index:
                    m = int(grp.loc[row_i, "midi"])
                    grp.loc[row_i, "predicted_fingering"] = chord_map.get(m, 3)

                prev_chord_map = chord_map
                prev_chord_midis = mids

        # ── Single notes ───────────────────────────────────────────────────────
        single = grp[~chord_mask].copy()
        if not single.empty:
            midi_seq = single["midi"].astype(int).tolist()
            measure_seq = single["measure"].astype(int).tolist()
            fingers = decode_position_dp(midi_seq, measure_seq, is_left=is_left)
            fingers = pedagogy_smooth(fingers, midi_seq, is_left=is_left)
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
    print(f"✅ Extracted {len(notes)} notes")

    print("\n📊 Parts and staves found:")
    print(notes[['part', 'staff']].drop_duplicates().to_string(index=False))

    left_hand_groups = compute_left_hand_groups(notes)
    print(f"\n👈 LEFT HAND groups detected: {sorted(left_hand_groups)}")

    preds = predict_fingering_for_notes(notes, left_hand_groups=left_hand_groups)

    print("\n🎯 Sample predictions:")
    for hand in ['RH', 'LH']:
        hand_notes = preds[preds['hand'] == hand]
        if not hand_notes.empty:
            print(f"\n{hand}:")
            print(hand_notes[['measure', 'pitch', 'midi', 'staff', 'hand', 'predicted_fingering']].head(10).to_string(index=False))

    out_path = Path("data/predictions") / f"{path.stem}_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"\n💾 Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
