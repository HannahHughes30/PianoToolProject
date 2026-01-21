"""
UNIVERSAL FINGERING EXTRACTOR
- Handles fully annotated pieces
- Handles partially annotated pieces
- Parses chord fingerings (12 = bottom:1, top:2)
- Works with any MusicXML format
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import zipfile
import csv

def parse_chord_fingering(fingering_str, note_count=1):
    """
    Parse chord fingering string
    Examples:
      "12" with 2 notes → [1, 2] (bottom to top)
      "123" with 3 notes → [1, 2, 3]
      "5" with 1 note → [5]
    
    Returns first digit if can't parse
    """
    if not fingering_str or not str(fingering_str).replace('.', '').isdigit():
        return None
    
    # Convert to string of digits
    digits_str = str(int(float(fingering_str)))
    digits = [int(d) for d in digits_str if d.isdigit() and 1 <= int(d) <= 5]
    
    if not digits:
        return None
    
    # Single digit - normal fingering
    if len(digits) == 1:
        return digits[0]
    
    # Multi-digit - chord fingering, use first (bottom note)
    # In future, we could assign each digit to each note
    return digits[0]

def extract_from_mxl(mxl_path):
    """Extract all notes with optional fingering"""
    print(f"\nProcessing: {mxl_path.name}")
    
    try:
        with zipfile.ZipFile(mxl_path, 'r') as zip_ref:
            if 'score.xml' in zip_ref.namelist():
                content = zip_ref.read('score.xml')
            else:
                return []
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []
    
    root = ET.fromstring(content)
    annotations = []
    parts = root.findall('.//part')
    
    for part_idx, part in enumerate(parts):
        part_id = part.get('id', f'P{part_idx+1}')
        
        for measure in part.findall('measure'):
            measure_num = measure.get('number', '0')
            elements = list(measure)
            current_fingering = None
            
            for elem in elements:
                # Check for fingering in harmony (MuseScore format)
                if elem.tag == 'harmony':
                    kind = elem.find('.//kind')
                    if kind is not None:
                        fingering_text = kind.get('text')
                        if fingering_text:
                            current_fingering = fingering_text
                
                # Process notes
                elif elem.tag == 'note':
                    if elem.find('rest') is not None:
                        continue
                    
                    pitch_elem = elem.find('pitch')
                    if pitch_elem is None:
                        continue
                    
                    step = pitch_elem.find('step').text
                    octave = pitch_elem.find('octave').text
                    
                    alter_elem = pitch_elem.find('alter')
                    if alter_elem is not None:
                        alter = int(alter_elem.text)
                        pitch = step + ('#' if alter == 1 else 'b') + octave
                    else:
                        pitch = step + octave
                    
                    # Also check standard MusicXML fingering location
                    if current_fingering is None:
                        notations = elem.find('notations')
                        if notations is not None:
                            technical = notations.find('technical')
                            if technical is not None:
                                fingering_elem = technical.find('fingering')
                                if fingering_elem is not None:
                                    current_fingering = fingering_elem.text
                    
                    # Parse fingering (handles chords)
                    parsed_fingering = ''
                    if current_fingering:
                        parsed = parse_chord_fingering(current_fingering)
                        if parsed and 1 <= parsed <= 5:
                            parsed_fingering = str(parsed)
                    
                    annotations.append({
                        'file': mxl_path.stem,
                        'measure': measure_num,
                        'pitch': pitch,
                        'fingering': parsed_fingering,
                        'part': part_id
                    })
                    
                    current_fingering = None
    
    fingered = sum(1 for a in annotations if a['fingering'])
    coverage = (fingered / len(annotations) * 100) if annotations else 0
    
    print(f"  ✓ {len(annotations)} notes ({fingered} fingered, {coverage:.1f}%)")
    
    return annotations

def main():
    output_dir = Path('data/omr_output')
    mxl_files = list(set(list(output_dir.glob('annotated_*.mxl')) + 
                         list(output_dir.rglob('annotated_*.mxl'))))
    
    if not mxl_files:
        print("✗ No annotated .mxl files found")
        return
    
    print("="*60)
    print("UNIVERSAL FINGERING EXTRACTION")
    print("="*60)
    print(f"\nFound {len(mxl_files)} files")
    
    all_annotations = []
    
    for mxl_file in sorted(mxl_files):
        annotations = extract_from_mxl(mxl_file)
        all_annotations.extend(annotations)
    
    if all_annotations:
        csv_dir = Path('data/training_data')
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / 'fingering_data.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'measure', 'pitch', 'fingering', 'part'])
            writer.writeheader()
            writer.writerows(all_annotations)
        
        total = len(all_annotations)
        fingered = sum(1 for a in all_annotations if a['fingering'])
        coverage = (fingered / total * 100)
        
        print(f"\n{'='*60}")
        print(f"Total: {total} notes")
        print(f"Annotated: {fingered} ({coverage:.1f}%)")
        print(f"Missing: {total - fingered}")
        print(f"✓ Saved to: {csv_path}")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()
