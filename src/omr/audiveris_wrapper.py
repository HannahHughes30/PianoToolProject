import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import sys
from pdf2image import convert_from_path
import cv2
import numpy as np
import shutil
import zipfile

class AudiverisOMR:
    def __init__(self, audiveris_path='/Applications/Audiveris.app'):
        self.audiveris_path = audiveris_path
        self.output_dir = Path('data/omr_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clear_all_cache(self):
        """Clear ALL Audiveris cache and output files"""
        print(f"Clearing all cached files...")
        if self.output_dir.exists():
            for item in self.output_dir.glob('*'):
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        print(f"✓ Cache cleared")
    
    def pdf_to_pages(self, pdf_path):
        """Convert PDF to individual page images"""
        print(f"Converting PDF to individual pages...")
        try:
            images = convert_from_path(pdf_path, dpi=300)
            print(f"PDF has {len(images)} pages")
            return images
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def has_music_content(self, image):
        """Check if image page has actual music notation (not blank/title page)"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:
                    horizontal_lines += 1
            
            return horizontal_lines > 5
        
        return False
    
    def process_image(self, image_path, output_name=None):
        """Process image with Audiveris and return the MusicXML file path"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if output_name is None:
            output_name = image_path.stem
        
        # Audiveris creates a DIRECTORY with this name
        output_dir = self.output_dir / output_name
        
        # Remove existing output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        cmd = [
            f"{self.audiveris_path}/Contents/MacOS/Audiveris",
            "-batch",
            "-export",
            "-output", str(output_dir),
            str(image_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Look for .mxl file inside the output directory
                mxl_files = list(output_dir.glob('*.mxl'))
                if mxl_files:
                    return mxl_files[0]
                
                # Look for .xml files if no .mxl found
                xml_files = list(output_dir.glob('*.xml'))
                if xml_files:
                    return xml_files[0]
                
                print(f"  ✗ No MusicXML files found in {output_dir}")
                return None
            else:
                print(f"  ✗ Audiveris failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Processing timed out")
            return None
        except FileNotFoundError:
            print(f"  ✗ Audiveris not found at {self.audiveris_path}")
            return None
    
    def parse_musicxml(self, musicxml_path):
        """Parse MusicXML with proper hand separation and chord grouping"""
        try:
            # Handle .mxl compressed files
            if str(musicxml_path).endswith('.mxl'):
                with zipfile.ZipFile(musicxml_path, 'r') as zip_ref:
                    temp_dir = self.output_dir / 'temp_extract'
                    temp_dir.mkdir(exist_ok=True)
                    zip_ref.extractall(temp_dir)
                    
                    xml_files = list(temp_dir.glob('*.xml'))
                    if xml_files:
                        musicxml_path = xml_files[0]
                    else:
                        return []
            
            tree = ET.parse(musicxml_path)
            root = tree.getroot()
            
            notes_data = []
            
            parts = root.findall('.//part')
            
            for part_idx, part in enumerate(parts):
                part_id = part.get('id', f'P{part_idx+1}')
                
                # Detect staff configuration
                staves = {}
                for measure in part.findall('measure'):
                    for attributes in measure.findall('attributes'):
                        for clef in attributes.findall('clef'):
                            staff_num = clef.get('number', '1')
                            clef_sign = clef.find('sign')
                            if clef_sign is not None:
                                staves[staff_num] = clef_sign.text
                
                # Determine hand assignment strategy
                if len(parts) == 2:
                    default_hand = 'RIGHT' if part_idx == 0 else 'LEFT'
                elif len(staves) >= 2:
                    default_hand = 'USE_STAFF'
                elif len(parts) == 1:
                    default_hand = 'USE_PITCH'
                else:
                    default_hand = f'PART{part_idx+1}'
                
                measure_number = 0
                
                for measure in part.findall('measure'):
                    measure_number = int(measure.get('number', measure_number + 1))
                    
                    for note_elem in measure.findall('note'):
                        # Skip rests
                        if note_elem.find('rest') is not None:
                            continue
                        
                        staff_elem = note_elem.find('staff')
                        staff_num = staff_elem.text if staff_elem is not None else '1'
                        
                        pitch_elem = note_elem.find('pitch')
                        if pitch_elem is not None:
                            step = pitch_elem.find('step').text
                            octave = pitch_elem.find('octave').text
                            alter_elem = pitch_elem.find('alter')
                            
                            pitch_name = step + octave
                            if alter_elem is not None:
                                alter = int(alter_elem.text)
                                if alter == 1:
                                    pitch_name = step + '#' + octave
                                elif alter == -1:
                                    pitch_name = step + 'b' + octave
                            
                            duration_elem = note_elem.find('duration')
                            duration = int(duration_elem.text) if duration_elem is not None else 0
                            
                            is_chord = note_elem.find('chord') is not None
                            
                            # Assign hand based on strategy
                            if default_hand == 'USE_STAFF':
                                note_hand = 'RIGHT' if staff_num == '1' else 'LEFT'
                            elif default_hand == 'USE_PITCH':
                                octave_num = int(octave)
                                note_hand = 'RIGHT' if octave_num >= 4 else 'LEFT'
                            else:
                                note_hand = default_hand
                            
                            note_data = {
                                'measure': measure_number,
                                'pitch': pitch_name,
                                'duration': duration,
                                'hand': note_hand,
                                'is_chord': is_chord,
                                'part_id': part_id
                            }
                            
                            notes_data.append(note_data)
            
            return notes_data
            
        except Exception as e:
            print(f"Error parsing MusicXML: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def format_notes_display(self, notes):
        """Format notes for clear display with hand and chord information"""
        if not notes:
            return
        
        print(f"\n{'='*80}")
        print(f"EXTRACTED NOTES ({len(notes)} total)")
        print(f"{'='*80}\n")
        
        current_measure = None
        chord_group = []
        note_count = 0
        
        for i, note in enumerate(notes):
            note_count += 1
            
            if note['measure'] != current_measure:
                if chord_group:
                    self._print_chord(chord_group)
                    chord_group = []
                
                current_measure = note['measure']
                print(f"\n--- Measure {current_measure} ---")
            
            if note['is_chord']:
                if not chord_group:
                    if i > 0:
                        chord_group.append(notes[i-1])
                chord_group.append(note)
            else:
                if chord_group:
                    self._print_chord(chord_group)
                    chord_group = []
                
                if not (i+1 < len(notes) and notes[i+1]['is_chord']):
                    hand_str = f"[{note['hand']}]".ljust(7)
                    print(f"  {note_count:3d}. {hand_str} {note['pitch']:4s} (duration: {note['duration']})")
            
            if note_count >= 200:
                remaining = len(notes) - 200
                if remaining > 0:
                    print(f"\n  ... and {remaining} more notes")
                break
        
        if chord_group:
            self._print_chord(chord_group)
    
    def _print_chord(self, chord_notes):
        """Print a chord group clearly"""
        if not chord_notes:
            return
        
        hand = chord_notes[0]['hand']
        pitches = [n['pitch'] for n in chord_notes]
        duration = chord_notes[0]['duration']
        
        print(f"  CHORD [{hand}]: {' + '.join(pitches)} (duration: {duration})")
    
    def process_and_extract(self, image_path, clear_cache=True):
        """Process PDF - PROCESSES ALL PAGES and combines notes"""
        image_path = Path(image_path)
        
        if clear_cache:
            self.clear_all_cache()
        
        if str(image_path).lower().endswith('.pdf'):
            print(f"\nProcessing PDF: {image_path.name}")
            pages = self.pdf_to_pages(image_path)
            
            if not pages:
                print("Could not convert PDF to pages")
                return None
            
            all_notes = []
            
            for page_num, page_image in enumerate(pages, 1):
                print(f"\nAnalyzing page {page_num}/{len(pages)}...")
                
                if not self.has_music_content(page_image):
                    print(f"  ✗ Page {page_num} appears to be blank or title page, skipping...")
                    continue
                
                print(f"  ✓ Page {page_num} has music notation, processing...")
                
                # Save temporary image
                temp_image = self.output_dir / f"temp_{image_path.stem}_page{page_num}.png"
                page_array = np.array(page_image)
                page_bgr = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(temp_image), page_bgr)
                
                # Process with unique name for each page
                output_name = f"{image_path.stem}_page{page_num}"
                musicxml_path = self.process_image(temp_image, output_name)
                
                # Clean up temp image
                if temp_image.exists():
                    temp_image.unlink()
                
                if musicxml_path and musicxml_path.exists():
                    print(f"  ✓ Found MusicXML: {musicxml_path.name}")
                    notes = self.parse_musicxml(musicxml_path)
                    if notes and len(notes) > 0:
                        print(f"  ✓ Extracted {len(notes)} notes from page {page_num}")
                        all_notes.extend(notes)
                    else:
                        print(f"  ✗ Page {page_num} processed but no notes found")
                else:
                    print(f"  ✗ Page {page_num} processing failed - no MusicXML output")
            
            if all_notes:
                print(f"\n✓ Total notes from all pages: {len(all_notes)}")
                return all_notes
            else:
                print("\n✗ No notes found on any page")
                return []
        
        else:
            print(f"\nProcessing image: {image_path.name}")
            musicxml_path = self.process_image(image_path)
            
            if musicxml_path is None:
                return None
            
            notes = self.parse_musicxml(musicxml_path)
            return notes


if __name__ == "__main__":
    omr = AudiverisOMR()
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = 'data/test_sheets/Twinkle_Twinkle_Little_Star.pdf'
    
    if not Path(test_image).exists():
        print(f"File not found: {test_image}")
        print("Usage: python src/omr/audiveris_wrapper.py <pdf_file>")
        sys.exit(1)
    
    print("="*60)
    print("Testing Audiveris OMR System")
    print("="*60)
    
    notes = omr.process_and_extract(test_image)
    
    if notes and len(notes) > 0:
        omr.format_notes_display(notes)
    else:
        print("\n✗ No notes extracted from any page.")
