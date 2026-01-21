import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Pick one annotated file
mxl_file = Path('data/omr_output/annotated_piano17_cmajor_page1.mxl')

print(f"Debugging: {mxl_file.name}\n")

with zipfile.ZipFile(mxl_file, 'r') as zip_ref:
    xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml')]
    xml_content = zip_ref.read(xml_files[0])
    
    # Save to temp file so we can inspect
    temp_file = Path('temp_debug.xml')
    temp_file.write_bytes(xml_content)
    
    root = ET.fromstring(xml_content)
    
    # Check for fingering tags
    fingerings = root.findall('.//fingering')
    print(f"Total <fingering> tags found: {len(fingerings)}")
    
    if fingerings:
        print("\nFirst few fingerings found:")
        for i, f in enumerate(fingerings[:5]):
            print(f"  Fingering {i+1}: {f.text}")
    
    # Let's look at the structure of a note with fingering
    print("\n" + "="*60)
    print("SAMPLE NOTE STRUCTURE:")
    print("="*60)
    
    # Find first note element
    for note in root.findall('.//note')[:3]:
        print("\n<note>")
        pitch = note.find('pitch')
        if pitch is not None:
            step = pitch.find('step')
            octave = pitch.find('octave')
            if step is not None and octave is not None:
                print(f"  Pitch: {step.text}{octave.text}")
        
        # Check for fingering in different locations
        fingering_direct = note.find('fingering')
        fingering_notations = note.find('.//notations/technical/fingering')
        fingering_any = note.find('.//fingering')
        
        if fingering_direct is not None:
            print(f"  Fingering (direct): {fingering_direct.text}")
        elif fingering_notations is not None:
            print(f"  Fingering (in notations): {fingering_notations.text}")
        elif fingering_any is not None:
            print(f"  Fingering (found somewhere): {fingering_any.text}")
        else:
            print(f"  Fingering: NONE")
        
        print("</note>")
    
    print("\n" + "="*60)
    print(f"Saved XML to: {temp_file}")
    print("You can open this file to see the full structure")

