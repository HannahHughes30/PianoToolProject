#!/usr/bin/env python3
"""Process 3 pieces for manual annotation"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from omr.audiveris_wrapper import AudiverisOMR

def main():
    omr = AudiverisOMR()
    
    print("="*60)
    print("Processing 3 Pieces for Manual Annotation")
    print("="*60)
    
    # Clear cache ONCE at the start
    omr.clear_all_cache()
    
    pieces = [
        'data/test_sheets/Piano7_candace.pdf',
        'data/test_sheets/piano5_cornflower.pdf', 
        'data/test_sheets/piano17_cmajor.pdf'
    ]
    
    for i, piece_path in enumerate(pieces, 1):
        piece_path = Path(piece_path)
        if not piece_path.exists():
            print(f"\n✗ File not found: {piece_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"PIECE {i}/{len(pieces)}: {piece_path.name}")
        print(f"{'='*60}")
        
        # Process WITHOUT clearing cache (clear_cache=False)
        notes = omr.process_and_extract(piece_path, clear_cache=False)
        
        if notes:
            print(f"✓ Successfully extracted {len(notes)} notes from {piece_path.name}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Files Ready for MuseScore Annotation")
    print("="*60)
    
    mxl_files = list(Path('data/omr_output').rglob('temp_*.mxl'))
    for mxl_file in sorted(mxl_files):
        print(f"✓ {mxl_file}")
    
    print(f"\n✓ Total .mxl files created: {len(mxl_files)}")
    print(f"\nTo annotate in MuseScore:")
    print(f"1. Open MuseScore")
    print(f"2. File → Open")
    print(f"3. Navigate to: ~/PianoToolProject/data/omr_output/")
    print(f"4. Open each temp_*.mxl file and add fingering numbers")

if __name__ == '__main__':
    main()
