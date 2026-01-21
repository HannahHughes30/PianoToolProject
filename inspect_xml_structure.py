import zipfile
from pathlib import Path

# Open one of your annotated files
mxl_file = Path('data/omr_output/annotated_piano17_cmajor_page1.mxl')

print(f"Inspecting: {mxl_file.name}\n")

with zipfile.ZipFile(mxl_file, 'r') as zip_ref:
    # List all files in the zip
    print("Files in .mxl archive:")
    for fname in zip_ref.namelist():
        print(f"  {fname}")
    
    # Read the actual score.xml file
    if 'score.xml' in zip_ref.namelist():
        content = zip_ref.read('score.xml').decode('utf-8')
        
        # Save it
        Path('temp_score.xml').write_text(content)
        print(f"\n✓ Extracted score.xml ({len(content)} bytes)")
        
        # Check for fingering
        if '<fingering>' in content.lower():
            print("\n✓ FOUND fingering tags!")
            
            # Count them
            count = content.lower().count('<fingering>')
            print(f"   Total fingering tags: {count}")
            
            # Show first occurrence
            idx = content.lower().find('<fingering>')
            snippet = content[max(0, idx-300):idx+200]
            print("\nContext around first <fingering>:")
            print("="*70)
            print(snippet)
            print("="*70)
        else:
            print("\n✗ No <fingering> tags found in score.xml")
            
            # Search for numbers that might be fingerings
            print("\nSearching for potential fingering numbers...")
            lines = content.split('\n')
            for i, line in enumerate(lines[:100]):
                if any(tag in line for tag in ['<technical>', '<articulations>', '<notations>']):
                    print(f"  Line {i}: {line.strip()[:100]}")

print(f"\nFull score saved to: temp_score.xml")
print("View with: cat temp_score.xml | less")
