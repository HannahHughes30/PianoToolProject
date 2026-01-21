import sys
from pathlib import Path
sys.path.append('src')

from image_processing import SheetMusicProcessor

test_path = 'data/test_sheets/Twinkle_Twinkle_Little_Star.pdf'
print(f"Input path: {test_path}")
print(f"Path.stem: {Path(test_path).stem}")
print(f"Path.name: {Path(test_path).name}")

processor = SheetMusicProcessor()
result = processor.preprocess_pipeline(test_path)
print(f"\nFiles should be saved as: Twinkle_Twinkle_Little_Star_*.png")
