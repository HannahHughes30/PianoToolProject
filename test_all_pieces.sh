#!/bin/bash
echo "Testing all piano pieces..."
echo "=========================="

for file in data/test_sheets/piano*.pdf; do
    echo ""
    echo "Processing: $(basename $file)"
    python src/omr/audiveris_wrapper.py "$file" 2>&1 | grep -E "(Success|Error|Extracted|Failed)"
    echo "---"
done

echo ""
echo "Testing complete!"
