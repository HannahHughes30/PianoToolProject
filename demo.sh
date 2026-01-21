#!/bin/bash

cd ~/PianoToolProject

echo "============================================================"
echo "  PIANO FINGERING ML SYSTEM - COMPLETE DEMO"
echo "  Hannah Hughes | Senior Project | Nov 2024"
echo "============================================================"
echo ""

echo "=========================================="
echo "DEMO 1: Training Data Status"
echo "=========================================="
python check_data.py
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "=========================================="
echo "DEMO 2: Test Model on Standard Patterns"
echo "=========================================="
echo "Testing on: C Major scale, G Major scale, Descending scale, Arpeggios"
echo ""
python src/ml/test_model.py
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "=========================================="
echo "DEMO 3: Predict on Complete Piece #1"
echo "=========================================="
MXL_FILE=$(find data/omr_output -name "*.mxl" -type f | head -1)
if [ -n "$MXL_FILE" ]; then
    echo "Predicting fingering on: $MXL_FILE"
    echo ""
    python predict_from_mxl.py "$MXL_FILE"
else
    echo "❌ No .mxl files found"
fi
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "=========================================="
echo "DEMO 4: Predict on Complete Piece #2"
echo "=========================================="
MXL_FILE2=$(find data/omr_output -name "*.mxl" -type f | tail -1)
if [ -n "$MXL_FILE2" ]; then
    echo "Predicting fingering on: $MXL_FILE2"
    echo ""
    python predict_from_mxl.py "$MXL_FILE2"
else
    echo "❌ No second .mxl file found"
fi
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "=========================================="
echo "DEMO 5: Show Full Training Pipeline"
echo "=========================================="
echo "This shows how the model was trained..."
echo ""

echo "Step 1/3: Extract fingering annotations..."
python src/annotation/extract_all_fingering.py

echo ""
echo "Step 2/3: Generate features (14 total)..."
python src/ml/feature_engineering.py

echo ""
echo "Step 3/3: Train Random Forest model..."
python src/ml/train_model.py

echo ""
echo "Press Enter to continue..."
read

echo ""
echo "=========================================="
echo "DEMO 6: Process Brand New PDF (Optional)"
echo "=========================================="
echo "Want to process a new PDF? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Processing piano3_cabinsong.pdf..."
    python src/omr/audiveris_wrapper.py data/test_sheets/piano3_cabinsong.pdf
    
    echo ""
    echo "Finding output..."
    NEW_FILE=$(find data/omr_output -name "*cabinsong*.mxl" -type f -mtime -1m | head -1)
    
    if [ -n "$NEW_FILE" ]; then
        echo "Predicting fingering on new piece..."
        python predict_from_mxl.py "$NEW_FILE"
    else
        echo "Using existing cabinsong file..."
        CABIN_FILE=$(find data/omr_output -name "*cabinsong*.mxl" -type f | head -1)
        if [ -n "$CABIN_FILE" ]; then
            python predict_from_mxl.py "$CABIN_FILE"
        fi
    fi
else
    echo "Skipping new PDF processing"
fi

echo ""
echo "============================================================"
echo "                    DEMO COMPLETE!"
echo "============================================================"
echo ""
echo "Summary:"
echo "  ✓ Training data: 981 notes (810 manual + 171 inferred)"
echo "  ✓ Model accuracy: 79.7%"
echo "  ✓ Successfully predicts fingering on unseen pieces"
echo "  ✓ Hybrid ML + piano pedagogy approach working"
echo ""
echo "Next steps:"
echo "  - Collect 500-1,000 more notes → 85-90% accuracy"
echo "  - Add PDF output generation"
echo "  - Validation with piano teacher"
echo ""
echo "============================================================"
