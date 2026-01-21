#!/bin/bash

cd ~/PianoToolProject

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   PIANO FINGERING AI SYSTEM - COMPLETE DEMONSTRATION       ║"
echo "║   Student: Hannah Hughes                                   ║"
echo "║   Phase 1: OMR Pipeline - COMPLETE                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# PART 1: PROJECT OVERVIEW
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 1: PROJECT OVERVIEW"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Project Structure:"
ls -la
echo ""
echo "Source Code Modules:"
ls -la src/
echo ""
echo "OMR Modules:"
ls -la src/omr/
echo ""
echo "Test Data:"
echo "  Piano pieces collected: $(ls data/test_sheets/*.pdf 2>/dev/null | wc -l)"
echo "  MusicXML files generated: $(ls data/omr_output/*.mxl 2>/dev/null | wc -l)"
echo ""
echo ""

# ============================================
# PART 2: IMAGE PROCESSING MODULE
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 2: IMAGE PROCESSING MODULE (src/image_processing.py)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Code:"
cat src/image_processing.py
echo ""
echo "Key Functions:"
grep "^def " src/image_processing.py
echo ""
echo ""

# ============================================
# PART 3: OMR WRAPPER MODULE
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 3: OMR WRAPPER MODULE (src/omr/audiveris_wrapper.py)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Total Lines: $(wc -l < src/omr/audiveris_wrapper.py)"
echo ""
echo "Class Structure:"
grep "^class\|^    def " src/omr/audiveris_wrapper.py
echo ""
echo "Full Code:"
cat src/omr/audiveris_wrapper.py
echo ""
echo ""

# ============================================
# PART 4: DEMO - SIMPLE C MAJOR SCALE
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 4: PROCESSING SIMPLE PIECE (C Major Scale)"
echo "═══════════════════════════════════════════════════════════"
echo ""
python src/omr/audiveris_wrapper.py data/test_sheets/piano17_cmajor.pdf
echo ""
echo ""

# ============================================
# PART 5: DEMO - PIECE WITH CHORDS
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 5: PROCESSING COMPLEX PIECE (With Chords & Both Hands)"
echo "═══════════════════════════════════════════════════════════"
echo ""
python src/omr/audiveris_wrapper.py data/test_sheets/Piano7_candace.pdf
echo ""
echo ""

# ============================================
# PART 6: DEMO - MULTI-PAGE WITH BLANK DETECTION
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 6: MULTI-PAGE PDF (Blank Page Detection)"
echo "═══════════════════════════════════════════════════════════"
echo ""
python src/omr/audiveris_wrapper.py data/test_sheets/piano4_fourpieces.pdf
echo ""
echo ""

# ============================================
# PART 7: COMPREHENSIVE TEST RESULTS
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 7: COMPREHENSIVE TEST RESULTS (All 19 Pieces)"
echo "═══════════════════════════════════════════════════════════"
echo ""
cat test_results_log.txt
echo ""
echo ""

# ============================================
# PART 8: STATISTICS & SUMMARY
# ============================================
echo "═══════════════════════════════════════════════════════════"
echo "PART 8: PROJECT STATISTICS & SUMMARY"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✓ Test Files Collected: 19 beginner piano pieces"
echo "✓ Successfully Processed: 19 (100% success rate)"
echo "✓ Total Notes Extracted: 3,288 notes"
echo "✓ Note Range: 72-271 notes per piece"
echo "✓ Average: 173 notes per piece"
echo ""
echo "Shortest Piece: piano3_cabinsong.pdf (72 notes)"
echo "Longest Piece: Piano8_floatingclouds.pdf (271 notes)"
echo ""
echo "Technologies Used:"
echo "  • Python 3.11+"
echo "  • OpenCV (image processing)"
echo "  • Audiveris (OMR engine)"
echo "  • pdf2image (PDF conversion)"
echo "  • xml.etree (MusicXML parsing)"
echo "  • NumPy (numerical operations)"
echo ""
echo "Phase 1: COMPLETE ✓"
echo "Next Phase: Annotate 10-12 pieces with fingering numbers"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "DEMO COMPLETE"
echo "═══════════════════════════════════════════════════════════"

