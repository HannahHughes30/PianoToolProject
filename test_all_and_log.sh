#!/bin/bash

echo "PIANO OMR TEST RESULTS - $(date)" > test_results_log.txt
echo "=======================================" >> test_results_log.txt
echo "" >> test_results_log.txt

for file in data/test_sheets/piano*.pdf data/test_sheets/Piano*.pdf; do
    if [ -f "$file" ]; then
        echo "Testing: $(basename $file)" | tee -a test_results_log.txt
        python src/omr/audiveris_wrapper.py "$file" 2>&1 | grep -E "(Successfully extracted|No notes)" | tee -a test_results_log.txt
        echo "---" | tee -a test_results_log.txt
    fi
done

echo "" >> test_results_log.txt
echo "Test complete!" | tee -a test_results_log.txt
