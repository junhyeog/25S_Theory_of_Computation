#!/bin/bash

# Base directories
SCRIPT_DIR=$(pwd)
SUBMISSION_DIR="$SCRIPT_DIR/submissions"
UNZIP_DIR="$SCRIPT_DIR/unzip"
INPUT_DIR="$SCRIPT_DIR/input"
OUTPUT_DIR="$SCRIPT_DIR/output"
ANSWER_DIR="$SCRIPT_DIR/answer"

# Create unzip folder if it doesn't exist
mkdir -p "$UNZIP_DIR"

# Process all zip files in the submissions folder
for zip_file in "$SUBMISSION_DIR"/*.zip; do
    if [ -e "$zip_file" ]; then
        # Get the file name without extension
        zip_base=$(basename "$zip_file" .zip)
        target_dir="$UNZIP_DIR/$zip_base"
        mkdir -p "$target_dir"

        # Extract contents to target_dir, ignoring internal top-level directory
        unzip -q -o "$zip_file" -d "$target_dir"
    fi
done


# Iterate through each extracted folder
for student_dir in "$UNZIP_DIR"/*; do
    if [ -d "$student_dir" ]; then
        echo "▶ Processing: $(basename "$student_dir")"

        # Run compile.sh
        if [ -f "$student_dir/compile.sh" ]; then
            echo "  ├─ Running compile.sh..."
            (cd "$student_dir" && chmod +x compile.sh && ./compile.sh)
        else
            echo "  ├─ ⚠ compile.sh not found"
        fi

        # Run run_algorithm.sh
        if [ -f "$student_dir/run_algorithm.sh" ]; then
            echo "  ├─ Running run_algorithm.sh..."
            (cd "$student_dir" && chmod +x run_algorithm.sh && timeout 10s ./run_algorithm.sh "$INPUT_DIR/input.txt" "$OUTPUT_DIR/output.txt")
            # Check if it timed out
            if [ $? -eq 124 ]; then
                echo "  └─ ⏰ Timeout: run_algorithm.sh took longer than 10 seconds. Skipping."
                echo ""
                continue
            fi
        else
            echo "  ├─ ⚠ run_algorithm.sh not found"
            echo ""
            continue
        fi

        # Run run_checker.sh
        if [ -f "$student_dir/run_checker.sh" ]; then
            echo "  ├─ Running run_checker.sh..."
            (cd "$student_dir" && chmod +x run_checker.sh && timeout 10s ./run_checker.sh "$INPUT_DIR/input.txt" "$INPUT_DIR/output_correct.txt" "$OUTPUT_DIR/correct.txt" )
            # Check if it timed out
            if [ $? -eq 124 ]; then
                echo "  └─ ⏰ Timeout: run_checker.sh took longer than 10 seconds. Skipping."
                echo ""
                continue
            fi
            (cd "$student_dir" && chmod +x run_checker.sh && timeout 10s ./run_checker.sh "$INPUT_DIR/input.txt" "$INPUT_DIR/output_incorrect.txt" "$OUTPUT_DIR/incorrect.txt" )
            # Check if it timed out
            if [ $? -eq 124 ]; then
                echo "  └─ ⏰ Timeout: run_checker.sh took longer than 10 seconds. Skipping."
                echo ""
                continue
            fi
        else
            echo "  ├─ ⚠ run_checker.sh not found"
            echo ""
            continue
        fi

        # Check output directory and compare with answers
        if [ -d "$OUTPUT_DIR" ]; then
            echo "  └─ Comparing output files..."

            for file in output.txt correct.txt incorrect.txt; do
                output_file="$OUTPUT_DIR/$file"
                answer_file="$ANSWER_DIR/$file"

                if [ -f "$output_file" ]; then
                    if diff -q -b "$output_file" "$answer_file" > /dev/null; then
                        echo "      ✓ $file matches the answer"
                    else
                        echo "      ✗ $file does NOT match the answer"
                    fi
                else
                    echo "      ⚠ $file missing in output directory"
                fi
            done
        else
            echo "  └─ ⚠ output folder not found after run.sh"
        fi

        echo ""
    fi
done
