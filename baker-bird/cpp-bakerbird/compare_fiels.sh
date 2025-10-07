#!/bin/bash
# filepath: compare_files.sh

# Check if a value is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <value>"
    echo "Example: $0 1 (to compare output1.txt and example1.txt)"
    exit 1
fi

val=$1
output_file="output${val}.txt"
example_file="example${val}.txt"

# Check if files exist
if [ ! -f "$output_file" ]; then
    echo "Error: $output_file does not exist"
    exit 1
fi

if [ ! -f "$example_file" ]; then
    echo "Error: $example_file does not exist"
    exit 1
fi

echo -e "\n========== $example_file =========="
cat "$example_file"
# Print header
echo "========== $output_file =========="
cat "$output_file"
