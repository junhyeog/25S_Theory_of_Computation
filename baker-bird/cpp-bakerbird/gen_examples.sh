#!/bin/bash

for i in $(seq 1 100); do
    output_file="example${i}.txt"
    echo "Generating ${output_file}..."
    ./input_generator "${output_file}"
done
