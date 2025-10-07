./compile.sh

# Loop over example1.txt to example1000.txt.
for i in $(seq 1 7); do
    input_file="example${i}.txt"
    output_file="output${i}.txt"
    check_file="check${i}.txt"
    
    # Run the main program with the input file and direct output to the corresponding output file.
    ./run_algorithm.sh $input_file $output_file
    ./run_checker.sh $input_file $output_file $check_file

    # read check_file and check it is equal to "yes"
    if [ -s $check_file ]; then
        result=$(cat $check_file)
        if [ "$result" = "yes" ]; then
            echo "Test case ${i} passed."
        else
            echo "Test case ${i} failed."
        fi
    else
        echo "Test case ${i} failed."
    fi

    
done
