import os


def change_class_label(activity_label_file, output_file):
    # Open the input file for reading
    with open(activity_label_file, 'r') as file:
        lines = file.readlines()

    # Open the output file for writing
    with open(output_file, 'w') as file:
        for line in lines:
            parts = line.split()
            # Check if the third column is '8' and replace it with '7'
            if parts[2] == '8':
                parts[2] = '7'
            # Write the modified line to the output file
            file.write('\t'.join(parts) + '\n')

    print(f"Processed lines have been written to {output_file}.")


def check_class_label(activity_label_file):
    with open(activity_label_file, 'r') as file:
        for line_num, line in enumerate(file, start=1):  # Enumerate lines starting from line 1
            parts = line.split()
            try:
                class_label = int(parts[2])
                if not (0 <= class_label <= 7) and not (class_label == 10 or class_label == 12):
                    raise ValueError(f"Line {line_num}: Class label '{class_label}' is not in the range of 0 to 7.")
            except (IndexError, ValueError) as e:
                print(f"Error processing line {line_num}: {str(e)}")


if __name__ == "__main__":

    # Define the input and output file names
    input_file = os.path.join(os.getcwd(), 'dataset\\activity_labels.txt')
    output_file = os.path.join(os.getcwd(), 'dataset\\new_activity_labels.txt')

    change_class_label(input_file, output_file)
    check_class_label(output_file)




