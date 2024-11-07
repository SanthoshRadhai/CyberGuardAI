import csv

input_file_path = 'train.csv'
output_file_path = 'cleaned_file.csv'

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        try:
            writer.writerow(row)
        except csv.Error as e:
            # Skip rows that cause errors
            print(f"Skipping problematic row: {e}")
