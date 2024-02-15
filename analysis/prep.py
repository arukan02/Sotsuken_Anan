import csv

# Input and output filenames result_20221014_yoko_1 
input_file = '1017/result_yoko.csv'
output_file = 'prep_yoko1.csv'


# Function to convert a string containing a list of numbers to a list of coordinate pairs
def parse_list(s):
    coordinates = [float(x.strip("[]")) for x in s.split(',')]
    return [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]

# Function to convert a value to [(0, 0)] if it's missing
def convert_value(value):
    if '[' in value:
        return parse_list(value)
    else:
        return [(0, 0)]

# Open the input and output files
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header line
    writer.writerow(['ID', 'Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12','Column13','Column14'])

    # Loop through the rows in the input CSV
    for row in reader:
        # Extract the ID and the list of values
        ID = int(row[0])
        values = [convert_value(value) for value in row[1:]]

        # Fill in missing values with [(0, 0)]
        while len(values) < 14:
            values.append([(0, 0)])

        # Write the modified row to the output CSV
        writer.writerow([ID] + [item for sublist in values for item in sublist])