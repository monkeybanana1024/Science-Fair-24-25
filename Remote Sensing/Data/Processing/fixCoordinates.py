import re

def convert_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    converted_lines = []
    for line in lines:
        # Check if the line is already correctly formatted
        if re.match(r'^-?\d+\.\d+,-\d+\.\d+$', line.strip()):
            converted_lines.append(line)
        else:
            # Remove any non-numeric characters except for the decimal point, minus sign, and digits
            clean_line = ''.join(c for c in line if c.isdigit() or c in '.-')
            # Try to extract the coordinate pair
            match = re.search(r'(-?\d+\.\d+)(-\d+\.\d+)', clean_line)
            if match:
                lat, lon = match.groups()
                converted_lines.append(f"{lat},{lon}\n")
            else:
                # If we can't extract coordinates, keep the original line
                converted_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(converted_lines)

    print(f"Coordinates in {file_path} have been successfully processed.")

file_path = 'Collection/coordinates.txt'
convert_coordinates(file_path)
