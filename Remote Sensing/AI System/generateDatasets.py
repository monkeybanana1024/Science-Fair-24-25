import os
import random
import csv

# Function to get all subdirectories
def get_subdirectories(root_dir):
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Function to write list to file
def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

# Function to create analytics.csv in each subdirectory
def create_analytics_file(directory):
    filepath = os.path.join(directory, 'analytics.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'MAE', 'MSE', 'RMSE', 'SPC', 'AV', 'PV'])  # Header row

# Main directory containing all subdirectories
root_directory = 'Data/Datasets/'

# Get all subdirectories
all_subdirs = get_subdirectories(root_directory)

# Shuffle the list of subdirectories
random.shuffle(all_subdirs)

# Calculate split indices
total = len(all_subdirs)
train_split = int(0.6 * total)
test_split = int(0.8 * total)

# Split the directories
train_dirs = all_subdirs[:train_split]
test_dirs = all_subdirs[train_split:test_split]
valid_dirs = all_subdirs[test_split:]

# Write to files
write_to_file('AI System/train.txt', train_dirs)
write_to_file('AI System/test.txt', test_dirs)
write_to_file('AI System/valid.txt', valid_dirs)

# Create analytics.csv in each test and validation directory
for subdir in test_dirs:
    create_analytics_file(os.path.join(root_directory, subdir))

for subdir in valid_dirs:
    create_analytics_file(os.path.join(root_directory, subdir))

print(f"Total directories: {total}")
print(f"Train set: {len(train_dirs)} directories")
print(f"Test set: {len(test_dirs)} directories")
print(f"Validation set: {len(valid_dirs)} directories")
