import os
import random
import csv

def get_subdirectories(root_dir):
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def create_analytics_file(directory):
    filepath = os.path.join(directory, 'analytics.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'MAE', 'MSE', 'RMSE', 'SPC', 'AV', 'PV'])

root_directory = 'Data/Datasets/'
all_subdirs = get_subdirectories(root_directory)
random.shuffle(all_subdirs)

# Split into 5 folds
fold_size = len(all_subdirs) // 5
folds = [all_subdirs[i:i + fold_size] for i in range(0, len(all_subdirs), fold_size)]

# Ensure we have exactly 5 folds
while len(folds) > 5:
    folds[-2].extend(folds[-1])
    folds.pop()

# Write each fold to a separate file
for i, fold in enumerate(folds, 1):
    write_to_file(f'AI System/f{i}.txt', fold)

# Create analytics.csv in each directory
for subdir in all_subdirs:
    create_analytics_file(os.path.join(root_directory, subdir))

print(f"Total directories: {len(all_subdirs)}")
for i, fold in enumerate(folds, 1):
    print(f"Fold {i}: {len(fold)} directories")
