import py7zr
import numpy as np
import tempfile
import os

# Specify the path to your 7z file and the file you want to access
seven_zip_file_path = 'n10_susc.7z'
file_to_access = 'n10_susc/n10_conus.tif'  # The file you want to access inside the 7z archive

# Create a temporary directory for extraction
with tempfile.TemporaryDirectory() as temp_dir:
    # Open the 7z file
    with py7zr.SevenZipFile(seven_zip_file_path, mode='r') as z:
        # Extract all files into the temporary directory
        z.extract(path=temp_dir)

    # Now access the specific file in the temporary directory
    extracted_file_path = os.path.join(temp_dir, file_to_access)

    # Check if the extracted file exists
    if os.path.exists(extracted_file_path):
        # Memory-map the extracted file
        mmap_array = np.memmap(extracted_file_path, dtype='S', mode='r', shape=os.path.getsize(extracted_file_path))

        # Now you can access the data from the memory-mapped array
        print(mmap_array.tobytes().decode('utf-8'))  # Assuming the content is text

        # Explicitly delete the memory-mapped array to release it before exiting the context
        del mmap_array  # This ensures that resources are released

    else:
        print(f"File {file_to_access} not found in the extracted files.")