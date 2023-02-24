"""

This code uses the hashlib library to calculate the MD5 hash of each file in a
given directory and subdirectories. It then stores the hash and path of each
file in a dictionary, and checks if the hash already exists in the dictionary.
If the hash already exists, it means that the file is a duplicate, and the code
removes it from the computer using the os.remove function.

To use the code, simply run it and enter the directory to scan when prompted.
Note that the code permanently deletes files, so use it with caution and make
sure to back up important files before running it.

"""


import os
import hashlib

def remove_duplicates(directory):
    # Create a dictionary to store file hashes and paths
    hashes = {}

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Get the full path of the file
            path = os.path.join(root, filename)

            # Calculate the hash of the file
            with open(path, 'rb') as file:
                filehash = hashlib.md5(file.read()).hexdigest()

            # Check if the hash already exists in the dictionary
            if filehash in hashes:
                print(f"Removing duplicate file: {path}")
                os.remove(path)
            else:
                hashes[filehash] = path

if __name__ == '__main__':
    directory = input("Enter the directory to scan: ")
    remove_duplicates(directory)
