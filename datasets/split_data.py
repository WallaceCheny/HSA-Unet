import os
import random
from pathlib import Path

def split_dataset(folder_path, train_ratio=0.8):
    """Splits a folder of files into train and test sets.

    Args:
        folder_path: Path to the folder containing the files.
        train_ratio: Proportion of data to include in the training set.
    """
    file_names = os.listdir(folder_path)
    random.shuffle(file_names)

    split_index = int(len(file_names) * train_ratio)
    train_files = file_names[:split_index]
    test_files = file_names[split_index:]

    with open('train.txt', 'w') as f:
        for file_name in train_files:
            f.write(file_name + '\n')

    with open('test.txt', 'w') as f:
        for file_name in test_files:
            f.write(file_name + '\n')

    print("Dataset split complete.")
    print(f"Training set: {len(train_files)} files")
    print(f"Testing set: {len(test_files)} files")

if __name__ == '__main__':
    # Example usage:
    relativ_path = 'stomach/images'
    folder_path = Path.cwd() / relativ_path # Replace with your folder path
    split_dataset(folder_path)