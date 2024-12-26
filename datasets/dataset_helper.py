import os
import shutil
import random

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

original_data_path = '/content/drive/MyDrive/HSAUnet/CVC-ClinicDB/CVC-ClinicDB/Original'  # Path to your original data
ground_truth_path = '/content/drive/MyDrive/HSAUnet/CVC-ClinicDB/CVC-ClinicDB/Ground Truth'  # Path to your ground truth data
train_folder = '/content/drive/MyDrive/HSAUnet/CVC-ClinicDB/train'  # Path for the training folder
test_folder = '/content/drive/MyDrive/HSAUnet/CVC-ClinicDB/test'  # Path for the testing folder
train_file = 'train.txt'
test_file = '/test.txt'

train_folder_img = os.path.join(train_folder, 'images')
train_folder_gt = os.path.join(train_folder, 'masks')
os.makedirs(train_folder_img, exist_ok=True)  # Create train folder if it doesn't exist
os.makedirs(train_folder_gt, exist_ok=True)  # Create test folder if it doesn't exist

test_folder_img = os.path.join(test_folder, 'images')
test_folder_gt = os.path.join(test_folder, 'masks')
os.makedirs(test_folder_img, exist_ok=True)  # Create train folder if it doesn't exist
os.makedirs(test_folder_gt, exist_ok=True)  # Create test folder if it doesn't exist

def move_files(file_path, destination_folder):
  with open(file_path, 'r') as f:
    for line in f:
      file_name = line.strip()
      source_file_name = os.path.join(original_data_path, file_name)
      ground_truth_file_name = os.path.join(ground_truth_path, file_name)

      destination_path = os.path.join(destination_folder + "/images/", file_name)
      destination_path_gt = os.path.join(destination_folder + "/masks/", file_name)
      shutil.move(source_file_name, destination_path)
      shutil.move(ground_truth_file_name, destination_path_gt)



if __name__ == '__main__':
    # Example usage:
    folder_path = 'D:\WorkSpaces\HSA-Unet\datasets\CVC-ClinicDB\Original'  # Replace with your folder path
    split_dataset(folder_path)

    move_files(train_file, train_folder)
    move_files(test_file, test_folder)
