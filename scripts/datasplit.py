import os
import random
import shutil
import numpy as np
random.seed(10)
encoders = ["VAE","RandomEncoder","safe_VAE","MrMPC"]
for encoder in encoders:
    # Define the paths for the input folder and the output folders
    input_folder_path = '/mnt/data/collected_data_'+encoder+'/'
    train_folder_path = '/mnt/newclstmdata/train'
    val_folder_path = '/mnt/newclstmdata/val'
    test_folder_path = '/mnt/newclstmdata/test'

    # Define the percentage split for each set
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1

    # Get a list of all the files in the input folder
    file_list = os.listdir(input_folder_path)
    # Adding a prefix based on encoder
    # file_list = [encoder+'_'+file for file in file_list]
    # TODO: Only get the files that are .npy
    file_list = [file for file in file_list if file.endswith('.npy')]
    # For the mpc encoder, we only want to use one view.

    # Shuffle the file list randomly
    random.shuffle(file_list)

    # Get the number of files in the input folder
    num_files = len(file_list)

    # Calculate the number of files in each set
    num_train_files = int(train_split * num_files)
    num_val_files = int(val_split * num_files)
    num_test_files = int(test_split * num_files)

    # Create the output folders if they don't already exist
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(val_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)

    # Move the first num_train_files files to the training folder
    for i in range(num_train_files):
        file_name = file_list[i]
        if encoder == "MrMPC":
            bothviews = np.load(os.path.join(input_folder_path, file_name)) # shape: (video length, 2, height, width, channels)
            view1 = bothviews[:,0,:,:,:]
            # We only want this view as the second one is segmented one.
            np.save(os.path.join(train_folder_path, encoder+"_"+file_name), view1)
            continue
        src_path = os.path.join(input_folder_path, file_name)
        dst_path = os.path.join(train_folder_path, encoder+"_"+file_name)
        shutil.copy(src_path, dst_path)

    # Move the next num_val_files files to the validation folder
    for i in range(num_train_files, num_train_files+num_val_files):
        file_name = file_list[i]
        if encoder == "MrMPC":
            bothviews = np.load(os.path.join(input_folder_path, file_name)) # shape: (video length, 2, height, width, channels)
            view1 = bothviews[:,0,:,:,:]
            # We only want this view as the second one is segmented one.
            np.save(os.path.join(val_folder_path, encoder+"_"+file_name), view1)
            continue
        src_path = os.path.join(input_folder_path, file_name)
        dst_path = os.path.join(val_folder_path, encoder+"_"+file_name)
        shutil.copy(src_path, dst_path)

    # Move the last num_test_files files to the testing folder
    for i in range(num_train_files+num_val_files, num_files):
        file_name = file_list[i]
        if encoder == "MrMPC":
            bothviews = np.load(os.path.join(input_folder_path, file_name)) # shape: (video length, 2, height, width, channels)
            view1 = bothviews[:,0,:,:,:]
            # We only want this view as the second one is segmented one.
            np.save(os.path.join(test_folder_path, encoder+"_"+file_name), view1)
            continue
        src_path = os.path.join(input_folder_path, file_name)
        dst_path = os.path.join(test_folder_path, encoder+"_"+file_name)
        shutil.copy(src_path, dst_path)

    print('Dataset split complete.')