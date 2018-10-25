# import libraries
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

# user-defined functions, can be found in this directory
from helper_functions import * 
from extract_bottleneck_features import *
from config import img_folder, num_images_per_batch, bottleneck_path


print("loading dataset...")
X_test = load_dataset(img_folder + 'test_photos', None, None, True, True)

num_images = X_test.shape[0]
print("Test set has {} samples.".format(num_images))

i, prev_i, done, progress_bar, num_files = num_images_per_batch, 0, False, tqdm(total=math.ceil(num_images/num_images_per_batch)), 1

while not done:
    if i >= num_images:
        X_test_parts = X_test[prev_i:]
        done = True
    else:
        X_test_parts = X_test[prev_i:i]


    # conver RGB image to 4D tensor with shape (1, 244, 244, 3)
    test_tensors = paths_to_tensor(img_folder, 'test_photos', X_test_parts).astype('float32')/255
    print("\nTest tenors size: {}".format(test_tensors.shape))

    # perform image augmentations
    print("perform image augmentations for test data")
    datagen_test = image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.05,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

    datagen_test.fit(test_tensors)


    # extract bottleneck features
    test_features = extract_Resnet50(test_tensors * 255)

    print("Test features shape: {}".format(test_features.shape))

    np.save(bottleneck_path + 'Resnet50_test_{}'.format(num_files), test_features)

    print("Successfully saved test bottleneck features for Resnet50_test_{}".format(num_files)) 

    prev_i = i
    i += num_images_per_batch
    num_files += 1
    progress_bar.update(1)