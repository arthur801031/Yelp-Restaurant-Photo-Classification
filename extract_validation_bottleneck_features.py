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


train_photo_to_biz_ids = pd.read_csv(img_folder + 'train_photo_to_biz_ids.csv.tgz', compression='gzip', sep=',')
train_photo_to_biz_ids.columns = ['train_photo_id', 'business_id']

train_labels = pd.read_csv(img_folder + 'train.csv.tgz', compression='gzip', sep=',')
train_labels.columns = ['business_id', 'labels']

print("loading dataset...")
X_validation, y_validation = load_dataset(img_folder + 'train_photos', train_photo_to_biz_ids, train_labels, True, False)

num_images = X_validation.shape[0]
print("Validation set has {} samples.".format(num_images))

i, prev_i, done, progress_bar, num_files = num_images_per_batch, 0, False, tqdm(total=math.ceil(num_images/num_images_per_batch)), 1

while not done:
    if i >= num_images:
        X_validation_parts = X_validation[prev_i:]
        y_validation_parts = y_validation[prev_i:]
        done = True
    else:
        X_validation_parts = X_validation[prev_i:i]
        y_validation_parts = y_validation[prev_i:i]
    
    # conver RGB image to 4D tensor with shape (1, 244, 244, 3)
    validation_tensors = paths_to_tensor(img_folder, 'train_photos', X_validation_parts).astype('float32')/255
    print("\n Validation tenors size: {}".format(validation_tensors.shape))

    # image pre-processing
    print("perform image augmentations for validation data")
    datagen_validation = image.ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.05,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')
    datagen_validation.fit(validation_tensors)

    # extract bottleneck features
    validation_features = extract_Resnet50(validation_tensors * 255)

    print("Validation features shape: {}".format(validation_features.shape))

    np.save(bottleneck_path + 'Resnet50_validation_{}'.format(num_files), validation_features)
    np.save(bottleneck_path + 'Resnet50_validation-labels_{}'.format(num_files), y_validation_parts)

    print("Successfully saved validation bottleneck features for Resnet50_validation_{}".format(num_files))
    prev_i = i
    i += num_images_per_batch
    num_files += 1
    progress_bar.update(1)