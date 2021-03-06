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
X_train, y_train = load_dataset(img_folder + 'train_photos', train_photo_to_biz_ids, train_labels, True, True)

num_images = X_train.shape[0]
print("Training set has {} samples.".format(num_images))

i, prev_i, done, progress_bar, num_files = num_images_per_batch, 0, False, tqdm(total=math.ceil(num_images/num_images_per_batch)), 1

while not done:
    if i >= num_images:
        X_train_parts = X_train[prev_i:]
        y_train_parts = y_train[prev_i:]
        done = True
    else:
        X_train_parts = X_train[prev_i:i]
        y_train_parts = y_train[prev_i:i]

    # conver RGB image to 4D tensor with shape (1, 244, 244, 3)
    train_tensors = paths_to_tensor(img_folder, 'train_photos', X_train_parts).astype('float32')/255
    print("\n Train tenors size: {}".format(train_tensors.shape))

    # perform image augmentations
    print("perform image augmentations for training data")
    datagen_train = image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.05,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

    datagen_train.fit(train_tensors)


    # extract bottleneck features
    train_features = extract_Resnet50(train_tensors * 255)

    print("Train features shape: {}".format(train_features.shape))

    np.save(bottleneck_path + 'Resnet50_train_{}'.format(num_files), train_features)
    np.save(bottleneck_path + 'Resnet50_train-labels_{}'.format(num_files), y_train_parts)

    print("Successfully saved train bottleneck features for Resnet50_train_{}".format(num_files))
    prev_i = i
    i += num_images_per_batch
    num_files += 1
    progress_bar.update(1)