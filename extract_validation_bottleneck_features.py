# import libraries
import pandas as pd
import numpy as np
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

# user-defined functions, can be found in this directory
from helper_functions import * 
from extract_bottleneck_features import *


img_folder = '../'
# img_folder = "C:\\Users\\I-Chun Liu\\Documents\\Local_Code\\final_project\\data"

train_photo_to_biz_ids = pd.read_csv(img_folder + '/train_photo_to_biz_ids.csv.tgz', compression='gzip', sep=',')
train_photo_to_biz_ids.columns = ['train_photo_id', 'business_id']

train_labels = pd.read_csv(img_folder + '/train.csv.tgz', compression='gzip', sep=',')
train_labels.columns = ['business_id', 'labels']

print("loading dataset...")
X_validation, y_validation = load_dataset(img_folder + '/train_photos', train_photo_to_biz_ids, train_labels, True, False)

print("Validation set has {} samples.".format(X_validation.shape[0]))

validation_tensors = paths_to_tensor(img_folder, X_validation).astype('float32')/255
print("Validation tenors size: {}".format(validation_tensors.shape))

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

np.save('./bottleneck_features/Resnet50_validation', validation_features)
np.save('./bottleneck_features/Resnet50_validation_labels', y_validation)
print("Successfully saved validation bottleneck features for Resnet50_train")