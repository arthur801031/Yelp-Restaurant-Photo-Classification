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
X_train, y_train = load_dataset(img_folder + '/train_photos', train_photo_to_biz_ids, train_labels, True, True)

print("Training set has {} samples.".format(X_train.shape[0]))

# conver RGB image to 4D tensor with shape (1, 244, 244, 3)
train_tensors = paths_to_tensor(img_folder, X_train).astype('float32')/255
print("Train tenors size: {}".format(train_tensors.shape))

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

np.save('./bottleneck_features/Resnet50_train', train_features)
np.save('./bottleneck_features/Resnet50_train_labels', y_train)
print("Successfully saved train bottleneck features for Resnet50_train")