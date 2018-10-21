# import libraries
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from matplotlib import pyplot as plt


# load train and validation datasets
def load_dataset(path, photos_to_biz, labels, should_split, return_train=True):
    # get filenames
    processed_filenames, processed_labels = [], []
    filenames = os.listdir(path=path)
    for filename in filenames:
        # remove filename that contains underscore
        if '_' not in filename:
            processed_filenames.append(filename)
    
    # delete this line when actually training!!!!!!!!!!!!!!!! only test code works for now
    # processed_filenames = processed_filenames[:1000]
    
    # get each photo's target labels
    for filename in processed_filenames:
        bus_id = photos_to_biz.query('train_photo_id=={}'.format(filename.split(".", 1)[0]))['business_id']
        this_labels = labels.query("business_id=={}".format(bus_id.iloc[0]))['labels'].iloc[0]
        
        tmp_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if isinstance(this_labels, float) and math.isnan(this_labels):
            # empty cell
            pass
        else:
            # multiple labels
            this_labels = this_labels.split(" ")
            for label in this_labels:
                tmp_labels[int(label)] = 1

        processed_labels.append(tmp_labels)
    
    if should_split:
        X_train, X_validation, y_train, y_validation = train_test_split(np.array(processed_filenames), 
                                                                        np.array(processed_labels), 
                                                                        test_size = 0.2, 
                                                                        random_state = 0)
        if return_train:
            return X_train, y_train
        else:    # return validation set
            return X_validation, y_validation
    
    
    # just return filenames and labels
    return np.array(processed_filenames), np.array(processed_labels)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_folder, img_filenames):
    list_of_tensors = [path_to_tensor(img_folder + '/train_photos/' + img_filename) for img_filename in img_filenames]
    return np.vstack(list_of_tensors)


def print_images(tensors, datagen_train, title="Untitled"):
    f1, subplots_array = plt.subplots(4,4, figsize=(20, 20))
    f1.suptitle(title, fontsize=35)
    if datagen_train:
        for X_batch in datagen_train.flow(tensors):
            row, col = 0, 0
            for i in range(0, 16):
                if col == 4:
                    col = 0
                    row += 1

                subplots_array[row, col].imshow(X_batch[i], interpolation='nearest', aspect='auto')
                col += 1

            # show the plot
            plt.show()
            break
            
    else:
        for i in range(0, 16):
            row, col = 0, 0
            for i in range(0, 16):
                if col == 4:
                    col = 0
                    row += 1

                subplots_array[row, col].imshow(tensors[i], interpolation='nearest', aspect='auto')
                col += 1

            # show the plot
            plt.show()
            break