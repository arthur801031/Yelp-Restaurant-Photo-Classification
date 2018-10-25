# import libraries
from tqdm import tqdm
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint  

# user-defined functions, can be found in this directory
from helper_functions import * 
from config import bottleneck_path


arr_train_valid_tuples = get_arr_train_valid_tuples(
    bottleneck_path + 'Resnet50_train',
    bottleneck_path + 'Resnet50_train-labels',
    bottleneck_path + 'Resnet50_validation',
    bottleneck_path + 'Resnet50_validation-labels'
)

# define model
model = Model(np.load(arr_train_valid_tuples[0][0]).shape[1:])

checkpointer = ModelCheckpoint(filepath='weights.best.from_Resnet50.hdf5', verbose=1, save_best_only=True)


# initlizations
epochs_of_all_files = 5
epochs_of_each_tuple = 5
batch_size = 32 
# start training
for i in tqdm(range(epochs_of_all_files)):

    for train_valid_tuple in arr_train_valid_tuples:

        train_features = np.load(train_valid_tuple[0])
        train_labels = np.load(train_valid_tuple[1])
        valid_features = np.load(train_valid_tuple[2])
        valid_labels = np.load(train_valid_tuple[3])
            
        model.fit(train_features,
                    train_labels,
                    epochs=epochs_of_each_tuple,
                    batch_size=batch_size,
                    validation_data=(valid_features, valid_labels),
                    callbacks=[checkpointer],
                    verbose=1,
                    shuffle=True)
    
        print("Validation Mean F1 score: {}".format(calculate_meanf1(model.predict(valid_features), valid_labels)))

