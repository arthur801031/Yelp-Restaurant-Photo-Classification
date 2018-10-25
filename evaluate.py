'''
Evaluate.py calculates the F1-mean score across the entire validation set
'''
# import libraries
# user-defined functions, can be found in this directory
from helper_functions import * 
from config import bottleneck_path

validation_features = load_bottleneck_features(bottleneck_path + 'Resnet50_validation')
validation_true_labels = load_bottleneck_features(bottleneck_path + 'Resnet50_validation-labels')

model = Model(validation_features.shape[1:])

model.load_weights('weights.best.from_Resnet50.hdf5')

print("Validation Set Mean F1 score: {}".format(calculate_meanf1(model.predict(validation_features), validation_true_labels)))