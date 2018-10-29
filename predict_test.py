# import libraries
import pandas as pd
import numpy as np
import math
import os
from tqdm import tqdm
# user-defined functions, can be found in this directory
from helper_functions import *
from config import img_folder, bottleneck_path, num_images_per_batch

NUM_BOTTLENECK_FEATURES = 24

def start_predict_from_scratch():
    test_photos_to_biz_id = pd.read_csv(img_folder + 'test_photo_to_biz.csv.tgz', compression='gzip', sep=',')
    test_photos_to_biz_id.columns = ['test_photo_id', 'business_id']
    test_photos_to_biz_id = test_photos_to_biz_id.dropna()

    # initilize output submission file format
    result_dict, result_dict_probs = {}, {}
    for bus_id in test_photos_to_biz_id['business_id']:
        result_dict[bus_id] = ""
        result_dict_probs[bus_id] = []

    # load test photo ids 
    photo_ids = load_test_photo_ids(img_folder + 'test_photos')

    model, prev_i, photo_ids_parts = None, 0, None
    # process each test' bottleneck feature and prediction separately due to computer's memory constraint 
    for i in tqdm(range(1, NUM_BOTTLENECK_FEATURES+1)):
        print("\nLoading test bottleneck features {}".format(i))
        bottleneck_feature = np.load(bottleneck_path + 'Resnet50_test_' + str(i) + '.npy')

        model = Model(bottleneck_feature.shape[1:])
        model.load_weights('weights.best.from_Resnet50.hdf5')   

        # compute photo_ids slice' indices
        prev_i_adjusted = prev_i * num_images_per_batch
        if i == NUM_BOTTLENECK_FEATURES:
            photo_ids_parts = photo_ids[prev_i_adjusted:]
        else:
            i_adjusted = i * num_images_per_batch
            photo_ids_parts = photo_ids[prev_i_adjusted:i_adjusted]

        print("Predicting test labels...")
        predict_labels = model.predict(bottleneck_feature)

        for p_id, p_label in zip(photo_ids_parts, predict_labels):        
            associated_bus_ids = test_photos_to_biz_id.loc[test_photos_to_biz_id['test_photo_id'] == int(p_id), 'business_id']
            for associated_bus_id in associated_bus_ids:
                result_dict_probs[associated_bus_id].append(p_label)

        prev_i = i
        print("Finished test bottleneck features {}\n\n".format(i))


    np.save('result_dict.npy', result_dict)
    np.save('result_dict_probs.npy', result_dict_probs)

    return result_dict, result_dict_probs


def load_from_files():
    result_dict = np.load('result_dict.npy').item()
    result_dict_probs = np.load('result_dict_probs.npy').item()
    return result_dict, result_dict_probs


result_dict, result_dict_probs = start_predict_from_scratch()
result_dict, result_dict_probs = load_from_files()


for bus_id in result_dict_probs:
    final_bus_label = ""
    arrs = result_dict_probs[bus_id]                    # return [[], [], [], [],...], which are image(s)' predicted labels for this business id
    if len(arrs) > 0:
        bus_id_labels_average = np.mean(arrs, axis=0)       # get the average probabilities for each predicted labels for this business id
        for idx, label_prob in enumerate(bus_id_labels_average):
            if label_prob >= 0.488:
                final_bus_label += str(idx) + " "
        final_bus_label = final_bus_label.strip()           # remove trailing space(s)
        result_dict[bus_id] = final_bus_label



final_result = pd.DataFrame.from_dict(list(result_dict.items()))
final_result.columns = ['business_id', 'labels']
final_result.to_csv('my_submission.csv', sep=',', encoding='utf-8', index=False)