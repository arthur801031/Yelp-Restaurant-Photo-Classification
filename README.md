# Udacity's Machine Learning Engineer Nanodegree Capstone Project - Yelp Restaurant Photo Classification

### Description:
In this project, I build a model that automatically tags restaurants with multiple labels using user-uploaded photos and labels provided by Yelp. This model uses Keras' pre-trained convolutional neural network, ResNet50, to extract bottleneck features. The bottleneck features are fed into a GlobalAverage2D layer and an output layer with sigmoid activation for training. This model scores a mean F1 score of 0.78339 on the test dataset. If you're interested in the detail of this project, please see report.pdf.

### Installations:
Create a conda environment with the following packages:
* Numpy
* Pandas
* Tensorflow
* Keras
* Sklearn
* Matplotlib
* Tqdm
* Glob
* opencv-python [optional: only use in Visualization.ipynb]

### Datasets:
The datasets can be found [here](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data).

### How to run:
**Step 1:** Change `config.py`:

`img_folder` is the file path that contains the datasets.

`bottleneck_path` is where you want your bottleneck features to be stored.

`slash` is `'\\'` for Windows OS and `'/'` for Unix/Mac OS.

**Step 2:** Extract training dataset's bottleneck features:
```
python extract_train_bottleneck_features.py
```

**Step 3:** Extract validation dataset's bottleneck features:
```
python extract_validation_bottleneck_features.py
```

**Step 4:** Extract test dataset's bottleneck features:
```
python extract_test_bottleneck_features.py
```

**Step 5:** Traing the model:
```
python train.py
```

**Step 6:** See validation dataset's mean F1 score:
```
python evaluate.py
```

**Step 7:** Generate prediction results:
```
python predict_test.py
```

If you just want to see the prediction results, you could comment out `result_dict, result_dict_probs = start_predict_from_scratch()` in `predict_test.py` and just run `python predict_test.py`. If you want to see the validation dataset's mean F1 score, you could run `python evaluate.py`. Make sure `weights.best.from_Resnet50.hdf5`, `result_dict.npy`, and `result_dict_probs.npy` are in the directory.

**Note:** Training dataset's bottleneck features takes up 70.2 GB of storage. Validation dataset's bottleneck features takes up 17.5 GB of storage. Test dataset's bottleneck features takes up 88.6 GB of storage.
