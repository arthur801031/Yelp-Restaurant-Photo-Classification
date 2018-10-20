
def extract_VGG16(tensor):
    from keras.applications.vgg16 import VGG16, preprocess_input
    print("processing tensor...")
    p_tensor = preprocess_input(tensor)
    print("extracting features using VGG16...")
    return VGG16(weights='imagenet', include_top=False).predict(p_tensor)

def extract_VGG19(tensor):
    from keras.applications.vgg19 import VGG19, preprocess_input
    print("processing tensor...")
    p_tensor = preprocess_input(tensor)
    print("extracting features using VGG19...")
    return VGG19(weights='imagenet', include_top=False).predict(p_tensor)

def extract_Resnet50(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    print("processing tensor...")
    p_tensor = preprocess_input(tensor)
    print("extracting features using Resnet50...")
    return ResNet50(weights='imagenet', include_top=False).predict(p_tensor)

def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    print("processing tensor...")
    p_tensor = preprocess_input(tensor)
    print("extracting features using Xception...")
    return Xception(weights='imagenet', include_top=False).predict(p_tensor)

def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    print("processing tensor...")
    p_tensor = preprocess_input(tensor)
    print("extracting features using InceptionV3...")
    return InceptionV3(weights='imagenet', include_top=False).predict(p_tensor)