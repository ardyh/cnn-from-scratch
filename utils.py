import os
import numpy as np
import pandas as pd
import skimage.io
from skimage.transform import resize
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_data(train_dir):
    '''
    Params:
        - train_dir : string = directory of data, 
                                relative to the file being run
    Returns:
        - target_names : array of string = names of target clases 
        - img_paths : array of string = list of image paths 
    '''
    target_names = []
    target_val = []
    img_paths = []
    for idx, (dirname, subdirs, filenames) in enumerate(os.walk(train_dir)):
        if (idx == 0):
            target_names = subdirs
        else:
            for filename in filenames:
                target_val.append(idx - 1)
                img_paths.append(os.path.join(dirname, filename))

    return target_names, target_val, img_paths

def k_fold_cross_val(model, X, y, input_shape, k=10):
    history = []
    kfold = KFold(n_splits=k)
    i = 0
    for train_idx, test_idx in kfold.split(X):
        print(f"Fold #{i}")
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model.train(X_train, y_train, epochs=6)
        preds = model.predict(X_test)

        history.append({
            'accuracy': accuracy_score(y, preds),
            'f1': f1_score(y, preds),
            'precision': precision_score(y, preds),
            'recall': recall_score(y, preds),
        })

        i+=1

    pd.DataFrame(history).to_csv("history.csv")
    return history


def load_and_resize_image(image_path, image_shape):
    try:
        raw_img = skimage.io.imread(image_path)
        raw_img = resize(raw_img, image_shape)
    except:
        raise Exception("Failed to load image")

    return raw_img