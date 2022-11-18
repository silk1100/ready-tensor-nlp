import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import datetime as dt
os.chdir("/opt/ml_vol/")

def preprocess(now):
    df = pd.read_csv("./inputs/data/training/textClassificationBaseMainInput/clickbait_train.csv", index_col='id')
    df_train, df_val = train_test_split(df, test_size=0.1)

    titles_train = df_train['text']
    train_labels = df_train['class']
    titles_val = df_val['text']
    val_labels = df_val['class']

    labelobj = LabelEncoder()
    train_labs = labelobj.fit_transform(train_labels)
    val_labs = labelobj.transform(val_labels)

    count_vect = CountVectorizer()
    x_train_vect = count_vect.fit_transform(titles_train)
    x_val_vect = count_vect.transform(titles_val)
    

    tfid_tr = TfidfTransformer()
    x_train_tfid = tfid_tr.fit_transform(x_train_vect).toarray()
    x_val_tfid = tfid_tr.transform(x_val_vect).toarray()
    X_train = np.concatenate([x_train_tfid, train_labs.reshape(-1,1)], axis=1)
    X_val = np.concatenate([x_val_tfid, val_labs.reshape(-1,1)], axis=1)
    np.savez(f"./inputs/data/training/textClassificationBaseMainInput/x_train_val_clickbait_{now}.npz",
     x_train_tfid, x_val_tfid)

    
    with open(f"./models/artifacts/pre_countVec_clickbait_{now}.p", 'wb') as f:
        joblib.dump(count_vect, f)
    with open(f"./models/artifacts/pre_tfid_clickbait_{now}.p", 'wb') as f:
        joblib.dump(tfid_tr, f)
    with open(f"./models/artifacts/pre_labelenc_clickbait_{now}.p", 'wb') as f:
        joblib.dump(labelobj, f)

    return X_train, X_val

if __name__ == "__main__":
    # preprocess()
    pass