import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import datetime as dt
from preprocess import preprocess
from sklearn.naive_bayes import MultinomialNB
import os

os.chdir("/opt/ml_vol")

now = dt.datetime.today().strftime("%y-%m-%d-%H-%M-%S")

if __name__ == "__main__":
    Xtrain, Xval = preprocess(now)

    # Xtrain_val = np.load(f"./data/training/textClassificationBaseMainInput/x_train_val_clickbait_{now}.npz")
    # with open(f"./models/artifacts/pre_countVec_clickbait_{now}.p", 'rb') as f:
    #     count_vect = joblib.load(f)
    # with open(f"./models/artifacts/pre_tfid_clickbait_{now}.p", 'rb') as f:
    #     tfid_td = joblib.load(f)
    # with open(f"./models/artifacts/pre_labelenc_clickbait_{now}.p", 'rb') as f:
    #     labelobj = joblib.load(f)

    X_train, y_train = Xtrain[:,:-1], Xtrain[:, -1]
    X_val, y_val= Xval[:,:-1], Xval[:, -1]

    clc = MultinomialNB()
    clc.fit(X_train, y_train)
    print(clc.score(X_train, y_train))
    print(clc.score(X_val, y_val))
