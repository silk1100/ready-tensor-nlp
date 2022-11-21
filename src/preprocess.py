import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os
import constants



def preprocess():
    # Read the training data
    df = pd.read_csv(os.path.join(constants.MAIN_DIR, constants.DATA_TRAINING),
         index_col=constants.id_field)

    # Split data to have a validation set
    df_train, df_val = train_test_split(df, test_size=0.1)

    # Extract labels and feature columns
    titles_train = df_train[constants.document_field]
    train_labels = df_train[constants.target_class]
    titles_val = df_val[constants.document_field]
    val_labels = df_val[constants.target_class]

    text_preprocess_pipeline = Pipeline([
        ('count_vec', CountVectorizer()),
        ('tfid', TfidfTransformer()),
    ])

    labelobj = LabelEncoder()
    train_labs = labelobj.fit_transform(train_labels)
    val_labs = labelobj.transform(val_labels)

    x_train_tfid = text_preprocess_pipeline.fit_transform(titles_train)
    x_val_tfid = text_preprocess_pipeline.transform(titles_val)

    X_train = np.concatenate([x_train_tfid.toarray(), train_labs.reshape(-1,1)], axis=1)
    X_val = np.concatenate([x_val_tfid.toarray(), val_labs.reshape(-1,1)], axis=1)

    with open(os.path.join(constants.MAIN_DIR,constants.MODELS_ARTIFACT_PREPROCESS), 'wb') as f:
        joblib.dump(text_preprocess_pipeline, f)

    with open(os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_LABEL_PREPROCESS), 'wb') as f:
        joblib.dump(labelobj, f)


    return X_train, X_val

if __name__ == "__main__":
    preprocess()
    pass