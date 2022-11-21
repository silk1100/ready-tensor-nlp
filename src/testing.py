import joblib
import constants
import pandas as pd
import os
import warnings
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def main():
    df = pd.read_csv(os.path.join(constants.MAIN_DIR, constants.DATA_TESTING), index_col=constants.id_field)
    
    if os.path.exists(os.path.join(constants.MAIN_DIR, constants.DATA_TESTING_KEY)):
        df_true = pd.read_csv(os.path.join(constants.MAIN_DIR, constants.DATA_TESTING_KEY),
         index_col=constants.id_field)
        try: 
            assert(len(df) == len(df_true))
        except Exception as e:
            constants.write_error("testing", "Length of testing key")
    else:
        df_true = None


    if not os.path.exists(os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_MODELS)):
        print("Model can not be found. Run train script first")
        constants.write_error("testing", "Model can't be found. Run train script first")
        return

    with open(os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_MODELS), 'rb') as f:
        model = joblib.load(f)

    if not os.path.exists(os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_PREPROCESS)):
        print("There is no a preprocess model")
        warnings.warn("There is no preprocessing model in models/artifact")
        pmodel = None
    else:
        with open(os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_PREPROCESS), 'rb') as f:
            pmodel = joblib.load(f)


    if not os.path.exists(os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_LABEL_PREPROCESS)):
        print("There is no a preprocess model for labels")
        warnings.warn("There is no preprocessing model in models/artifact for labels")
        plmodel = None
    else:
        with open(os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_LABEL_PREPROCESS), 'rb') as f:
            plmodel = joblib.load(f)

    if pmodel:
        X = pmodel.transform(df[constants.document_field])
    else:
        X = df[constants.document_field].values
    
    ypred = model.predict(X)
    df_pred = pd.DataFrame(data=ypred, index=df.index)
    df_pred.to_csv(constants.OUTPUT_TESTING)

    if df_true is not None:
        ytrue = df_true[constants.target_class]
        ytrue = plmodel.transform(ytrue)
        print("confusion matrix: ", confusion_matrix(ytrue, ypred))
        print("balanced accuracy score: ", balanced_accuracy_score(ytrue, ypred))
        print("f1-score: ", f1_score(ytrue, ypred))
        print("precision: ", precision_score(ytrue, ypred))
        print("recall: ", recall_score(ytrue, ypred))


if __name__ == "__main__":
    main()