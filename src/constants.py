import os
import json

os.chdir("/opt/ml_vol")
MAIN_DIR = "/opt/ml_vol"

DATA_CONFIG = "inputs/data_config/clickbait_schema.json"
DATA_TRAINING = "inputs/data/training/textClassificationBaseMainInput/clickbait_train.csv"
DATA_TESTING = "inputs/data/testing/textClassificationBaseMainInput/clickbait_test.csv"
DATA_TESTING_KEY = "inputs/data/testing/textClassificationBaseMainInput/clickbait_test_key.csv"

MODELS_ARTIFACT_PREPROCESS = "models/artifacts/preprocess_pipeline.p"
MODELS_ARTIFACT_LABEL_PREPROCESS = "models/artifacts/preprocess_label_pipeline.p"
MODELS_ARTIFACT_MODELS = "models/artifacts/model_pipeline.p"

OUTPUT_ERRORS = "outputs/errors"
OUTPUT_HPT = "outputs/hpt_errors"
OUTPUT_TESTING = "outputs/testing_outputs/preds.csv"


def read_config():
    with open(DATA_CONFIG, 'r') as f:
        data = json.load(f)
    return data['inputDatasets']['textClassificationBaseMainInput']

def write_error(source_script:str, msg: str, mode=None):
    if mode is None:
        if os.path.exists(os.path.join(OUTPUT_ERRORS, "testing.txt")):
            mode = 'a'
        else:
            mode = 'w'

    with open(os.path.join(OUTPUT_ERRORS, f"{source_script}.txt"), mode) as f:
        f.write(msg)


DATA = read_config()
id_field = DATA['idField']
target_class = DATA['targetField']
document_field = DATA['documentField']