import sys
sys.path.append('.')
sys.path.append('./src/backend')
sys.path.append('./src')
sys.path.append('/usr/src')
sys.path.append('/usr/src/backend')
sys.path.append('/usr/')
print(sys.path)

from flask import Flask, request
from src.backend.utils import write_error
from src.backend import testing
from src.backend import utils 
import pandas as pd
from io import StringIO

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello World!!"

@app.route('/ping')
def pingme():
    try:
        return {'status_code':200}
    except Exception as e:
        write_error("frontend-app", f"Failed to /ping due to {e}")

@app.route("/infer", methods=['POST'])
def infer():
    if request.method == "POST":
        try:
            df = pd.read_csv(StringIO(request.get_data().decode('utf-8')))
            df_p = testing.infer(df)
            return df_p.to_dict()
        except Exception as e:
            write_error('frontend-app', f"Failed to make prediction because of {e}")
    else:
        write_error("frontend-app", "/infer must receive a POST request")

    return {'status_code': 500}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)