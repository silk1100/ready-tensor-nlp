# Text Classification Phase I
## Description (TL/DR)
This project implements the specifications of [ReadyTensor](https://beta.readytensor.com/).

All python scripts can be found within `./src/` directory. 


    src
    ├── preprocess.py  # Preprocess text column using scikit-learn pipeline         
    ├── constants.py   # Containing all directory and utility functions 
    ├── train.py       # Main script for training. Training Entry point
    ├── test           # Main script for prediction. Testing entry point
    ├── __init__.py             


## How to run
1. Build the docker image.<br>
`sudo docker buildx build -t <imagename> .`
2. Create a docker volume to mount your data<br>
`sudo docker volume create --name <vname> --opt --type=none --opt device=<absolute-path-to-ml_vol> --opt o=bind`
3. Run the training script.<br>
`sudo docker run --rm --name <container-name> --mount source=<vname>,target=/opt/ml_vol <imagename> train`
4. Check the testing results of the trained model. <br>
`sudo docker run -v <vname>:/opt/ml_vol --rm <imagename> test`
