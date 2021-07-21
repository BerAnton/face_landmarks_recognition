# Facial landmarks recognition

Solution to Facial landmarks recognition challenge (MADE Mail.ru) - https://www.kaggle.com/c/made-cv-2021-contest-01-facial-landmarks
ResNext50 was used for prediction 971 points on face.
Metric for the competition was MSE. Result for this net - 9.71.

# Install requirements:
CUDA 11.1 must be installed on machine.<br>
Libs can be obtained via requirements:<br>
``` pip install -r requirements.txt ```

# Configure train and predict process:
Folder "configs" contains two yml files:
    - train_params.yml - describes configuration of train process.
    - predict_params.yml - describes configuration of prediction process.

# Train:
``` python main.py train --config-file path_to_config ```

Default is "configs/train_params.yml"

# Predict
``` python main.py predict --config-file path_to_config ```

Default is "configs/predict_params.yml"
