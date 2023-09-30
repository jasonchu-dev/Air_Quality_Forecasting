# Air Quality Time Series Forecasting

Prediction on air quality dataset using deep learning.

Dataset: https://www.kaggle.com/datasets/aayushkandpal/air-quality-time-series-data-uci

## Neural Network

Convolutional Recurrent Neural Network:
* 2 Expansion blocks: Expand features into more features column wise
    * 1d convolutional layer
    * Batch normalization
    * Sigmoid activation
* LSTM
    * 2 layers
    * 24 neurons per hidden layer
* 2 Compression blocks: Scale features down in both axis
    * 1d convolutional layer
    * Batch normalization
    * Sigmoid activation
* 1d convolutional layer: Final scale down to right dimensions
* Sigmoid activation

## Run

Clone
```
    git clone https://github.com/jasonchu-dev/Air_Quality_Forecasting.git
```
Download dataset and unzip. Create a folder called "Data" in project directory. Move excel dataset from unzipped and into "Data".

Get pip then pip install modules
```
    pip install -r requirements.txt
```
Or get conda then conda install
```
    conda install --file requirements.txt
```
Run scripts
To train
```
    bash train.sh
```
To test
```
    bash test.sh
```
Or do both
```
    bash train_test.sh
```
Latest models and training logs will be saved in a "models" and "logs"

## Others

To clean accumalation of models and logs
```
    bash clean.sh
```
Adjust hyperparameters if need be in configs folder. Current hyperparameters are suggested.
