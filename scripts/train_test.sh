#!/bin/bash

echo "Train - standing by..."
python ../src/train.py --hyperparameters ../configs/hyperparameters.yaml

echo "Test - standing by..."
python ../src/test.py