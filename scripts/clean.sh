#!/bin/bash

logs="../logs"
models="../models"
data="../data"
ext="pkl"
__pycache__="../src/__pycache__"

if [ -d "$logs" ]; then
    rm -r "$logs"
    echo "$logs deleted"
else
    echo "$logs not found"
fi

if [ -d "$models" ]; then
    rm -r "$models"
    echo "$models deleted"
else
    echo "$models not found"
fi

if [ -d "$data" ]; then
    find "$data" -type f -name "*.$ext" -exec rm {} \;
    echo "$ext deleted from $data"
else
    echo "$data not found"
fi

if [ -d "$__pycache__" ]; then
    rm -r "$__pycache__"
    echo "$__pycache__ deleted"
else
    echo "$__pycache__ not found"
fi