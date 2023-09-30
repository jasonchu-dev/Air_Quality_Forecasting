#!/bin/bash

logs="../logs"
models="../models"
__pycache__="../src/__pycache__"
data="../data"
ext="pkl"

delete_folders() {
    local folder="$1"
    if [ -d "$folder" ]; then
        rm -r "$folder"
        echo "$folder deleted"
    else
        echo "$folder not found"
    fi
}

delete_files() {
    local folder="$1"
    local ext="$2"
    if [ -d "$folder" ]; then
        find "$folder" -type f -name "*.$ext" -exec rm {} \;
        echo "$ext deleted from $folder"
    else
        echo "$folder not found"
    fi
}

delete_folders "$logs"
delete_folders "$models"
delete_folders "$__pycache__"
delete_files "$data" "$ext"