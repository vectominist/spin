#!/bin/bash

libri_dir=$1
json_dir=$2

mkdir -p $json_dir

python3 prepare_data.py \
    $libri_dir \
    $json_dir \
    --split train-clean-100 \
            dev-clean \
            dev-other \
    --sort-by-len
