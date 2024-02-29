#!/bin/bash

while getopts g:m: flag
do 
    case "${flag}" in 
        g) cuda=${OPTARG};;
        m) pretrained_models=$(echo ${OPTARG} | tr "," "\n");;
    esac
done 


export CUDA_VISIBLE_DEVICES=$cuda

eval "$(conda shell.bash hook)"


conda activate easyedit
    
python -m edit_minigpt4

exit_code=$?
if [[ $exit_code = 1 ]]; then 
    exit 
fi
    
