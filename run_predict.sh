#!/bin/bash

# TO PREDICT
export BERT_BASE_DIR=/uncased_L-12_H-768_A-12  # download BERT model from Google repo first
export DATA_DIR= # the data you want use for prediction
export TRAINED_CLASSIFIER= # directory of your trained model, for example, reddit data

# the following line runs 4 workers (if you have multiple GPUS)
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python run_classifier_hvd.py \
    --task_name=reddit \
    --do_predict=True \
    --data_dir=$DATA_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$TRAINED_CLASSIFIER \
    --max_seq_length=128 \
    --output_dir=results/