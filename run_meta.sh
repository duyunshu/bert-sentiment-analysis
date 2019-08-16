#!/bin/bash

# TO TRAIN
export BERT_BASE_DIR=/uncased_L-12_H-768_A-12  # download BERT model from Google repo first
export DATA_DIR=/data/metacritic

# the following line runs 4 workers (if you have multiple GPUS)
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python run_classifier_hvd.py \
    --task_name=meta \
    --do_train=True \
    --do_eval=True \
    --data_dir=$DATA_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=4.0 \
    --output_dir=results/
    # you can also freeze the BERT layers by adding --freeze=True