#!/bin/bash

# TO CREATE PRETRAINNING DATA
# This outputs tfrecord file
export BERT_BASE_DIR=/uncased_L-12_H-768_A-12  # download BERT model from Google repo first
export DATA_DIR=/pretrain_data/reddit_pretraintext.txt
export OUT_DATA_DIR=pretrain_results/

python create_pretraining_data.py \
  --input_file=$DATA_DIR \
  --output_file=$OUT_DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3


# TO RUN PRETRAINING
export BERT_BASE_DIR=/home/ydu/BERT/uncased_L-12_H-768_A-12  # download BERT model from Google repo first
export INPUT_DIR=pretrain_results/ # where your $OUT_DATA_DIR is
export DATA_DIR=model_results/ # where you want to save the pretraining model
 
# the following line runs 4 workers (if you have multiple GPUS)
mpirun -np 4 \
  -H localhost:4 \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python run_pretraining_hvd.py \
  --input_file=$INPUT_DIR/tf_examples.tfrecord \
  --output_dir=$DATA_DIR/ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=10000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \