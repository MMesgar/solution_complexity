#!/bin/sh

# theano device
device=gpu0

# model parameters
MODE='-nonstatic'
WRD2VEC='-word2vec' # -word2vec -rand
N_EPOCHS=50
BATCH_SIZE=32
DROPOUT_RATE=0.5


exp_id='nonstatic_word2vec_one_avg_2_3'
OUT_DIR='./evalutions/'$exp_id'/'
IN_DIR='./data/corpus_2_3/'

mkdir $OUT_DIR

THEANO_FLAGS=device=$device,floatX=float32,lib.cnmem=0.7 python -u model_one_avg.py $MODE $WRD2VEC $N_EPOCHS $BATCH_SIZE  $DROPOUT_RATE $IN_DIR $OUT_DIR
