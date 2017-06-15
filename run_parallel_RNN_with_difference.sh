#!/bin/sh

# model parameters
MODE='-nonstatic'
WRD2VEC='-word2vec' # -word2vec -rand
N_EPOCHS=50
BATCH_SIZE=32
DROPOUT_RATE=0.5
max_sl=600

# corpus 2-3
exp_id='prnn_with_difference_basic_2_3'
OUT_DIR='./evalutions/'$exp_id'/'
IN_DIR='./data/corpus_2_3/'

mkdir -p $OUT_DIR

THEANO_FLAGS=device=gpu0,floatX=float32,lib.cnmem=0.7 nohup python -u model_prnn_with_difference_basic.py $MODE $WRD2VEC $N_EPOCHS $BATCH_SIZE  $DROPOUT_RATE $IN_DIR $OUT_DIR $max_sl > ./script_out/$exp_id.log &

# corpus 4-5
exp_id='prnn_with_difference_basic_4_5'
OUT_DIR='./evalutions/'$exp_id'/'
IN_DIR='./data/corpus_4_5/'

mkdir -p $OUT_DIR

THEANO_FLAGS=device=gpu1,floatX=float32,lib.cnmem=0.7 nohup python -u model_prnn_with_difference_basic.py $MODE $WRD2VEC $N_EPOCHS $BATCH_SIZE  $DROPOUT_RATE $IN_DIR $OUT_DIR $max_sl > ./script_out/$exp_id.log &

# corpus gr-5
exp_id='prnn_with_difference_basic_gr_5'
OUT_DIR='./evalutions/'$exp_id'/'
IN_DIR='./data/corpus_gr_5/'

mkdir -p $OUT_DIR

THEANO_FLAGS=device=gpu2,floatX=float32,lib.cnmem=0.7 nohup python -u model_prnn_with_difference_basic.py $MODE $WRD2VEC $N_EPOCHS $BATCH_SIZE  $DROPOUT_RATE $IN_DIR $OUT_DIR $max_sl > ./script_out/$exp_id.log  &

# corpus all
exp_id='prnn_with_difference_basic_all'
OUT_DIR='./evalutions/'$exp_id'/'
IN_DIR='./data/corpus_all/'

mkdir -p $OUT_DIR

THEANO_FLAGS=device=gpu3,floatX=float32,lib.cnmem=0.7 nohup python -u model_prnn_with_difference_basic.py $MODE $WRD2VEC $N_EPOCHS $BATCH_SIZE  $DROPOUT_RATE $IN_DIR $OUT_DIR $max_sl > ./script_out/$exp_id.log  &