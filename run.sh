#!/bin/bash
#

. cmd.sh
. path.sh

# Directory with a trained acoustic model
src_exp_dir=../kaldi-et/build/exp/nnet5a_pnorm_fast

# Training data
src_data_dir=../kaldi-et/build/data/train

#Test data
src_testdata_dir=../kaldi-et/build/data/test

# Training lang directory
src_lang_dir=../kaldi-et/build/data/lang

# Directory where the duration model will be created
dur_model_dir=../kaldi-et/build/exp/dur_model_nnet5a_pnorm_fast

# Baseline decoding lattices
src_decode_dir=../kaldi-et/build/exp/nnet5a_pnorm_fast/decode

graph_dir=../kaldi-et/build/exp/tri3b/graph

score_cmd=../kaldi-et/local/score.sh

PYLEARN_DIR=~/tools/pylearn2

nj=`cat $src_exp_dir/num_jobs`
decode_nj=`cat $src_decode_dir/num_jobs`


stage=0
. utils/parse_options.sh # accept options

# Create aligned training data
steps/nnet2/align.sh --nj $nj --cmd "$train_cmd" \
    ../kaldi-et/build/data/train ../kaldi-et/build/data/lang \
    ../kaldi-et/build/exp/nnet5a_pnorm_fast ../kaldi-et/build/exp/nnet5a_pnorm_fast_ali



# Train a duration model
./local/train_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" \
  --language ESTONIAN \
  ../kaldi-et/build/data/train ../kaldi-et/build/data/lang ../kaldi-et/build/exp/nnet5a_pnorm_fast_ali ../kaldi-et/build/exp/dur_model_nnet5a_pnorm_fast

# Rescore lattices using the duration model
./local/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
  --language ESTONIAN --fillers "<sil>,++garbage++" \
  ../kaldi-et/build/exp/tri3b/graph \
  ../kaldi-et/build/data/test \
  ../kaldi-et/build/exp/nnet5a_pnorm_fast/decode \
  ../kaldi-et/build/exp/dur_model_nnet5a_pnorm_fast \
  ../kaldi-et/build/exp/dur_model_nnet5a_pnorm_fast/decode
