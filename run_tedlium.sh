#!/bin/bash
#

. cmd.sh
. path.sh

# Directory where Kaldi's tedlium recipe is trained
tedlium_dir=../tedlium/s5

nj=80
decode_nj=8

# Train a duration model
#./local/train_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir ~/tools/pylearn2 \
  #--stage 4 \
  #--language ENGLISH \
  #--stress-dict local/lat-model/data/en/cmudict.0.7a.lc \
  #$tedlium_dir/data/train $tedlium_dir/data/lang $tedlium_dir/exp/tri3_ali $tedlium_dir/exp/dur_model_tri3 || exit 1

# Rescore lattices using the duration model
./local/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
  --language ENGLISH --fillers "!SIL,[BREATH],[NOISE],[COUGH],[SMACK],[UM],[UH]" --stress-dict local/lat-model/data/en/cmudict.0.7a.lc \
  --score-cmd $tedlium_dir/local/score.sh \
  --stage 0 \
  $tedlium_dir/data/lang \
  $tedlium_dir/exp/tri3/graph \
  $tedlium_dir/data/dev \
  $tedlium_dir/exp/tri3_mmi_b0.1/decode_dev_it4 \
  $tedlium_dir/exp/dur_model_tri3 \
  $tedlium_dir/exp/dur_model_tri3/decode_dev_it4 || exit 1
