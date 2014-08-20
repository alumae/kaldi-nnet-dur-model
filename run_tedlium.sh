#!/bin/bash
#
# Copyright 2014 Tanel Alumäe
# License: BSD 3-clause
#
# Example of how to train a duration model on tedlium data, using
# it's exp/tri3_mmi_b0.1 as a baseline model. 
# You must train the baseline model and run the corresponding decoding
# experiments prior to running this script.
#
# Also, create a symlink: ln -s <kaldi-nnet-dur-model-dir>/dur-model
#
# Then run the script from the $KALDI_ROOT/egs/tedlium/s5 directory.
#

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


nj=80            # Must be the same as when training the baseline model
decode_nj=8      # Must be the same as when decoding using the baseline 

stage=0 # resume training with --stage=N
pylearn_dir=~/tools/pylearn2

. utils/parse_options.sh || exit 1;

if [ $stage -le 0 ]; then
  # Align training data using a model in exp/tri3_mmi_b0.1
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/tri3_mmi_b0.1 exp/tri3_mmi_b0.1_ali || exit 1
fi

if [ $stage -le 1 ]; then
  # Train a duration model based on alignments in exp/tri3_mmi_b0.1_ali
  ./dur-model/train_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir \
    --stage 0 \
    --language ENGLISH \
    --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
    data/train data/lang exp/tri3_mmi_b0.1_ali exp/dur_model_tri3_mmi_b0.1 || exit 1
fi

if [ $stage -le 2 ]; then
  # Rescore dev lattices using the duration model
  ./dur-model/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
    --language ENGLISH --fillers "!SIL,[BREATH],[NOISE],[COUGH],[SMACK],[UM],[UH]" --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
    --scales "0.2 0.3" --penalties "0.15 0.18 0.20" \
    --stage 0 \
    data/lang \
    exp/tri3/graph \
    data/dev \
    exp/tri3_mmi_b0.1/decode_dev_it4 \
    exp/dur_model_tri3_mmi_b0.1 \
    exp/dur_model_tri3_mmi_b0.1/decode_dev_it4 || exit 1
fi

if [ $stage -le 3 ]; then
  # Rescore test lattices using the duration model
  ./dur-model/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
    --language ENGLISH --fillers "!SIL,[BREATH],[NOISE],[COUGH],[SMACK],[UM],[UH]" --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
    --scales "0.2 0.3" --penalties "0.15 0.18 0.20" \
    --stage 0 \
    data/lang \
    exp/tri3/graph \
    data/test \
    exp/tri3_mmi_b0.1/decode_test_it4 \
    exp/dur_model_tri3_mmi_b0.1 \
    exp/dur_model_tri3_mmi_b0.1/decode_test_it4 || exit 1
fi
