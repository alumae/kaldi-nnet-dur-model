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
# Aggregating traing data for duration model needs more RAM than default in our SLURM
aggregate_data_args="--mem=8g"


. utils/parse_options.sh || exit 1;

if [ $stage -le 0 ]; then
  # Align training data using a model in exp/tri3_mmi_b0.1
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/tri3_mmi_b0.1 exp/tri3_mmi_b0.1_ali || exit 1
fi

if [ $stage -le 1 ]; then
  # Train a duration model based on alignments in exp/tri3_mmi_b0.1_ali
  ./dur-model/train_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args="$aggregate_data_args" \
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


#=== RESULTS ===
#DEV SPEAKERS
 #WER 35.9 | 507 17792 | 69.9 23.0 7.0 5.8 35.9 98.6 | 0.029 | exp/tri1/decode_dev/score_12/ctm.filt.filt.sys
#%WER 31.6 | 507 17792 | 74.1 19.9 6.0 5.7 31.6 96.8 | -0.093 | exp/tri2/decode_dev/score_15/ctm.filt.filt.sys
#%WER 27.4 | 507 17792 | 78.3 16.6 5.1 5.7 27.4 96.4 | -0.145 | exp/tri3/decode_dev/score_16/ctm.filt.filt.sys
#%WER 32.0 | 507 17792 | 74.6 20.0 5.4 6.6 32.0 97.4 | -0.229 | exp/tri3/decode_dev.si/score_13/ctm.filt.filt.sys
#%WER 23.9 | 507 17792 | 81.2 14.8 4.0 5.0 23.9 93.9 | -0.108 | exp/tri3_mmi_b0.1/decode_dev_it4/score_13/ctm.filt.filt.sys
#%WER 23.3 | 507 17792 | 81.8 14.4 3.8 5.1 23.3 93.9 | -0.132 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.2_p0.15/score_14/ctm.filt.filt.sys
#%WER 23.3 | 507 17792 | 81.7 14.3 3.9 5.1 23.3 93.9 | -0.127 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.2_p0.18/score_14/ctm.filt.filt.sys
#%WER 23.3 | 507 17792 | 81.7 14.4 3.9 5.0 23.3 93.9 | -0.132 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.2_p0.20/score_14/ctm.filt.filt.sys
#%WER 23.2 | 507 17792 | 81.8 14.4 3.7 5.1 23.2 93.7 | -0.154 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.3_p0.15/score_15/ctm.filt.filt.sys
#%WER 23.3 | 507 17792 | 81.9 14.4 3.7 5.2 23.3 93.7 | -0.154 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.3_p0.18/score_14/ctm.filt.filt.sys
#%WER 23.2 | 507 17792 | 81.8 14.3 3.9 5.0 23.2 93.7 | -0.155 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.3_p0.20/score_15/ctm.filt.filt.sys

#TEST SPEAKERS
#%WER 34.9 | 1155 27512 | 69.5 24.1 6.4 4.5 34.9 96.8 | 0.100 | exp/tri1/decode_test/score_13/ctm.filt.filt.sys
#%WER 29.9 | 1155 27512 | 74.6 20.2 5.3 4.5 29.9 95.3 | 0.016 | exp/tri2/decode_test/score_15/ctm.filt.filt.sys
#%WER 24.8 | 1155 27512 | 79.4 16.4 4.2 4.2 24.8 93.5 | -0.054 | exp/tri3/decode_test/score_16/ctm.filt.filt.sys
#%WER 29.9 | 1155 27512 | 74.6 20.3 5.2 4.5 29.9 95.2 | -0.115 | exp/tri3/decode_test.si/score_16/ctm.filt.filt.sys
#%WER 21.3 | 1155 27512 | 82.4 14.0 3.6 3.8 21.3 92.4 | 0.045 | exp/tri3_mmi_b0.1/decode_test_it4/score_13/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 82.7 13.8 3.5 3.6 20.8 91.5 | 0.028 | exp/dur_model_tri3/decode_test_it4_s0.2_p0.18/score_15/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 83.0 13.7 3.3 3.8 20.8 91.4 | 0.029 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.2_p0.15/score_14/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 82.9 13.7 3.4 3.7 20.8 91.6 | 0.024 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.2_p0.18/score_14/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 82.9 13.7 3.4 3.7 20.8 91.5 | 0.024 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.2_p0.20/score_14/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 83.1 13.8 3.1 4.0 20.8 90.6 | 0.008 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.3_p0.15/score_14/ctm.filt.filt.sys
#%WER 20.9 | 1155 27512 | 83.0 13.8 3.2 3.9 20.9 90.9 | 0.006 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.3_p0.18/score_14/ctm.filt.filt.sys
#%WER 20.9 | 1155 27512 | 83.0 13.8 3.3 3.9 20.9 91.0 | 0.007 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.3_p0.20/score_14/ctm.filt.filt.sys
