#!/bin/bash
#
# Copyright 2014 Tanel Alumäe
# License: BSD 3-clause
#
# Example of how to train a duration model on tedlium data, using
# it's exp/nnet2_online/nnet_ms_sp as the baseline model. 
# You must train the baseline model and run the corresponding decoding
# experiments prior to running this script.
#
# Also, create a symlink: ln -s <kaldi-nnet-dur-model-dir>/dur-model
#
# Then run the script from the $KALDI_ROOT/egs/tedlium/s5 directory.
#
#

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


nj=80            # Must be the same as when training the baseline model
decode_nj=8      # Must be the same as when decoding using the baseline 

stage=0 # resume training with --stage=N
pylearn_dir=~/tools/pylearn2
# Aggregating traing data for duration model needs more RAM than default in our SLURM cluster
aggregate_data_args="--mem 8g"

left_context=4
right_context=2

h0_dim=400
h1_dim=400

. utils/parse_options.sh || exit 1;

if [ $stage -le 0 ]; then
  # Align training data using a model in exp/nnet2_online/nnet_ms_sp_ali
  steps/nnet2/align.sh --nj $nj --cmd "$train_cmd" --use-gpu no  \
      --transform-dir "$transform_dir" --online-ivector-dir exp/nnet2_online/ivectors_train_hires \
      data/train_hires data/lang exp/nnet2_online/nnet_ms_sp exp/nnet2_online/nnet_ms_sp_ali || exit 1
fi



if [ $stage -le 1 ]; then
  # Train a duration model based on alignments in exp/nnet2_online/nnet_ms_sp_ali
  ./dur-model/train_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd --mem 8g" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
    --stage 0 \
    --left-context $left_context --right-context $right_context \
    --language ENGLISH \
    --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
    --h0-dim $h0_dim --h1-dim $h1_dim \
    data/train_hires data/lang exp/nnet2_online/nnet_ms_sp_ali exp/dur_model_nnet_ms_sp || exit 1
fi



if [ $stage -le 2 ]; then
	# Decode dev/test data, try different duration model scales and phone insertion penalties.
	for decode_set in  dev test; do
	  num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
	  # Rescore dev lattices using the duration model
	  ./dur-model/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd --mem 8g" --nj $num_jobs \
		--language ENGLISH --fillers "!SIL,[BREATH],[NOISE],[COUGH],[SMACK],[UM],[UH]" --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
		--scales "0.2 0.3 0.4" --penalties "0.11 0.13 0.15 0.17 0.19 0.21" \
		--stage 0 \
		--left-context $left_context --right-context $right_context \
		data/lang \
		exp/tri3/graph \
		data/${decode_set}_hires \
		exp/nnet2_online/nnet_ms_sp_online/decode_${decode_set}_utt_offline.rescore \
		exp/dur_model_nnet_ms_sp \
		exp/nnet2_online/nnet_ms_sp_online/decode_${decode_set}_utt_offline.rescore.dur-rescore || exit 1;
	done
fi

#RESULTS

#Baseline results for DEV:

#$ for x in exp/nnet2_online/*/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null | grep dev

#%WER 14.1 | 507 17792 | 88.6 8.6 2.8 2.7 14.1 86.6 | -0.101 | exp/nnet2_online/nnet_ms_sp/decode_dev/score_10_0.0/ctm.filt.filt.sys
#%WER 13.0 | 507 17792 | 89.5 7.7 2.8 2.5 13.0 83.6 | -0.172 | exp/nnet2_online/nnet_ms_sp/decode_dev.rescore/score_10_0.0/ctm.filt.filt.sys
#%WER 14.0 | 507 17792 | 88.5 8.5 3.0 2.6 14.0 86.8 | -0.104 | exp/nnet2_online/nnet_ms_sp_online/decode_dev/score_11_0.0/ctm.filt.filt.sys
#%WER 13.0 | 507 17792 | 89.5 7.8 2.8 2.4 13.0 84.0 | -0.294 | exp/nnet2_online/nnet_ms_sp_online/decode_dev.rescore/score_10_0.0/ctm.filt.filt.sys
#%WER 14.4 | 507 17792 | 88.4 8.9 2.7 2.8 14.4 88.0 | -0.116 | exp/nnet2_online/nnet_ms_sp_online/decode_dev_utt/score_10_0.0/ctm.filt.filt.sys
#%WER 13.2 | 507 17792 | 89.3 7.9 2.8 2.5 13.2 83.8 | -0.284 | exp/nnet2_online/nnet_ms_sp_online/decode_dev_utt.rescore/score_10_0.0/ctm.filt.filt.sys
#%WER 13.9 | 507 17792 | 88.8 8.6 2.6 2.7 13.9 86.8 | -0.127 | exp/nnet2_online/nnet_ms_sp_online/decode_dev_utt_offline/score_10_0.0/ctm.filt.filt.sys
#%WER 12.7 | 507 17792 | 89.6 7.7 2.7 2.3 12.7 83.6 | -0.285 | exp/nnet2_online/nnet_ms_sp_online/decode_dev_utt_offline.rescore/score_10_0.0/ctm.filt.filt.sys

#After rescoring with the duration model:
#$ for x in exp/nnet2_online/*/decode*dur-rescore; do [ -d $x ] && grep Sum $x/s*/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null | grep dev 

#%WER 12.1 | 507 17792 | 90.0 7.4 2.6 2.1 12.1 82.1 | -0.348 | exp/nnet2_online/nnet_ms_sp_online/decode_dev_utt_offline.rescore.dur-rescore/s0.3_p0.11/score_11_0.5/ctm.filt.filt.sys

#Baseline results fro TEST
#$ for x in exp/nnet2_online/*/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null | grep test

#%WER 13.2 | 1155 27512 | 88.5 8.3 3.2 1.7 13.2 81.6 | -0.110 | exp/nnet2_online/nnet_ms_sp/decode_test/score_10_0.5/ctm.filt.filt.sys
#%WER 11.8 | 1155 27512 | 90.0 7.2 2.8 1.8 11.8 78.1 | -0.195 | exp/nnet2_online/nnet_ms_sp/decode_test.rescore/score_10_0.0/ctm.filt.filt.sys
#%WER 13.1 | 1155 27512 | 88.8 8.4 2.8 2.0 13.1 81.4 | -0.151 | exp/nnet2_online/nnet_ms_sp_online/decode_test/score_10_0.0/ctm.filt.filt.sys
#%WER 11.7 | 1155 27512 | 90.0 7.2 2.8 1.8 11.7 78.1 | -0.292 | exp/nnet2_online/nnet_ms_sp_online/decode_test.rescore/score_10_0.0/ctm.filt.filt.sys
#%WER 13.6 | 1155 27512 | 88.4 8.7 2.8 2.1 13.6 82.5 | -0.128 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt/score_10_0.0/ctm.filt.filt.sys
#%WER 12.1 | 1155 27512 | 89.7 7.4 2.8 1.9 12.1 79.7 | -0.304 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt.rescore/score_10_0.0/ctm.filt.filt.sys
#%WER 13.0 | 1155 27512 | 88.9 8.3 2.8 1.9 13.0 82.5 | -0.147 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt_offline/score_10_0.0/ctm.filt.filt.sys
#%WER 11.7 | 1155 27512 | 90.0 7.2 2.7 1.8 11.7 79.4 | -0.290 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt_offline.rescore/score_10_0.0/ctm.filt.filt.sys

#After rescoring with the duration model:
#$ for x in exp/nnet2_online/*/decode*dur-rescore; do [ -d $x ] && grep Sum $x/s*/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null | grep test

#%WER 10.9 | 1155 27512 | 90.8 6.9 2.3 1.8 11.0 78.6 | -0.351 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt_offline.rescore.dur-rescore/s0.2_p0.11/score_10_0.0/ctm.filt.filt.sys
