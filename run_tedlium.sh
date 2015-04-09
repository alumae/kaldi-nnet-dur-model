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

left_context=4
right_context=2

h0_dim=400
h1_dim=400
speaker_projection_dim=100

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
		--ignore_speakers true \
		data/lang \
		exp/tri3/graph \
		data/${decode_set}_hires \
		exp/nnet2_online/nnet_ms_sp_online/decode_${decode_set}_utt_offline \
		exp/dur_model_nnet_ms_sp \
		exp/dur_model_nnet_ms_sp/decode_${decode_set}_utt_offline || exit 1;
	done
fi

exit

if [ $stage -le 4 ]; then
  # Train a duration model based on alignments in exp/tri3_mmi_b0.1_ali
  ./dur-model/train_sat_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
    --stage 0 \
    --left-context $left_context --right-context $right_context \
    --language ENGLISH \
    --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
    --h0-dim $h0_dim --h1-dim $h1_dim --speaker-projection-dim $speaker_projection_dim \
    data/train data/lang exp/tri3_mmi_b0.1_ali exp/dur_model_sat_tri3_mmi_b0.1 || exit 1
fi


if [ $stage -le 5 ]; then
  for set in dev test; do 
    ## Create a new data directory from hypotheses
    #echo "Creating fake data directory for $set data from baseline hypotheses"
    #rm -rf exp/tri3_mmi_b0.1/decode_${set}_it4/data
    #mkdir -p exp/tri3_mmi_b0.1/decode_${set}_it4/data    
    #cp -r data/${set}/* exp/tri3_mmi_b0.1/decode_${set}_it4/data
    #rm -rf exp/tri3_mmi_b0.1/decode_${set}_it4/data/split*
    ## FIXME: we take the hyps of LM weight 12 -- should take the best one based on dev set performance
    #cat exp/tri3_mmi_b0.1/decode_${set}_it4/scoring/12.tra | int2sym.pl -f 2- data/lang/words.txt > exp/tri3_mmi_b0.1/decode_${set}_it4/data/text
  
    # Align training data using a model in exp/tri3_mmi_b0.1
    echo "Aligning $set data"
    steps/align_fmllr.sh --nj $decode_nj --cmd "$train_cmd" \
        data/${set} data/lang exp/tri3_mmi_b0.1 exp/tri3_mmi_b0.1_${set}_ali || exit 1
  done
fi



if [ $stage -le 6 ]; then
  for set in dev test; do 
    echo "Adapting SAT duration model to $set speakers"
    ./dur-model/adapt_dur_model_sat.sh --nj $decode_nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
      --stage 0 \
      --language ENGLISH \
      --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
      --h0-dim $h0_dim --h1-dim $h1_dim --speaker-projection-dim $speaker_projection_dim \      
      data/$set data/lang exp/tri3_mmi_b0.1_${set}_ali \
      exp/dur_model_sat_tri3_mmi_b0.1 exp/dur_model_sat_tri3_mmi_b0.1_${set} || exit 1
  done
fi

if [ $stage -le 7 ]; then
  for set in dev test; do 
    echo "Decoding using SAT duration model adapted to $set speakers"
    # Rescore lattices using the adapted SAT duration model
    ./dur-model/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
      --language ENGLISH --fillers "!SIL,[BREATH],[NOISE],[COUGH],[SMACK],[UM],[UH]" --stress-dict dur-model/python/lat-model/data/en/cmudict.0.7a.lc \
      --scales "0.2 0.3 0.4" --penalties "0.15 0.18 0.20 0.22" \
      --stage 0 \
      data/lang \
      exp/tri3/graph \
      data/${set} \
      exp/tri3_mmi_b0.1/decode_${set}_it4 \
      exp/dur_model_sat_tri3_mmi_b0.1_${set} \
      exp/dur_model_sat_tri3_mmi_b0.1_${set}/decode_${set}_it4 || exit 1
    done
fi



#=== RESULTS ===

#DEV SPEAKERS

#%WER 35.9 | 507 17792 | 69.9 23.0 7.0 5.8 35.9 98.6 | 0.029 | exp/tri1/decode_dev/score_12/ctm.filt.filt.sys
#%WER 31.6 | 507 17792 | 74.1 19.9 6.0 5.7 31.6 96.8 | -0.093 | exp/tri2/decode_dev/score_15/ctm.filt.filt.sys
#%WER 27.4 | 507 17792 | 78.3 16.6 5.1 5.7 27.4 96.4 | -0.145 | exp/tri3/decode_dev/score_16/ctm.filt.filt.sys
#%WER 32.0 | 507 17792 | 74.6 20.0 5.4 6.6 32.0 97.4 | -0.229 | exp/tri3/decode_dev.si/score_13/ctm.filt.filt.sys
#%WER 23.9 | 507 17792 | 81.2 14.8 4.0 5.0 23.9 93.9 | -0.108 | exp/tri3_mmi_b0.1/decode_dev_it4/score_13/ctm.filt.filt.sys
#%WER 23.3 | 507 17792 | 82.0 14.5 3.5 5.3 23.3 93.7 | -0.136 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.2_p0.15/score_13/ctm.filt.filt.sys
#%WER 23.3 | 507 17792 | 82.0 14.4 3.6 5.3 23.3 93.7 | -0.136 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.2_p0.18/score_13/ctm.filt.filt.sys
#%WER 23.3 | 507 17792 | 81.9 14.4 3.7 5.2 23.3 93.7 | -0.137 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.2_p0.20/score_13/ctm.filt.filt.sys
#%WER 23.2 | 507 17792 | 81.8 14.6 3.6 5.0 23.2 93.7 | -0.141 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.3_p0.15/score_15/ctm.filt.filt.sys
#%WER 23.2 | 507 17792 | 82.1 14.4 3.5 5.4 23.2 93.7 | -0.167 | exp/dur_model_tri3_mmi_b0.1/decode_dev_it4_s0.3_p0.18/score_13/ctm.filt.filt.sys
#%WER 23.2 | 507 17792 | 81.9 14.4 3.7 5.1 23.2 93.5 | -0.141 | exp/dur_model_t ri3_mmi_b0.1/decode_dev_it4_s0.3_p0.20/score_14/ctm.filt.filt.sys


#TEST SPEAKERS

#%WER 34.9 | 1155 27512 | 69.5 24.1 6.4 4.5 34.9 96.8 | 0.100 | exp/tri1/decode_test/score_13/ctm.filt.filt.sys
#%WER 29.9 | 1155 27512 | 74.6 20.2 5.3 4.5 29.9 95.3 | 0.016 | exp/tri2/decode_test/score_15/ctm.filt.filt.sys
#%WER 24.8 | 1155 27512 | 79.4 16.4 4.2 4.2 24.8 93.5 | -0.054 | exp/tri3/decode_test/score_16/ctm.filt.filt.sys
#%WER 29.9 | 1155 27512 | 74.6 20.3 5.2 4.5 29.9 95.2 | -0.115 | exp/tri3/decode_test.si/score_16/ctm.filt.filt.sys
#%WER 21.3 | 1155 27512 | 82.4 14.0 3.6 3.8 21.3 92.4 | 0.045 | exp/tri3_mmi_b0.1/decode_test_it4/score_13/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 83.0 13.7 3.3 3.8 20.8 91.8 | 0.024 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.2_p0.15/score_14/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 83.0 13.6 3.4 3.8 20.8 91.7 | 0.025 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.2_p0.18/score_14/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 83.0 13.6 3.4 3.7 20.8 91.8 | 0.021 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.2_p0.20/score_14/ctm.filt.filt.sys
#%WER 20.7 | 1155 27512 | 83.2 13.7 3.0 3.9 20.7 91.0 | 0.003 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.3_p0.15/score_14/ctm.filt.filt.sys
#%WER 20.7 | 1155 27512 | 83.1 13.7 3.2 3.8 20.7 91.3 | 0.006 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.3_p0.18/score_14/ctm.filt.filt.sys
#%WER 20.8 | 1155 27512 | 83.2 13.8 3.0 4.0 20.8 91.3 | -0.012 | exp/dur_model_tri3_mmi_b0.1/decode_test_it4_s0.3_p0.20/score_13/ctm.filt.filt.sys
