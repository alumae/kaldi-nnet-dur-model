#!/bin/bash
#
# Copyright 2014 Tanel Alumäe
# License: BSD 3-clause
#
# Example of how to train a duration model on swahili data, using
# it's exp/system1/tri3b_mmi_b0.1 as a baseline model. 
# You must train the baseline model and run the corresponding decoding
# experiments prior to running this script.
#
# Also, create a symlink: ln -s <kaldi-nnet-dur-model-dir>/dur-model
#
# Then run the script from the $KALDI_ROOT/egs/swahili/s5 directory.
#
#

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


nj=20            # Must be the same as when training the baseline model
decode_nj=4      # Must be the same as when decoding using the baseline 

left_context=4
right_context=2

h0_dim=200
h1_dim=200
speaker_projection_dim=100
utterance_projection_dim=10

stage=0 # resume training with --stage=N
pylearn_dir=~/tools/pylearn2
# Aggregating traing data for duration model needs more RAM than default in our SLURM
aggregate_data_args="--mem=8g"


. utils/parse_options.sh || exit 1;


if [ $stage -le 0 ]; then
  # Align training data using a model in exp/tri3_mmi_b0.1
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/system1/tri3b_mmi_b0.1 exp/system1/tri3b_mmi_b0.1_ali || exit 1
fi



if [ $stage -le 1 ]; then
  # Train a duration model based on alignments in exp/system1/tri3b_mmi_b0.1_ali
  ./dur-model/train_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
    --stage 0 \
    --language SWAHILI \
    --h0-dim $h0_dim --h1-dim $h1_dim \
    --left-context $left_context --right-context $right_context \
    data/train data/lang exp/system1/tri3b_mmi_b0.1_ali exp/system1/dur_model_tri3b_mmi_b0.1 || exit 1
fi


#0.07 0.09 0.11 0.13 0.15 0.17 0.19
if [ $stage -le 2 ]; then
  # Rescore test lattices using the duration model
  ./dur-model/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
    --language SWAHILI --fillers "!SIL,\<UNK\>,\<laughter\>,\<music\>" \
    --left-context $left_context --right-context $right_context \
    --scales "0.2 0.3 0.4" --penalties "0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.19" \
    --stage 0 \
    --ignore_speakers true \
    data/lang \
    exp/system1/tri3b/graph \
    data/test \
    exp/system1/tri3b_mmi_b0.1/decode_test \
    exp/system1/dur_model_tri3b_mmi_b0.1 \
    exp/system1/dur_model_tri3b_mmi_b0.1/decode_test || exit 1
fi

if [ $stage -le 3 ]; then
  # Train a SAT duration model based on alignments in exp/system1/tri3b_mmi_b0.1_ali
  ./dur-model/train_sat_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
    --stage 0 \
    --language SWAHILI \
    --h0-dim $h0_dim --h1-dim $h1_dim --speaker-projection-dim $speaker_projection_dim \
    --left-context $left_context --right-context $right_context \
    data/train data/lang exp/system1/tri3b_mmi_b0.1_ali exp/system1/dur_model_sat_tri3b_mmi_b0.1 || exit 1
fi


if [ $stage -le 4 ]; then
  for set in test; do 
    # Create a new data directory from hypotheses
    echo "Creating fake data directory for $set data from baseline hypotheses"
    mkdir -p exp/system1/tri3b_mmi_b0.1/decode_test/data
    cp -r data/${set}/* exp/system1/tri3b_mmi_b0.1/decode_${set}/data
    rm -rf exp/system1/tri3b_mmi_b0.1/decode_${set}/data/split*
    cat exp/system1/tri3b_mmi_b0.1/decode_${set}/scoring/12.tra | int2sym.pl -f 2- data/lang/words.txt > exp/system1/tri3b_mmi_b0.1/decode_${set}/data/text
  done
fi


if [ $stage -le 5 ]; then
  for set in test; do 
    # Align (fake) training data using a model in exp/tri3_mmi_b0.1
    echo "Aligning $set data"
    steps/align_fmllr.sh --nj $decode_nj --cmd "$train_cmd" \
        exp/system1/tri3b_mmi_b0.1/decode_${set}/data data/lang exp/system1/tri3b_mmi_b0.1 exp/system1/tri3b_mmi_b0.1_${set}_ali || exit 1
  done
fi


if [ $stage -le 6 ]; then
  for set in test; do 
    echo "Adapting SAT duration model to $set speakers"
    ./dur-model/adapt_dur_model_sat.sh --nj $decode_nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
      --stage 0 \
      --language SWAHILI \
      --h0-dim $h0_dim --h1-dim $h1_dim --speaker-projection-dim $speaker_projection_dim \
      --left-context $left_context --right-context $right_context \
      exp/system1/tri3b_mmi_b0.1/decode_${set}/data data/lang exp/system1/tri3b_mmi_b0.1_${set}_ali \
      exp/system1/dur_model_sat_tri3b_mmi_b0.1 exp/system1/dur_model_sat_tri3b_mmi_b0.1_${set} || exit 1
  done
fi

#0.07 0.09 0.11 0.13 0.15 0.17 0.19
if [ $stage -le 7 ]; then
  for set in test; do 
    echo "Decoding using SAT duration model adapted to $set speakers"
    # Rescore lattices using the adapted SAT duration model
    ./dur-model/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
      --language SWAHILI --fillers "!SIL,\<UNK\>,\<laughter\>,\<music\>"  \
      --left-context $left_context --right-context $right_context \
      --scales "0.2 0.3 0.4" --penalties "0.07 0.09 0.11 0.13 0.15 0.17 0.19" \
      --stage 0 \
      data/lang \
      exp/system1/tri3b/graph \
      data/${set} \
      exp/system1/tri3b_mmi_b0.1/decode_${set} \
      exp/system1/dur_model_sat_tri3b_mmi_b0.1_${set} \
      exp/system1/dur_model_sat_tri3b_mmi_b0.1_${set}/decode_${set} || exit 1
    done
fi




if [ $stage -le 8 ]; then
  echo "Train a utterance-adapted duration model based on alignments in exp/system1/tri3b_mmi_b0.1_ali"
  ./dur-model/train_sat_dur_model.sh --nj $nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
    --stage 2 \
    --language SWAHILI \
    --h0-dim $h0_dim --h1-dim $h1_dim --speaker-projection-dim $utterance_projection_dim \
    --per-utt true \
    --left-context $left_context --right-context $right_context \
    data/train data/lang exp/system1/tri3b_mmi_b0.1_ali exp/system1/dur_model_sat_per_utt_tri3b_mmi_b0.1 || exit 1
fi

if [ $stage -le 9 ]; then
  for set in test; do 
    echo "Adapting per-utterance SAT duration model to $set utterances"
    ./dur-model/adapt_dur_model_sat.sh --nj $decode_nj --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --pylearn-dir $pylearn_dir --aggregate-data-args "$aggregate_data_args" \
      --stage 0 \
      --language SWAHILI \
      --h0-dim $h0_dim --h1-dim $h1_dim --speaker-projection-dim $utterance_projection_dim \
      --per-utt true \
      --left-context $left_context --right-context $right_context \
      exp/system1/tri3b_mmi_b0.1/decode_${set}/data data/lang exp/system1/tri3b_mmi_b0.1_${set}_ali \
      exp/system1/dur_model_sat_per_utt_tri3b_mmi_b0.1 exp/system1/dur_model_sat_per_utt_tri3b_mmi_b0.1_${set} || exit 1
  done
fi

#0.07 0.09 0.11 0.13 0.15 0.17 0.19
if [ $stage -le 10 ]; then
  for set in test; do 
    echo "Decoding using per-utterance SAT duration model adapted to $set utterances"
    # Rescore lattices using the adapted SAT duration model
    ./dur-model/decode_dur_model.sh --cmd "$train_cmd" --cuda-cmd "$cuda_cmd" --nj $decode_nj \
      --language SWAHILI --fillers "!SIL,\<UNK\>,\<laughter\>,\<music\>"  \
      --left-context $left_context --right-context $right_context \
      --scales "0.2 0.3 0.4" --penalties "0.07 0.09 0.11 0.13 0.15 0.17 0.19" \
      --stage 0 \
      --per-utt true \
      data/lang \
      exp/system1/tri3b/graph \
      data/${set} \
      exp/system1/tri3b_mmi_b0.1/decode_${set} \
      exp/system1/dur_model_sat_per_utt_tri3b_mmi_b0.1_${set} \
      exp/system1/dur_model_sat_per_utt_tri3b_mmi_b0.1_${set}/decode_${set} || exit 1
    done
fi
