#!/bin/bash



stage=0
cmd=run.pl
cuda_cmd=run.pl
nj=4

language=ENGLISH
stress_dict=
pylearn_dir=~/tools/pylearn2
aggregate_data_args=

left_context=3
right_context=3

h0_dim=300
h1_dim=300

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 4 ]; then
  echo "Usage: local/train_dur_model.sh [options] <data-dir> <lang-dir> <ali-dir> <durmodel_dir> "
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --cuda-cmd                      # specify how to run the neural network training (done using Theano/Pylearn)."
  echo "    --stage (0|1|2)                 # start training script from part-way through."
  echo "    --pylearn-dir (0|1|2)           # base directory of Pylearn2."  
  echo "    --language ENGLISH|ESTONIAN|FINNISH  # language of the data."
  echo "    --stress-dict file              # Dictionary with lexical stress "  
  echo
  echo "e.g.:"
  echo "local/train_dur_model.sh data/train data/lang exp/tri3a_ali exp/durmodel_tri3a"
  echo "Produces duration model in in: exp/durmodel_tri3a"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
alidir=$3
dir=$4



if ! [ $nj -eq `cat $alidir/num_jobs` ]; then
  echo "$0: Number of jobs mismatch with alignments dir: $nj versus `cat $alidir/num_jobs`";
  exit 1;
fi

# Create lattices from aligned training data
if [ $stage -le 1 ]; then
  dur-model/ali_to_phone_lattice.sh --nj $nj --cmd "$cmd" \
    $data $lang $alidir ${alidir}_phone_lat || exit 1;
fi


if [ -n "$stress_dict" ]; then
  stress_arg="--stress $stress_dict";
fi


# Extract training data
if [ $stage -le 2 ]; then
  echo "Converting phone-aligned training data to duration model training data"
  mkdir -p $dir
  # Save transition model in text form
  show-transitions $lang/phones.txt $alidir/final.mdl > $dir/transitions.txt || exit 1;

  $cmd JOB=1:$nj $dir/log/lat_to_data.JOB.log \
    dur-model/python/lat-model/lat_to_data.py \
      --left-context $left_context \
      --right-context $right_context \
      --language $language \
      --utt2spk $data/utt2spk \
      $stress_arg \
      --write-features $dir/ali-lat.JOB.features \
      --save $dir/ali-lat.JOB.pkl.joblib \
      $dir/transitions.txt $lang/phones/nonsilence.txt $lang/words.txt ${alidir}_phone_lat/ali-lat.JOB.gz || exit 1;
      
fi

# Accumulate training data to a single file
if [ $stage -le 3 ]; then
  echo "Aggregating duration model training data to a single dataset"
  $cmd $aggregate_data_args $dir/log/aggregate_data.log \
  for i in `seq 1 $nj`\; do \
    echo $dir/ali-lat.\$i.pkl.joblib $dir/ali-lat.\$i.features\; done \| \
    xargs dur-model/python/lat-model/aggregate_data.py --shuffle --save $dir/ali-lat.pkl.joblib --savedev $dir/ali-lat_dev.pkl.joblib --write-features $dir/ali-lat.features || exit 1;
  rm $dir/ali-lat.*.pkl.joblib $dir/ali-lat.*.features
fi




# Train a non-SAT model
if [ $stage -le 5 ]; then
  echo "Training duration model"
  cat dur-model/durmodel_template.yaml | \
  MODEL_SAVE_PATH=$dir/durmodel_best.pkl \
  TRAIN_PKL=$dir/ali-lat.pkl.joblib \
  DEV_PKL=$dir/ali-lat_dev.pkl.joblib \
  INPUT_DIM=`wc -l $dir/ali-lat.features | awk '{print $1}'` \
  NUM_SPEAKERS=`wc -l $data/spk2utt | awk '{print $1}'` \
  H0_DIM=$h0_dim \
  H1_DIM=$h1_dim \
  envsubst > $dir/durmodel.yaml
  $cuda_cmd $dir/log/train.log \
    PYTHONPATH=dur-model/python/pylearn2/ \
    $pylearn_dir/pylearn2/scripts/train.py $dir/durmodel.yaml || exit 1;
 

fi
