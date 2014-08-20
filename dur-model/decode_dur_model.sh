#!/bin/bash

cmd=run.pl
cuda_cmd=run.pl
nj=4
skip_scoring=false
scoring_opts=
language=ENGLISH
stress_dict=
pylearn_dir=~/tools/pylearn2

scales="0.3 0.4 0.5"
penalties="0.1 0.15 0.20"

fillers="<sil>"

score_cmd=./local/score.sh
stage=0

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 6 ]; then
  echo "Usage: local/decode_dur_model.sh [options] <lang-dir> <graph-dir> <data-dir> <src-decode-dir> <durmodel_dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --cuda-cmd                      # specify how to run the neural network command (done using Theano/Pylearn)."
  echo "    --stage (0|1|2)                 # start training script from part-way through."
  echo "    --language ENGLISH|ESTONIAN|FINNISH  # language of the data."
  echo "    --fillers filler1,filler2       # comma-seperated list of filler words."
  echo "    --stress-dict file              # Dictionary with lexical stress "    
  echo
  echo "e.g.:"
  echo "dur-model/decode_dur_model.sh data/lang exp/tri3/graph data/train data/lang exp/tri3a_ali exp/durmodel_tri3a"
  echo "Produces duration model in in: exp/durmodel_tri3a"
  exit 1;
fi

langdir=$1
graphdir=$2
data=$3 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
src_decode_dir=$4
dur_model_dir=$5
decode_dir=$6



if ! [ $nj -eq `cat $src_decode_dir/num_jobs` ]; then
    echo "$0: Number of jobs mismatch with decode dir: $nj versus `cat $src_decode_dir/num_jobs`";
    exit 1;
fi


# Convert decoding lattices to word-aligned lattices
if [ $stage -le 0 ]; then
  echo "$0: Converting decoding lattices to word-aligned lattices..."
  mkdir -p $dur_model_dir/decode/log
  $cmd JOB=1:$nj $decode_dir/log/lattice-to-phone-lattice.JOB.log \
    lattice-align-words $graphdir/phones/word_boundary.int \
      $src_decode_dir/../final.mdl "ark:gunzip -c $src_decode_dir/lat.JOB.gz\|" ark,t:- \| \
     gzip -c \> $decode_dir/ali_lat.JOB.gz || exit 1
fi


if [ -n "$stress_dict" ]; then
  stress_arg="--stress $stress_dict";
fi

# Add duration model scores to decoding lattices, resulting in "extended lattices"
if [ $stage -le 1 ]; then
  echo "$0: Add duration model scores to decoding lattices, resulting in 'extended lattices'"
  mkdir -p $decode_dir/log
  $cuda_cmd JOB=1:$nj $decode_dir/log/process_lattice.JOB.log \
    set -o pipefail \; \
    zcat $decode_dir/ali_lat.JOB.gz \| \
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 PYTHONPATH=dur-model/python/pylearn2/ \
      ./dur-model/python/lat-model/process_lattice.py \
        --left-context 3 \
        --right-context 3 \
        --read-features $dur_model_dir/ali-lat.features \
        --output-extended-lat true \
        --language $language \
        --fillers "$fillers" \
        $stress_arg \
        $dur_model_dir/transitions.txt $langdir/phones/nonsilence.txt $graphdir/words.txt \
        $dur_model_dir/durmodel_best.pkl \| \
     gzip -c \> $decode_dir/ali_lat_extended.JOB.gz || exit 1
fi

# Combine duration model scores with the 'graph' score
if [ $stage -le 2 ]; then
  for scale in $scales; do
    for penalty in $penalties; do
      echo "$0: Using scale $scale and phone penalty $penalty to rescore the lattices"
      extended_decode_dir=${decode_dir}_s${scale}_p${penalty};
      mkdir -p $extended_decode_dir;      
      cp ${src_decode_dir}/../final.mdl ${extended_decode_dir}/../final.mdl
      cp ${src_decode_dir}/num_jobs ${extended_decode_dir}/num_jobs
      $cmd JOB=1:$nj $extended_decode_dir/log/extended_lat_to_lat.JOB.log \
        set -o pipefail \; \
        zcat $decode_dir/ali_lat_extended.JOB.gz \| \
        dur-model/python/lat-model/extended_lat_to_lat.py $scale $penalty \| \
        gzip -c \> $extended_decode_dir/lat.JOB.gz || exit 1
        
      if ! $skip_scoring ; then
        if [ ! -x $score_cmd ]; then
          echo "Not scoring because $score_cmd does not exist or not executable."
        else
          echo "$0: Scoring..."
          $score_cmd --cmd "$cmd" $scoring_opts $data $graphdir $extended_decode_dir || exit 1      
        fi
      fi
     
    done;
  done
fi

