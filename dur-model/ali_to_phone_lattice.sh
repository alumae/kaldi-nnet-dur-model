#!/bin/bash

cmd=run.pl
nj=4

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 4 ]; then
  echo "Usage: local/ali_to_phone_lattice.sh [options] <data-dir> <lang-dir> <ali-dir> <phone-lattice-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --use-segments (true|false)     # use segments and reco2file_and_channel files "
  echo "                                    # to produce a ctm relative to the original audio"
  echo "                                    # files, with channel information (typically needed"
  echo "                                    # for NIST scoring)."
  echo "e.g.:"
  echo "local/ali_to_phone_lattice.sh data/train data/lang exp/tri3a_ali exp/tri3a_phone_lat"
  echo "Produces aligned phone lattices in: exp/tri3a_phone_lat"
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


model=$alidir/final.mdl # assume model one level up from decoding dir.

for f in $lang/words.txt $lang/phones/word_boundary.int \
     $model $alidir/ali.1.gz $lang/oov.int; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;

mkdir -p $dir/log



$cmd JOB=1:$nj $dir/log/ali_to_phone_lattice.JOB.log \
   linear-to-nbest "ark:gunzip -c $alidir/ali.JOB.gz|" \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/text |" \
     '' '' ark:- \| \
    lattice-align-words $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
    gzip -c '>' $dir/ali-lat.JOB.gz || exit 1;
    

echo "$0: done getting phone alignments."
