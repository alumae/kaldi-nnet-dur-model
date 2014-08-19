# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# Run locally:
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl

# EENet Grid:
export train_cmd="slurm.pl -p long"
export decode_cmd="slurm.pl -p long"
export cuda_cmd="slurm.pl -p long --constraint=K20 --gres=gpu:1 --mem=8g -x idu38"


