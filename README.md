INTRODUCTION
============

Kaldi implementation of a neural network phone duration model, as described in
the paper:

Tanel Alumäe. Neural network phone duration model for speech recognition. 
Interspeech 2014, Singapore.
https://phon.ioc.ee/dokuwiki/lib/exe/fetch.php?media=people:tanel:icassp2014-durmodel.pdf

The duration model has been tested on English, Estonian and Finnish.

We provide a [recipe for the TEDLIUM dataset](run_tedlium.sh). The baseline system uses online 
multisplice speed-perturbed DNN models, with rescoring of the lattices with large language model from Cantab Research. Duration model decreases 
WER from 12.7% to 12.1% for the development set and from 11.7% to 11.0% for the test set.


DEPENDENCIES
============

  * Python 2.6 (with argparse) or 2.7
  * Theano
  * Pylearn2
  
Theano can be installed using python's `pip` utility:

    pip install Theano --user

This installs Theano locally (not systemwide). More instructions: 
http://deeplearning.net/software/theano/install.html

Pylearn2 should be cloned from Github, see 
http://deeplearning.net/software/pylearn2/#download-and-installation
  
  
USAGE
=====

See [run_tedlium.sh](run_tedlium.sh) for a sample script that trains a duration model
on TEDLIUM data, on top of already trained DNN models. 

Duration model is trained using Pylearn2 that itself uses Theano. Theano
can use GPU which makes the training much faster (takes about 1 hour on
TEDLIUM data, when using a Tesla K20). You should use the ~/.theanorc 
file to instruct Theano to use the GPU:

    [global]
    device = gpu 
    floatX = float32

If you use a cluster, you should also instruct Theano to use a 
machine-local temporary directory for its compilation directory. Set the
following line in the [global] section of the .theanorc file:

    base_compiledir=/tmp/%(user)s/theano.NOBACKUP


ADDING A NEW LANGUAGE
=====================

All language specific details are defined in `dur-model/python/lat-model/data/languages.yaml`.

CITING
======

You can cite the following paper if you use this software:

    @InProceedings{alumae2014,
      author={Alum\"{a}e, Tanel},
      title={Neural network phone duration model for speech recognition},
      booktitle={Interspeech 2014},
      address={Singapore},
      year=2014
    }
