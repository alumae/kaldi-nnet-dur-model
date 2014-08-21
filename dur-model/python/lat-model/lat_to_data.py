#!/usr/bin/env python

import sys
from collections import OrderedDict
import numpy as np
import argparse
import codecs
import gzip

from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

from durmodel_utils import read_transitions
import lattice
import durmodel_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-features', action="store", dest="read_features_filename",
                        help="Read features from file")
    parser.add_argument('--write-features', action="store", dest="write_features_filename",
                        help="Read features from file")
    parser.add_argument('--save', action="store", dest="save_filename", help="Save data to file")
    parser.add_argument('--language', action='store', dest='language', help="Language of the data", default="ESTONIAN")
    parser.add_argument('--stress', action='store', dest='stress_dict_filename', help="Stress dictionary")
    parser.add_argument('--left-context', action='store', dest='left_context', help="Left context length", default=2, type=int)
    parser.add_argument('--right-context', action='store', dest='right_context', help="Left context length", default=2, type=int)
    parser.add_argument('--no-duration-feature', action='store_true', dest='no_use_duration', help="Don't Use duration features")

    parser.add_argument('transitions', metavar='transitions.txt', help='Transition model, produced with show-transitions')
    parser.add_argument('nonsilence', metavar='nonsilence.txt', help='Nonsilence phonemes')
    parser.add_argument('words', metavar='words.txt', help='words.txt file')
    parser.add_argument('train_lattice', metavar='ali-lat.txt', help='Aligned phone lattice')

    args = parser.parse_args()
    durmodel_utils.LEFT_CONTEXT = args.left_context
    durmodel_utils.RIGHT_CONTEXT = args.right_context

    transitions = read_transitions(args.transitions)

    print >> sys.stderr, "DEBUG: transitions[%d] = %s" % (len(transitions) -2,  transitions[-2])
    print >> sys.stderr, "DEBUG: transitions[%d] = %s" % (len(transitions) -1,  transitions[-1])

    print >> sys.stderr, "Reading non-silence phonemes"
    nonsilence_phonemes = set()
    for l in open(args.nonsilence):
        nonsilence_phonemes.add(l.partition("_")[0])

    print >> sys.stderr, "Reading words.txt"
    word_list = []
    for l in codecs.open(args.words, encoding="UTF-8"):
        word_list.append(l.split()[0])

    stress_dict = None
    if args.stress_dict_filename:
        print >> sys.stderr, "Reading stress dictionary"
        stress_dict = durmodel_utils.load_stress_dict(args.stress_dict_filename)


    num_sentences_read = 0
    full_features_and_durs = []
    sentence_lines = []

    print >> sys.stderr, "Processing alignements..."
    for l in gzip.open(args.train_lattice):
        if len(l.strip()) > 0:
            sentence_lines.append(l)
        elif len(sentence_lines) > 0:
            try:
                lat = lattice.parse_aligned_lattice(sentence_lines)
                #print >> sys.stderr, "Processing lattice %s" % lat.name
                features_and_durs = []
                for arc in lat.arcs:
                    features_and_dur_seq = durmodel_utils.make_local(arc.start_frame, arc.word_id, arc.phone_ids, transitions, word_list, nonsilence_phonemes,
                                                               language=args.language, stress_dict=stress_dict)
                    #print features_and_dur_seq
                    features_and_durs.append(features_and_dur_seq)
                full_features_and_durs.extend(durmodel_utils.make_linear(features_and_durs, nonsilence_phonemes))
            except IOError as e:
                print >> sys.stderr, "I/O error({0}): {1} -- {2} when processing lattice {3}".format(e.errno, e.strerror, e.message,  lat.name)
            num_sentences_read += 1
            sentence_lines = []
    print >> sys.stderr, "Read alignments for %d utterances" % num_sentences_read

    feature_dict = OrderedDict()
    if args.read_features_filename:
        print ".. Reading features from %s" % args.read_features_filename
        feature_dict = OrderedDict()
        for (i, feature) in enumerate(codecs.open(args.read_features_filename, encoding="UTF-8")):
            feature_dict[feature.strip()] = i

    else:
        for (_features, d) in full_features_and_durs:
            for f in _features:
                feature_name = f[0]
                feature_dict.setdefault(feature_name, len(feature_dict))
                #print feature_dict

    if args.write_features_filename:
        print ".. Writing features to %s" % args.write_features_filename
        with codecs.open(args.write_features_filename, 'w', encoding="UTF-8") as f:
            for feature in feature_dict:
                print >> f, feature

    num_items = len(full_features_and_durs)
    context_matrix = np.zeros((num_items, len(feature_dict)), dtype=np.float16)
    print ".. Created matrix of shape ", context_matrix.shape, " and size ", context_matrix.size

    y = np.zeros((num_items, 2), dtype=np.float)
    print ".. Created outcome matrix of shape", y.shape, " and size ", y.size
    for (i, (_features, dur)) in enumerate(full_features_and_durs):
        for feature in _features:
            (feature_name, value) = feature
            feature_id = feature_dict.get(feature_name, -1)
            if feature_id >= 0:
                context_matrix[i, feature_id] = value

        y[i, 0] = dur

    #print context_matrix

    ds = dense_design_matrix.DenseDesignMatrix(X=context_matrix, y=y)

    if args.save_filename:
        serial.save(args.save_filename, ds)
