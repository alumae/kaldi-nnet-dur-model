#!/usr/bin/env python

import sys
import codecs
import argparse
from collections import OrderedDict
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import vector_spaces_dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace

from pylearn2.utils import serial

import sklearn.utils
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-features', action="store", dest="read_features_filename",
                        help="Read features from file")
    parser.add_argument('--write-features', action="store", dest="write_features_filename",
                        help="Read features from file")
    parser.add_argument('--devpercent', action="store", dest="devpercent", default=5, type=float,
                        help="Use the bottom N% from each source file as dev data")

    parser.add_argument('--save', action="store", dest="save_filename", help="Save data to file")
    parser.add_argument('--savedev', action="store", dest="savedev_filename", help="Save development data to file")
    parser.add_argument('--shuffle', action="store_true", help="Shuffle data before train/dev split")
    parser.add_argument('data', metavar='data-file features-file', nargs='+', help='Pairs of data and features files')

    args = parser.parse_args()

    if divmod(len(args.data), 2)[1] != 0:
        print >> sys.stderr, "Odd number of files given?"
        sys.exit(1)

    sources = [(args.data[i*2], args.data[i*2+1]) for i in range(len(args.data) / 2)]

    feature_dict = OrderedDict()
    if args.read_features_filename:
        print ".. Reading features from %s" % args.read_features_filename
        feature_dict = OrderedDict()
        for (i, feature) in enumerate(codecs.open(args.read_features_filename, encoding="UTF-8")):
            feature_dict[feature.strip()] = i
    else:
        print ".. Accumulating features from %s to %s" % (sources[0][1], sources[-1][1])
        for f in [s[1] for s in sources]:
            for (i, feature) in enumerate(codecs.open(f, encoding="UTF-8")):
                feature_dict.setdefault(feature.strip(), len(feature_dict))

    if args.write_features_filename:
        print ".. Writing features to %s" % args.write_features_filename
        with codecs.open(args.write_features_filename, 'w', encoding="UTF-8") as f:
            for feature in feature_dict:
                print >> f, feature


    X = np.zeros((0, len(feature_dict)), dtype=np.float16)
    speakers = np.zeros((0, 1), dtype=np.int)
    y = np.zeros((0, 1), dtype=np.float)


    for source in sources:
        print ".. Reading from data file %s with features from %s" % (source[0], source[1])
        source_feature_dict = OrderedDict()
        for (i, feature) in enumerate(codecs.open(source[1], encoding="UTF-8")):
            source_feature_dict[feature.strip()] = i

        dataset = serial.load(source[0])
        num_examples = dataset.get_num_examples()

        features_data = dataset.get_data()[0]
        speakers_data = dataset.get_data()[1]
        dur_data = dataset.get_data()[2]

        start = X.shape[0]
        X.resize((start + num_examples, len(feature_dict)))
        for (fname, fvalue) in source_feature_dict.iteritems():
            X[start:, feature_dict[fname]] = features_data[:, fvalue]

        speakers.resize((start + num_examples, 1))
        speakers[start:, :] = speakers_data[:, :]

        y.resize((start + num_examples, 1))
        y[start:, :] = dur_data[:, :]

    num_speakers = dataset.get_data_specs()[0].components[1].max_labels

    if args.shuffle:
        print ".. Shuffling data"
        X, speakers, y = sklearn.utils.shuffle(X, speakers, y)
        print ".. Done shuffling"

    num_dev = int(X.shape[0] * 0.01 * args.devpercent)
    print ".. Using %f percent of data (%d examples) as development data" % (args.devpercent, num_dev)

    if num_dev > 0:
        X_dev = X[-num_dev:]
        X = X[:-num_dev]
        speakers_dev = speakers[-num_dev:]
        speakers = speakers[:-num_dev]
        y_dev = y[-num_dev:]
        y = y[:-num_dev]

    space = CompositeSpace([VectorSpace(dim=len(feature_dict)),
                            IndexSpace(dim=1, max_labels=num_speakers),
                            VectorSpace(dim=1)])
    source = ('features', 'speakers', 'targets')
    final_dataset = vector_spaces_dataset.VectorSpacesDataset(
        data=(X, speakers, y),
        data_specs=(space, source))

    if args.save_filename:
        serial.save(args.save_filename, final_dataset)

    if args.savedev_filename:
        final_dataset_dev = vector_spaces_dataset.VectorSpacesDataset(
            data=(X_dev, speakers_dev, y_dev),
            data_specs=(space, source))

        serial.save(args.savedev_filename, final_dataset_dev)
