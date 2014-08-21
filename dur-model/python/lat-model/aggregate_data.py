#!/usr/bin/env python

import sys
import codecs
import argparse
from collections import OrderedDict
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial
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
    y = np.zeros((0, 2), dtype=np.float)

    X_dev = np.zeros((0, len(feature_dict)), dtype=np.float16)
    y_dev = np.zeros((0, 2), dtype=np.float)

    print ".. Using %f percent of data as development data" % args.devpercent

    for source in sources:
        print ".. Reading from data file %s with features from %s" % (source[0], source[1])
        source_feature_dict = OrderedDict()
        for (i, feature) in enumerate(codecs.open(source[1], encoding="UTF-8")):
            source_feature_dict[feature.strip()] = i

        ddm = serial.load(source[0])
        num_dev = int(0.01 * args.devpercent * ddm.X.shape[0])
        start = X.shape[0]
        X.resize((start + ddm.X.shape[0] - num_dev, len(feature_dict)))
        for (fname, fvalue) in source_feature_dict.iteritems():
            X[start:, feature_dict[fname]] = ddm.X[0:-num_dev, fvalue]

        y.resize((start + ddm.y.shape[0] - num_dev, 2))
        y[start:, :] = ddm.y[0:-num_dev, :]

        start = X_dev.shape[0]
        X_dev.resize((start + num_dev, len(feature_dict)))
        for (fname, fvalue) in source_feature_dict.iteritems():
            X_dev[start:, feature_dict[fname]] = ddm.X[-num_dev:, fvalue]

        y_dev.resize((start + num_dev, 2))
        y_dev[start:, :] = ddm.y[-num_dev:]


    ddm = dense_design_matrix.DenseDesignMatrix(X=X, y=y)
    if args.save_filename:
        serial.save(args.save_filename, ddm)

    ddm_dev = dense_design_matrix.DenseDesignMatrix(X=X_dev, y=y_dev)
    if args.savedev_filename:
        serial.save(args.savedev_filename, ddm_dev)
