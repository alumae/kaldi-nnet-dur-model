#!/usr/bin/env python
"""
Adds duration model scores to lattices

@author: tanel
"""
import sys
import argparse
from collections import OrderedDict
import numpy as np
import theano
import cPickle
import codecs
import lattice
import durmodel_utils


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-features', action="store", dest="read_features_filename",
                        help="Read features from file")
    parser.add_argument('--duration-scale', action="store", dest="duration_scale", help="Weight of duration model",
                        type=float, default=0.10)
    parser.add_argument('--phone-penalty', action="store", dest="phone_penalty", help="Phone insertion penalty",
                        type=float, default=0.16)
    parser.add_argument('--output-extended-lat', action="store", dest="output_extended_lat", help="Output lattice with extra duration model scores",
                        type=bool, default=True)
    parser.add_argument('--language', action='store', dest='language', help="Language of the data", default="ESTONIAN")
    parser.add_argument('--fillers', action='store', dest='fillers', help="List of comma-seperated filler words", default="<sil>,++garbage++")
    parser.add_argument('--stress', action='store', dest='stress_dict_filename', help="Stress dictionary")
    parser.add_argument('--left-context', action='store', dest='left_context', help="Left context length", default=2, type=int)
    parser.add_argument('--right-context', action='store', dest='right_context', help="Left context length", default=2, type=int)
    parser.add_argument('--skip-fillers', action='store_true', dest='skip_fillers', help="Don't calculate posterior probabilities for fillers", default=True)
    parser.add_argument('--utt2spk', action='store', dest='utt2spk', help="Use the mapping in the given file to add speaker ID to each sample")
    parser.add_argument('--speakers', action='store', dest='speakers', help="List of speakers in training data, in sorted order")

    parser.add_argument('transitions', metavar='transitions.txt', help='Transition model, produced with show-transitions')
    parser.add_argument('nonsilence', metavar='nonsilence.txt', help='Nonsilence phonemes')
    parser.add_argument('words', metavar='words.txt', help='words.txt file')
    parser.add_argument('model', metavar='durmodel.pkl', help='duration model')

    args = parser.parse_args()
    durmodel_utils.LEFT_CONTEXT = args.left_context
    durmodel_utils.RIGHT_CONTEXT = args.right_context

    transitions = durmodel_utils.read_transitions(args.transitions)

    print >> sys.stderr, "Reading non-silence phonemes"
    nonsilence_phonemes = set()
    for l in open(args.nonsilence):
        nonsilence_phonemes.add(l.partition("_")[0])

    print >> sys.stderr, "Reading words.txt"
    word_list = []
    for l in codecs.open(args.words, encoding="UTF-8"):
        word_list.append(l.split()[0])

    filler_words = ["<eps>"] + args.fillers.split(",")
    print >> sys.stderr, "Fillers: ", filler_words

    stress_dict = None
    if args.stress_dict_filename:
        print >> sys.stderr, "Reading stress dictionary"
        stress_dict = durmodel_utils.load_stress_dict(args.stress_dict_filename)

    feature_dict = OrderedDict()
    if args.read_features_filename:
        print >> sys.stderr, ".. Reading features from %s" % args.read_features_filename
        feature_dict = OrderedDict()
        for (i, feature) in enumerate(codecs.open(args.read_features_filename, encoding="UTF-8")):
            feature_dict[feature.strip()] = i

    speaker_ids = None
    if args.speakers:
        print >> sys.stderr, ".. Reading speakers from %s" % args.speakers
        speaker_ids = {}
        for l in open(args.speakers):
            ss = l.split()
            speaker_ids.setdefault(ss[0], len(speaker_ids))

    utt2spkid = None
    if args.utt2spk:
        assert speaker_ids is not None
        print >> sys.stderr, "Reading utterance->speaker mapping from %s" % args.utt2spk
        utt2spkid = {}
        for l in open(args.utt2spk):
            ss = l.split()
            utt2spkid[ss[0]] = speaker_ids[ss[1]]


    model = cPickle.load(open(args.model))

    use_speaker_id=True
    sym_input = model.get_input_space().make_theano_batch()
    sym_Y = model.fprop(sym_input)
    try:
        model_fprop_function = theano.function(sym_input, sym_Y, on_unused_input='warn')
    except:
        use_speaker_id=False
        sym_input = theano.tensor.matrix(dtype=theano.config.floatX)
        model_fprop_function = theano.function([sym_input], model.fprop(sym_input))


    sym_y = theano.tensor.matrix(dtype=theano.config.floatX)
    sym_pdf_params = theano.tensor.matrix(dtype=theano.config.floatX)
    model_logprob_function = theano.function([sym_y, sym_pdf_params], model.layers[-1].logprob(sym_y, sym_pdf_params))

    lattice_lines = []
    for l in sys.stdin:
        l = l.strip()
        if len(l) == 0:
            lat = lattice.parse_aligned_lattice(lattice_lines)
            print >> sys.stderr, "Processing lattice %s" % lat.name
            features_and_durs = []
            for arc in lat.arcs:
                features_and_dur_seq = durmodel_utils.make_local(arc.start_frame, arc.word_id, arc.phone_ids, transitions, word_list,
                                                                 nonsilence_phonemes, language=args.language, stress_dict=stress_dict)
                features_and_durs.append(features_and_dur_seq)
            contexts = durmodel_utils.get_context_features_and_durs(lat, features_and_durs)
            assert len(contexts) == len(features_and_durs)
            assert len(contexts) == len(lat.arcs)
            i = 0
            for (context, local_feature_and_dur_seq) in zip(contexts, features_and_durs):
                #print >> sys.stderr, word_list[lat.arcs[i].word_id]
                if args.skip_fillers:
                    if word_list[lat.arcs[i].word_id] in filler_words:
                        i += 1
                        continue

                #print >> sys.stderr, "Processing word %s" % word_list[lat.arcs[i].word_id]
                full_word_features = durmodel_utils.compile_features_for_word(context, local_feature_and_dur_seq)
                num_phones = len(full_word_features)
                #print >> sys.stderr, lat.arcs[i].start_frame, lat.arcs[i].end_frame, word_list[lat.arcs[i].word_id].encode('utf-8')

                context_matrix = np.zeros((num_phones, len(feature_dict)), dtype=theano.config.floatX)
                if utt2spkid:
                    speaker_vector = np.ones((num_phones, 1), dtype=np.int) * utt2spkid[lat.name]
                else:
                    speaker_vector = np.zeros((num_phones, 1), dtype=np.int)

                y = np.zeros((num_phones, 1), dtype=theano.config.floatX)
                for (j, (phone_features, dur)) in enumerate(full_word_features):
                    #print >> sys.stderr, "  phone %d" % (j)
                    for (feature_name, value) in phone_features:
                        feature_id = feature_dict.get(feature_name, -1)
                        if feature_id >= 0:
                            context_matrix[j, feature_id] = value

                    y[j] = dur

                    #print  y
                #print context_matrix

                if use_speaker_id:
                    dist_params = model_fprop_function(context_matrix, speaker_vector)
                else:
                    dist_params = model_fprop_function(context_matrix)

                #mean = dist_params[:, 0]
                #sigma = np.exp(dist_params[:, 1])

                #n = np.exp(dist_params[:, 0]) # n > 0
                #p = sigmoid(dist_params[:, 1]) # p in (0,1)


                #prob_vector = np.exp(
                #    -(np.log(y[:, 0]) - log_mu_and_sigma[:, 0]) ** 2 / ( 2 * np.exp(log_mu_and_sigma[:, 1]) ** 2)) / \
                #              (y[:, 0] * np.exp(log_mu_and_sigma[:, 1]) * np.sqrt(2 * np.pi))
                #sigma_square = sigma ** 2
                #log_prob_vector = \
                #    (0.5 * (np.log(2 * np.pi * sigma_square) + 0.5 * ((y_target - mean) ** 2) / sigma_square)).sum(axis=1)

                log_prob_vector = model_logprob_function(y, dist_params)

                #print >> sys.stderr,prob_vector
                if not args.output_extended_lat:
                    total_word_dur_score = log_prob_vector.sum() * args.duration_scale - num_phones * np.log(args.phone_penalty) * args.duration_scale
                    #print >> sys.stderr, -total_word_dur_score
                    lat.arcs[i].score1 -= total_word_dur_score
                else:
                    lat.arcs[i].additional_score1 = log_prob_vector.sum()
                    lat.arcs[i].additional_score2 = num_phones
                i += 1
            if not args.output_extended_lat:
                lat.to_lat(sys.stdout)
            else:
                lat.to_extended_lat(sys.stdout)

            lattice_lines = []
        else:
            lattice_lines.append(l)
