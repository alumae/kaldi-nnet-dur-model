# -*- coding: UTF-8 -*-
'''
Created on Sep 9, 2013

@author: tanel
'''
import sys
from itertools import groupby
import math
import re
import syllabifier

phoneme_classes = {"Vowel": u"a e i o u ou ae oe ue".split(),
                   "Consonant": u"f h j k l m n p r s sh t v".split(),
                   "Stop": u"p k t".split(),
                   "Labiodental": u"f v".split(),
                   "fricative": u"f v s sh h".split(),
                   "Nasal": u"m n".split(),
                   "Liquid": u"l r".split(),
                   "Sibil": u"sh s".split()}

extended_phoneme_classes = {}
all_short_phonemes = set()
for (klass, _phonemes) in phoneme_classes.iteritems():
    extended_phoneme_classes[klass] = _phonemes + [phoneme * 2 for phoneme in _phonemes]
    #extended_phoneme_classes[klass + "_short"] = phonemes
    #extended_phoneme_classes[klass + "_long"] = [phoneme * 2 for phoneme in phonemes]
    for phoneme in _phonemes:
        all_short_phonemes.add(phoneme)

#extended_phoneme_classes["Short"] = all_short_phonemes
extended_phoneme_classes["Long"] = [p * 2 for p in all_short_phonemes]

en_phoneme_classes = {
    'consonant': ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
                  'W', 'Y', 'Z', 'ZH'],
    'vowel': ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'],
    'stop': ['P', 'PD', 'B', 'T', 'D', 'K', 'G', 'CH', 'JH'],
    'nasal': ['M', 'N', 'NG'],
    'fricative': ['S', 'SH', 'Z', 'F', 'V', 'TH', 'DH', 'ZH', 'HH'],
    #'diphthong' : ['AY', 'EY', 'OW', 'OY', 'UW', 'AW'],
    'semivowel': ['L', 'R', 'ER', 'W', 'Y']
}


## Finnish

fi_phoneme_classes = {"Vowel": u"a e i o u ä ö ü".split(),
                      "Consonant": u"f h j k l m n p r s t v q b d g".split(),
                      "Stop1": u"b g d".split(),
                      "Stop2": u"p k t q".split(),
                      "Labiodental": u"f v".split(),
                      "fricative": u"f v s h".split(),
                      "Nasal": u"m n".split(),
                      "Liquid": u"l r".split()}

fi_extended_phoneme_classes = {}
fi_all_short_phonemes = set()
for (klass, phonemes) in fi_phoneme_classes.iteritems():
    fi_extended_phoneme_classes[klass] = phonemes + [phoneme * 2 for phoneme in phonemes]
    for phoneme in phonemes:
        fi_all_short_phonemes.add(phoneme)

fi_extended_phoneme_classes["Long"] = [p * 2 for p in fi_all_short_phonemes]



LANGUAGES = {
    "ESTONIAN": {
        "phoneme_classes": extended_phoneme_classes,
        "syllabifier_conf": syllabifier.Estonian
    },
    "ENGLISH": {
        "phoneme_classes": en_phoneme_classes,
        "syllabifier_conf": syllabifier.English
    },
    "FINNISH": {
        "phoneme_classes": fi_phoneme_classes,
        "syllabifier_conf": None
    }

}

LEFT_CONTEXT = 2
RIGHT_CONTEXT = 2

USE_DURATION_FEATURE = True

SKIP_FILLERS = True


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dur_function(dur):
    return (sigmoid(dur * 0.1) - 0.5) * 2.0


def encode(alist):
    return [(key, len(list(group))) for key, group in groupby(alist)]

def syllabify(phonemes, language, nonsilence_phonemes):
    syllabifier_conf = LANGUAGES[language]['syllabifier_conf']
    if syllabifier_conf is None:
        return [[0] * len(phonemes)]
    if len(phonemes) == 1 and phonemes[0] not in nonsilence_phonemes:
        return phonemes

    syllables = syllabifier.syllabify(syllabifier_conf, phonemes)
    return [s[1] + s[2] + s[3] for s in syllables]


def load_stress_dict(filename):
    result = {}
    for l in open(filename):
        if l.startswith(";;;"):
            continue
        ss = l.split()
        word = ss[0]
        word = re.sub("(\d)$", "", word)
        phonemes = [re.sub("\d", "", x).upper() for x in ss[1:]]
        stress = [0] * len(phonemes)
        for i, x in enumerate(ss[1:]):
            if x.endswith("1"):
                stress[i] = 1
            if x.endswith("2"):
                stress[i] = 2

        result.setdefault(word, []).append((phonemes, stress))
    return result


def get_stress(word, phonemes, stress_dict):
    prons = stress_dict.get(word, [])
    for pron in prons:
        if pron[0] == phonemes:
            return pron[1]
    return [0] * len(phonemes)


def phone_runlengths_from_frames(frames, transitions):
    phone_runlengths = []
    n = 0
    last_phone = ""
    for f in frames:
        # if phone != last_phone
        # note that it's not correct if there are identical phonemes after each other
        # but for some reason relying on transitions to final states fails sometimes
        if n > 0 and transitions[f][0] != last_phone:
            phone_runlengths.append((last_phone.partition("_")[0], n))
            n = 1
        else:
            n += 1
        last_phone = transitions[f][0]
    phone_runlengths.append((last_phone.partition("_")[0], n))
    return phone_runlengths

def make_local(start_frame, word_id, frames, transitions, word_list, nonsilence_phonemes, language="ESTONIAN", stress_dict=None):

    phone_rl_names = phone_runlengths_from_frames(frames, transitions)

    #print >> sys.stderr, phone_rl_names

    word = word_list[word_id]

    features_and_dur_seq = []
    syllables = syllabify([p[0] for p in phone_rl_names], language, nonsilence_phonemes)
    syllable_ids = []
    i = 1
    for s in syllables:
        for p in s:
            syllable_ids.append(i)
        i += 1
    i = 0
    current_start_frame = start_frame
    for (phone, dur) in phone_rl_names:
        features = []
        features.append(("%s" % phone, 1))
        for (kl, phonemes) in LANGUAGES[language]["phoneme_classes"].iteritems():
            if phone in phonemes:
                features.append((kl, 1))
        features.append(("syllable", syllable_ids[i]))

        features_and_dur_seq.append((features, dur))

        i += 1
        current_start_frame += dur

    features_and_dur_seq[0][0].append(("word_initial", 1))
    features_and_dur_seq[-1][0].append(("word_final", 1))
    if stress_dict:
        stress = get_stress(word, [p[0].upper() for p in phone_rl_names], stress_dict)
        for i, s in enumerate(stress):
            if s > 0:
                features_and_dur_seq[i][0].append(("stress%d" % s, 1))

    return features_and_dur_seq


def make_linear(feature_and_dur_seqs, nonsilence_phonemes):
    full_feature_seq = []
    local_feature_seq = []
    for feature_and_dur_seq in feature_and_dur_seqs:
        for (feature_set, dur) in feature_and_dur_seq:
            local_feature_seq.append((feature_set, dur))
    i = 0
    for feature_and_dur_seq in feature_and_dur_seqs:
        for (feature_list, dur) in feature_and_dur_seq:
            is_filler = True
            tmp_feature_set = set(feature_list)
            #print >> sys.stderr, tmp_feature_set
            if SKIP_FILLERS:
                for phoneme in nonsilence_phonemes:
                    if (phoneme, 1) in tmp_feature_set:
                        is_filler = False
                        break
                if is_filler:
                    i += 1
                    continue
            full_feature_list = []
            full_feature_list.extend(feature_list)
            for j in range(1, LEFT_CONTEXT + 1):
                if i - j >= 0:
                    full_feature_list.extend(
                        [("pos-%d:%s" % (j, s), value) for (s, value) in local_feature_seq[i - j][0] if not s.startswith("_")])
                    if USE_DURATION_FEATURE:
                        full_feature_list.append(("pos-%d:dur" % j, dur_function(local_feature_seq[i - j][1])))
                else:
                    full_feature_list.append(("pos-%d:<s>" % j, 1))
                    if USE_DURATION_FEATURE:
                        full_feature_list.append(("pos-%d:dur" % j, dur_function(10)))

            for j in range(1, RIGHT_CONTEXT + 1):
                if i + j < len(local_feature_seq):
                    full_feature_list.extend(
                        [("pos+%d:%s" % (j, s), value) for (s, value) in local_feature_seq[i + j][0] if not s.startswith("_")])
                else:
                    full_feature_list.append(("pos+%d:</s>" % j, 1))

            full_feature_seq.append((full_feature_list, dur))
            i += 1

    return full_feature_seq


def get_context_features_and_durs(lattice, feature_and_dur_seqs):
    contexts = []

    for i, arc in enumerate(lattice.arcs):
        #print "--- processing arc", arc
        contexts_map = {}
        prev_arcs = lattice.get_previous_arcs(arc)
        if len(prev_arcs) > 0:
            prev_arc = prev_arcs[0]
            #print "prev_arc: ", prev_arc
            prev_arc_id = prev_arc.id
            index_of_prev_phone = -1
            for j in range(1, LEFT_CONTEXT + 1):
                #print "finding context", -j
                contexts_map[-j] = feature_and_dur_seqs[prev_arc_id][index_of_prev_phone][0] +\
                                   [("dur", dur_function(feature_and_dur_seqs[prev_arc_id][index_of_prev_phone][1]))]

                index_of_prev_phone -= 1
                if index_of_prev_phone < -len(feature_and_dur_seqs[prev_arc_id]):
                    prev_arcs = lattice.get_previous_arcs(prev_arc)
                    if len(prev_arcs) > 0:
                        prev_arc = prev_arcs[0]
                        prev_arc_id = prev_arc.id
                        #print "new prev arc:", prev_arc
                        index_of_prev_phone = -1
                    else:
                        contexts_map[-j - 1] = [("<s>", 1), ("dur", dur_function(10))]
                        break
        else:
            for j in range(1, LEFT_CONTEXT + 1):
                contexts_map[-j] = [("<s>", 1),
                                    ("dur", dur_function(10))]


        next_arcs = lattice.get_next_arcs(arc)
        if len(next_arcs) > 0:
            next_arc = next_arcs[0]
            #print "next_arc: ", next_arc
            next_arc_id = next_arc.id
            index_of_next_phone = 0
            for j in range(1, RIGHT_CONTEXT + 1):
                #print "finding context", j
                contexts_map[j] = feature_and_dur_seqs[next_arc_id][index_of_next_phone][0]
                index_of_next_phone += 1
                if index_of_next_phone >= len(feature_and_dur_seqs[next_arc_id]):
                    next_arcs = lattice.get_next_arcs(next_arc)
                    if len(next_arcs) > 0:
                        next_arc = next_arcs[0]
                        next_arc_id = next_arc.id
                        #print "new next arc:", next_arc
                        index_of_next_phone = 0
                    else:
                        contexts_map[j + 1] = [("</s>", 1)]
                        break
        else:
            contexts_map[1] = [("</s>", 1)]

        contexts.append(contexts_map)
    return contexts


def compile_features_for_word(context, local_feature_seq):
    full_feature_seq = []
    i = 0
    for (feature_set, dur) in local_feature_seq:
        full_feature_list = []
        full_feature_list.extend(feature_set)
        for j in range(1, LEFT_CONTEXT + 1):
            delta_pos = i - j
            if delta_pos >= 0:
                full_feature_list.extend(
                    [("pos-%d:%s" % (j, s), value) for (s, value) in local_feature_seq[i - j][0] if not s.startswith("_")])
                if USE_DURATION_FEATURE:
                    full_feature_list.append(("pos-%d:dur" % j, dur_function(local_feature_seq[i - j][1])))
            else:
                full_feature_list.extend([("pos-%d:%s" % (j, s), value) for (s, value) in context.get(delta_pos, [])])

        for j in range(1, RIGHT_CONTEXT + 1):
            if i + j < len(local_feature_seq):
                full_feature_list.extend(
                    [("pos+%d:%s" % (j, s), value) for (s, value) in local_feature_seq[i + j][0] if not s.startswith("_")])
            else:
                full_feature_list.extend(
                    [("pos+%d:%s" % (j, s), value) for (s, value) in context.get(j - (len(local_feature_seq) - i - 1), [])])

        full_feature_seq.append((full_feature_list, dur))
        i += 1
    return full_feature_seq


def read_transitions(filename):
    # a list of transitions, add None to make it aligned with transition IDs
    transitions = [None]
    final_transition_states = {}
    print >> sys.stderr, "Reading transition model..."
    current_phone = None
    for l in open(filename):
        if l.startswith("Transition-state "):
            ss = l.split()
            current_phone = ss[4]
            hmm_state = int(ss[7])
            final_transition_states[current_phone] = hmm_state
        elif l.startswith(" Transition-id = "):
            ss = l.split()
            to_state = None
            if len(ss) == 9 and ss[7] == "->":
                to_state = int(ss[8][:-1])
            transitions.append((current_phone, to_state))
        else:
            raise Exception("Unexpected line in transition model data: ", l)
    print >> sys.stderr, "Finding final states"
    for (i, transition) in enumerate(transitions):
        if transition is None:
            continue
        if transition[1] is not None and final_transition_states[transition[0]] + 1 == transition[1]:
            transitions[i] = (transition[0], transition[1], True)
        else:
            transitions[i] = (transition[0], transition[1], False)
    return transitions