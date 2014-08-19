# -*- coding: UTF-8 -*-

'''
Created on Sep 9, 2013

@author: tanel
'''
import unittest
import os
import codecs
import durmodel_utils

class FeatureTests(unittest.TestCase):

    def testEncode(self):
        seq1 = [1, 1, 5, 5, 5, 106, 4, 4, 4]
        seq_rl = durmodel_utils.encode(seq1)
        self.assertEquals([(1, 2), (5, 3), (106, 1), (4,3)], seq_rl)

    #def testLocal(self):
    #    #komm
    #    word_id = 1
    #    words = ["<eps>", "koää"]
    #    seq = "59_59_59_59_59_93_93_93_93_93_93_93_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80".split("_")
    #    phone_map = {59 : "k", 93 : "o", 80: "aeae"}
    #    features_and_dur_seq = durmodel_utils.make_local(word_id, seq, phone_map, words)
    #    print features_and_dur_seq
    #    self.assertEquals([5,7,19], [x[1] for x in features_and_dur_seq])
    #    self.assert_("Stop" in features_and_dur_seq[0][0])
    #    self.assert_("syllable=1" in features_and_dur_seq[0][0])

    def test_syllabify(self):
        self.assertEquals([["t", "a"], ["n", "e", "l"]], durmodel_utils.syllabify(["t", "a", "n", "e", "l"], language='ESTONIAN',
                                                                                  nonsilence_phonemes=set('t a n e l'.split())))
        self.assertEquals([["k", "o"], ["mm", "i"]], durmodel_utils.syllabify(["k", "o", "mm", "i"], language='ESTONIAN',
                                                                              nonsilence_phonemes=set('k o mm i'.split())))
        self.assertEquals([["a"], ["j", "u"]], durmodel_utils.syllabify(["a", "j", "u"], language='ESTONIAN',
                                                                        nonsilence_phonemes="a j u".split()))
        self.assertEquals([["AH", "B"], ["S", "T", "EY", "N"]], durmodel_utils.syllabify("AH B S T EY N".split(), language='ENGLISH',
                                                                                         nonsilence_phonemes="AH B S T EY N".split()))

    def _testFullFeatures(self):
        seqs = []
        words = ["<eps>", "komm", "Tanel"]
        #komm
        seqs.append((1, "59_59_59_59_59_93_93_93_93_93_93_93_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80".split("_")))
        #SIL
        seqs.append((0, "6_6_6_6_6_6_6".split("_")))
        #Tanel
        seqs.append((2, "135_135_135_135_135_135_135_135_13_13_13_13_13_13_13_13_13_13_85_85_85_85_85_85_85_25_25_25_25_25_25_25_25_25_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68_68".split("_")))
        phone_map = {6: "SIL", 13: "a", 25: "e", 59 : "k", 68: "l", 93 : "o", 80: "mm", 85 : "n", 135 : "t"}
        features_and_durs = []
        for (word, seq) in seqs:
            features_and_durs.append(durmodel_utils.make_local(word, seq, phone_map, words))
            #print features_and_durs

        full_features_and_durs = durmodel_utils.make_linear(features_and_durs)
        self.assertEqual(8, len(full_features_and_durs))
        print full_features_and_durs[1][0]
        self.assert_("pos-1:dur=%f" % durmodel_utils.dur_function(5) in full_features_and_durs[1][0])
        for (f, d) in full_features_and_durs:
            print "%d    %s" % (d, " ".join(f))
        #pass





    def test_compile_features_for_word(self):
        context = {-2: set([("foo", 1), ("bar", 1), ("dur", 0.377)]), -1: set([("fii", 1), ("dur", 0.553)]), +1: set([("</s>", 1)])}
        local_feature_and_dur_seq = [(set([("bii", 1), ("boo", 1)]), 5), (set([("buu", 1), ("bee", 1)]), 6)]
        result = durmodel_utils.compile_features_for_word(context, local_feature_and_dur_seq)
        #print result[0]
        self.assertEqual(result[0][1], 5)
        self.assert_(("pos-2:foo", 1) in result[0][0])
        self.assert_(("bii", 1) in result[0][0])
        self.assert_(("pos-1:dur", 0.553) in result[0][0])
        self.assert_(("pos-2:fii", 1) in result[1][0])
        self.assert_(("pos+2:</s>", 1) in result[-2][0])
        self.assert_(("pos+1:</s>", 1) in result[-1][0])

        self.assert_(("pos-1:dur", durmodel_utils.dur_function(5)) in result[-1][0])


    def _test_english(self):
        phone_map = {}
        for l in open(os.path.dirname(__file__) + "/test_data/en/phones.txt"):
            ss = l.split()
            phone_map[int(ss[1])] = ss[0].partition("_")[0]

        word_list = []
        for l in open(os.path.dirname(__file__) + "/test_data/en/words.txt"):
            word_list.append(l.split()[0])

        #ONCE
        word_id = 7119
        seq = [int(x) for x in "146_146_146_146_146_146_146_146_146_146_146_146_146_146_146_146_16_16_16_16_16_16_16_96_96_96_96_96_119_119_119_119_119_119_119_119_119_119".split("_")]
        stress_dict = durmodel_utils.load_stress_dict(os.path.dirname(__file__) + "/test_data/en/cmudict.0.7a")
        features_and_dur_seq = durmodel_utils.make_local(word_id, seq, phone_map, word_list, language='ENGLISH', stress_dict=stress_dict)
        self.assert_('consonant' in features_and_dur_seq[-1][0])
        self.assert_('nasal' in features_and_dur_seq[-2][0])
        self.assert_('stress1' not in features_and_dur_seq[0][0])
        self.assert_('stress1' in features_and_dur_seq[1][0])

        #ARTHUR
        word_id = 563
        seq = [int(x) for x in "6_6_6_6_6_116_116_116_116_116_116_116_116_116_116_116_132_132_132_132_132_132_132_132_132_132_132_132_132_51_51_51_51_51_51_51_51_51_51_51_51_51_51_51".split("_")]
        features_and_dur_seq = durmodel_utils.make_local(word_id, seq, phone_map, word_list, language='ENGLISH', stress_dict=stress_dict)
        self.assert_('syllable=2' in features_and_dur_seq[-1][0])
        print features_and_dur_seq


    def _test_fisher(self):
        phone_map = {}
        for l in open(os.path.dirname(__file__) + "/test_data/fisher_english/phones.txt"):
            ss = l.split()
            phone_map[int(ss[1])] = ss[0].partition("_")[0]

        word_list = []
        for l in open(os.path.dirname(__file__) + "/test_data/fisher_english/words.txt"):
            word_list.append(l.split()[0])

        #ONCE
        word_id = 21442
        seq = [int(x) for x in "161_161_161_161_161_31_31_31_31_111_111_111_111_111_134_134_134_134_134_134".split("_")]
        stress_dict = durmodel_utils.load_stress_dict(os.path.dirname(__file__) + "/test_data/fisher_english/cmudict.0.7a_lowercase")
        features_and_dur_seq = durmodel_utils.make_local(word_id, seq, phone_map, word_list, language='ENGLISH', stress_dict=stress_dict)
        self.assert_('consonant' in features_and_dur_seq[-1][0])
        self.assert_('nasal' in features_and_dur_seq[-2][0])
        self.assert_('stress1' not in features_and_dur_seq[0][0])
        self.assert_('stress1' in features_and_dur_seq[1][0])

        #ARTHUR
        word_id = 1650
        seq = [int(x) for x in "21_21_21_21_21_21_21_21_21_21_21_21_21_21_131_131_131_131_131_131_147_147_147_147_147_147_147_147_147_147_147_147_147_147_66_66_66_66_66_66_66_66_66".split("_")]
        features_and_dur_seq = durmodel_utils.make_local(word_id, seq, phone_map, word_list, language='ENGLISH', stress_dict=stress_dict)
        self.assert_('syllable=2' in features_and_dur_seq[-1][0])
        print features_and_dur_seq


    def test_tedlium(self):
        transitions = durmodel_utils.read_transitions(os.path.dirname(__file__) + "/test_data/tedlium/transitions.txt")
        nonsilence_phonemes = set()
        for l in open(os.path.dirname(__file__) + "/test_data/tedlium/nonsilence.txt"):
            nonsilence_phonemes.add(l.partition("_")[0])

        word_list = []
        for l in codecs.open(os.path.dirname(__file__) + "/test_data/tedlium/words.txt", encoding="UTF-8"):
            word_list.append(l.split()[0])

        stress_dict = durmodel_utils.load_stress_dict(os.path.dirname(__file__) + "/data/en/cmudict.0.7a.lc")

        # about
        word_id = 437
        frames = [int(s) for s in "1794_2062_2061_2061_2300_5602_5601_5650_5662_5661_5661_4808_4807_4807_4807_4807_4832_4831_4831_4831_4860_4859_4859_22924_22923_22923_23018_23118".split("_")]

        features_and_dur_seq = durmodel_utils.make_local(0, word_id, frames, transitions, word_list, nonsilence_phonemes, language="ENGLISH", stress_dict=stress_dict)
        print features_and_dur_seq
        self.assert_(('syllable', 1) in features_and_dur_seq[0][0])
        self.assert_(('AH', 1) in features_and_dur_seq[0][0])
        self.assert_(('vowel', 1) in features_and_dur_seq[0][0])
        self.assert_(5, features_and_dur_seq[0][1])

        self.assert_(('stress1', 1) in features_and_dur_seq[2][0])

        self.assert_(('syllable', 2) in features_and_dur_seq[1][0])

        self.assert_(('stop', 1) in features_and_dur_seq[-1][0])
        self.assert_(5, features_and_dur_seq[0][-1])


    def test_stress(self):
        word = "abstain"
        phones = "AH B S T EY N".split()
        stress_dict = durmodel_utils.load_stress_dict(os.path.dirname(__file__) + "/data/en/cmudict.0.7a.lc")
        self.assertEqual(['AH', 'B', 'S', 'T', 'EY', 'N'], stress_dict["abstain"][0][0])
        self.assertEqual([0, 0, 0, 0, 1, 0], stress_dict["abstain"][0][1])
        stress = durmodel_utils.get_stress(word, phones, stress_dict)
        self.assertEqual([0, 0, 0, 0, 1, 0], stress)
        # test unknown word
        stress = durmodel_utils.get_stress("tanel", "T AH N EH L".split(), stress_dict)
        self.assertEqual([0, 0, 0, 0, 0], stress)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
