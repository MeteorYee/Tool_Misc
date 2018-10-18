# -*- coding: utf-8 -*-
#
# Author: Xingyu Yi
#
# Created at: 10/18/2018
#
# Description: Take a document and get the corresponding TF-PDF
# (TF-PDF: K. K. Bun, M. Ishizuka: "Topic Extraction from News Archive Using TF*PDF Algorithm")
#
# Last Modified at: 10/18/2018, by: Xingyu Yi


from __future__ import division

from collections import defaultdict

# import numpy as np

import argparse
import math
import os
# import time


class TfPdfGenerator(object):
    def __init__(self, stop_words_path = None):
        self._stop_set = set([])

        if stop_words_path != None:
            assert os.path.exists(stop_words_path)
            self._stop_words_path = stop_words_path
            self._load_stop_words()


    # Handle all the files in data_path.
    # As the TF-PDF paper suggests:
    # We consider each file as a channel, and each sentence in one file as
    # a document.
    def __call__(self, data_path):
        assert os.path.exists(data_path)
        print("Analyzing all the files in:", data_path)

        tfpdf_dict = defaultdict(float)
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            if os.path.isfile(file_path):
                tfpdf_res = self.get_one_channel_weight(file_path)
                for tk, v in tfpdf_res:
                    tfpdf_dict[tk] += v

        return [(tk, v) for tk, v in tfpdf_dict.items()]


    # Parameters:
    # @in_file: The input file. The text in the file should be TOKENIZED ideally.
    #           Each sentence is considered as a document. One file, one channel.
    #
    # Returns:
    # tfpdf_res: the result will be fed into this list, in a form of:
    # [(token, TF-PDF), ... ]
    def get_one_channel_weight(self, in_file):
        assert os.path.exists(in_file)
        # Total number of document in channel c
        N_c = 0
        # Number of document in channel c where term j occurs
        # document frequency dict
        df_dict = defaultdict(int)
        # term frequency in the channel
        tf_dict = defaultdict(int)
        with open(in_file, 'r') as fin:
            for doc in fin:
                tk_list = doc.strip().split()
                tk_set = set([])
                for tk in tk_list:
                    tk = tk.lower()
                    tf_dict[tk] += 1
                    tk_set.add(tk)

                for tk in tk_set:
                    df_dict[tk] += 1

                N_c += 1

        ##################################################
        # The two methods below work equally efficiently.#
        ##################################################

        # I.
        ### start of numpy method ###
        ### remember to import numpy if using this method ###
        # start = time.time()
        # vocab_list = []
        # pdf_list = []
        # freq_list = []

        # for tk, df in df_dict.items():
        #     if tk in self._stop_set:
        #         continue
        #     vocab_list.append(tk)
        #     pdf_list.append(df)
        #     freq_list.append(tf_dict[tk])

        # # PDF vector, [V, ], ( exp(n_jc/N_c) )
        # v_pdf = np.array(pdf_list)
        # v_pdf = np.exp(v_pdf / N_c)
        # # frequency vector, [V, ], ( |F_jc| )
        # v_freq = np.array(freq_list)
        # l2_norm = np.linalg.norm(v_freq)
        # v_freq = v_freq / l2_norm
        # v_tfpdf = v_freq * v_pdf

        # tfpdf_res = [(tk, v) for tk, v in zip(vocab_list, v_tfpdf)]
        # end = time.time()
        # print("numpy method spent %.3fs" % (end - start))
        ### end of numpy method

        # II.
        ### start of standard python method ###
        # start = time.time()
        vocab_list = []
        freq_list = []
        l2_norm = 0
        for tk, df in df_dict.items():
            if tk in self._stop_set:
                continue
            vocab_list.append(tk)
            pdf = math.exp(df / N_c)
            tf = tf_dict[tk]
            l2_norm += tf * tf
            freq_list.append(tf * pdf)

        l2_norm = math.sqrt(l2_norm)
        tfpdf_res = [(tk, v/l2_norm) for tk, v in zip(vocab_list, freq_list)]
        # end = time.time()
        # print("standard python method spent %.3fs" % (end - start))
        ### end of standard python method ###

        return tfpdf_res


    # load the stop words
    def _load_stop_words(self):
        print("Try to use the stop words in:", self._stop_words_path)
        with open(self._stop_words_path, 'r') as fin:
            for words in fin:
                for wd in words.strip().split(','):
                    self._stop_set.add(wd)
        # add the comma
        self._stop_set.add(',')


    # output the given tfidf results (tfpdf_res) into the designated file
    def write_to_file(self, tfpdf_res, output_file):
        print("Writing results to:", output_file)
        with open(output_file, 'w') as fout:
            for values in tfpdf_res:
                value_string = '\t'.join([str(v) for v in values])
                fout.write(value_string + '\n')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type = str,
        default = "./elisa.som-eng.y3.bitext/train.eng.tok",
        help = """the input file used to generate TF-PDF""")

    parser.add_argument(
        "--data_path",
        type = str,
        default = "./elisa.som-eng.y3.bitext/train_shards",
        help = """the data path for all the channels (several input
            files, as the TF-PDF paper described)""")

    parser.add_argument(
        "--output_file",
        type = str,
        default = "./train.eng.tok.tfpdf.shuf",
        help = """the output file for the generated TF-PDF scores""")

    parser.add_argument(
        "--stop_words_path",
        type = str,
        default = "./stop_words_en.csv",
        help = """The path to the stop words, a csv file.""")

    args = parser.parse_args()

    # demo code
    input_file = args.input_file
    data_path = args.data_path
    output_file = args.output_file
    stop_words_path = args.stop_words_path

    tfpdf = TfPdfGenerator(stop_words_path)

    # calculate the tf-pdf scores for the given file, which means just ONE channel
    # tfpdf_res = tfpdf.get_one_channel_weight(input_file)

    # calculate the tf-pdf scores for the given path, which means several channels
    tfpdf_res = tfpdf(data_path)
    
    assert tfpdf_res != None
    # sort it by tf-pdf score in a descending order
    tfpdf_res.sort(key = lambda x : x[1], reverse = True)
    tfpdf.write_to_file(tfpdf_res, output_file)