# -*- coding: utf-8 -*-
#
# Author: Xingyu Yi
#
# Created at: 10/15/2018
#
# Description: Take a document and get the corresponding TF-IDF
#
# Last Modified at: 10/16/2018, by: Xingyu Yi


from __future__ import division

from collections import defaultdict

import argparse
import math
import os
import pickle


class TfIdfGenerator(object):
    def __init__(self, idf_dict_path):
        self._idf_dict_path = idf_dict_path

        if os.path.exists(idf_dict_path):
            self._load_idf_dict()
        else:
            print("idf_dict doesn't exist, needs to be generated")
            self._idf_dict = None


    # Parameters:
    # @in_file: The input file. The text in the file should be TOKENIZED ideally.
    #
    # Returns:
    # tfidf_res: the result will be fed into this list, in a form of:
    # [(token, TF, IDF, TF-IDF), ... ]
    def __call__(self, in_file):
        if self._idf_dict == None:
            print("We need IDF scores.")
            print("Try to use the method to get one.")
            return None

        assert os.path.exists(in_file)
        cnt_dict = defaultdict(int)
        total_tk = 0
        with open(in_file, 'r') as fin:
            for line in fin:
                tk_list = line.strip().split()
                for tk in tk_list:
                    cnt_dict[tk] += 1
                    total_tk += 1

        tfidf_res = []
        for tk, tcnt in cnt_dict.items():
            # int / int = float, __future__.division already imported
            tfreq = tcnt / total_tk
            if tk in self._idf_dict:
                idf = self._idf_dict[tk]
            else:
                idf = self._idf_dict["unk"]

            tfidf_res.append((tk, tfreq, idf, tfreq*idf))

        return tfidf_res


    # Calculate the IDF scores, given that we've already known the document
    # frequency for each token.
    # The parameter is the number of documents.
    def _calculate_idf(self, N):
        LOG = math.log
        log_N = LOG(N)
        self._idf_dict["unk"] = log_N
        for tk, df in self._idf_dict.items():
            # use (df+1) to prevent log(0)
            self._idf_dict[tk] = log_N - LOG(df + 1)


    # dump the idf_dict using pickle
    def _dump_idf_dict(self):
        print("Dump idf_dict to:", self._idf_dict_path)
        with open(self._idf_dict_path, 'wb') as fout:
            pickle.dump(self._idf_dict, fout)


    # load the idf_dict using pickle
    def _load_idf_dict(self):
        print("Load idf_dict from:", self._idf_dict_path)
        with open(self._idf_dict_path, 'rb') as fin:
            self._idf_dict = pickle.load(fin)


    # Parameters:
    # @instream: an iterable object which can generate texts, such as a file
    # input stream or a list; (The text should be TOKENIZED ideally.) Each
    # text in the generator is considered as a document for calculating
    # the IDF score.
    def get_idf_score(self, instream):
        self._idf_dict = defaultdict(int)
        # the number of documents
        doc_num = 0
        for doc in instream:
            tk_list = doc.strip().split()
            # get the unqiue tokens in the doc
            tk_set = set([])
            for tk in tk_list:
                tk_set.add(tk)

            for tk in tk_set:
                self._idf_dict[tk] += 1

            doc_num += 1

        # The above is collecting the document frequency for each token.
        # Now let's calculate the idf scores.
        self._calculate_idf(doc_num)
        self._dump_idf_dict()


    # output the given tfidf results (tfidf_res) into the designated file
    def write_to_file(self, tfidf_res, output_file):
        print("Writing results to:", output_file)
        with open(output_file, 'w') as fout:
            for values in tfidf_res:
                value_string = '\t'.join([str(v) for v in values])
                fout.write(value_string + '\n')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type = str,
        default = "./elisa.som-eng.y3.bitext/train.eng.tok",
        help = """the input file used to generate IDF dict""")

    parser.add_argument(
        "--output_file",
        type = str,
        default = "./train.eng.tok.tfidf",
        help = """the output file for the generated TF-IDF scores""")

    parser.add_argument(
        "--idf_dict_path",
        type = str,
        default = "./idf_dict.pk",
        help = """The path to the IDF dict, it should be a pickle dumped object.
            If it doesn't exist, a new idf_dict will be generated to this path.""")

    args = parser.parse_args()

    # demo code
    input_file = args.input_file
    output_file = args.output_file
    idf_dict_path = args.idf_dict_path

    tfidf = TfIdfGenerator(idf_dict_path)
    # generate the idf_dict, if the idf_dict is already there, skip this step
    with open(input_file, 'r') as fin:
        tfidf.get_idf_score(fin)

    # calculate the tf-idf scores for the given file
    tfidf_res = tfidf(input_file)
    assert tfidf_res != None
    # sort it by tf-idf score in a descending order
    tfidf_res.sort(key = lambda x : x[3], reverse = True)
    tfidf.write_to_file(tfidf_res, output_file)