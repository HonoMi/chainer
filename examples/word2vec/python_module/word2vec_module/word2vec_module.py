# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from utility import kernprof_preprocess


def create_vocab(filenames, count_threshold=0, unk_rep='<unk>'):
    word2count_dict = defaultdict(int)
    for filename in filenames:
        print(filename)
        for i, line in enumerate(open(filename, 'r')):
            print(i)
            for word in line.rstrip().split():
                word2count_dict[word] += 1

    word2wid = {unk_rep: 0}
    wid = 1
    for word, count in sorted(word2count_dict.items(), key=lambda x: x[0]):
        if count > count_threshold:
            word2wid[word] = wid
            wid += 1


class WordsGenerator:

    def __init__(self, filenames, batch_size=10000000):
        self.filenames = filenames
        self.batch_size = batch_size

    def __call__(self):
        words_batch = []
        for filename in self.filenames:
            for line in open(filename, 'r'):
                words = line.rstrip().split()
                for word in words:
                    words_batch.append(words)
                    if len(words_batch) >= self.batch_size:
                        yield words_batch
                        words_batch = []
        if len(words_batch) != 0:
            yield words_batch
