#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, os
import json
import time
import nltk
import random
from tqdm import tqdm
from torch.utils.data import Dataset

def preprocess_sws(data_path):
    # dev & test

    with open(data_path, 'r') as f:
        j = json.load(f)
    testres = []
    for k, v in tqdm(j.items()):
        testres.append([(k, v['sentence']), [], []])
    return testres

def preprocess_sws_ds(data_path):
    # train

    with open(data_path, 'r') as f:
        lines = f.readlines()
        trainres = []
        for line in tqdm(lines):
            now = json.loads(line)
            now = [now['sent'], now['targets'], now['substitutes']]
            trainres.append(now)
        random.shuffle(trainres)

    return trainres

class get_dataset(Dataset):

    def __init__(self, data_lst, mode='train'):

        with open(os.path.join('data/', '%s.en-de.en'%mode), 'w') as f:
            pass
        with open(os.path.join('data/', '%s.en-de.de'%mode), 'w') as f:
            pass
        if mode == 'test':
            with open(os.path.join('data/', '%s.en-de.idx'%mode), 'w') as f:
                pass

        for sent, targets, reses in tqdm(data_lst):

            if mode == 'test' or mode == 'valid':
                idx, sent = sent[0], sent[1]

            for i in range(len(targets)):
                if targets[i][-1] in '.,?!':
                    targets[i] = targets[i][:-1]
                if targets[i] and targets[i][0] in '.,?!':
                    targets[i] = targets[i][1:]
                targets[i] = re.sub('\(', '', targets[i])
                targets[i] = re.sub('\)', '', targets[i])
            for i in range(len(targets)):
                if reses[i][-1] in '.,?!':
                    reses[i] = reses[i][:-1]
                if reses[i] and reses[i][0] in '.,?!':
                    reses[i] = reses[i][1:]
                reses[i] = re.sub('\(', '', reses[i])
                reses[i] = re.sub('\)', '', reses[i])

            sent = re.sub("\s+", " ", sent)
            sent = re.sub("、", ",", sent)
            sent = re.sub("\(", " (", sent)
            sent = re.sub("\)", ") ", sent)
            sent = re.sub("\[", " [", sent)
            sent = re.sub("\]", "] ", sent)
            sent = re.sub(",", ", ", sent)
            sent = re.sub("\.", ". ", sent)
            sent = re.sub("\?", "? ", sent)
            sent = re.sub("!", "! ", sent)
            sent = re.sub(" ,", ",", sent)
            sent = re.sub(" \.", ".", sent)
            sent = re.sub(" \?", "?", sent)
            sent = re.sub(" !", "!", sent)
            sent = re.sub("\s+", " ", sent)
            sent = re.sub("\. \"", ".\"", sent)
            sent = re.sub("\" \.", "\".", sent)
            sent = re.sub(", \"", ".\"", sent)
            sent = re.sub("\" ,", "\".", sent)
            while sent and sent[-1] == ' ': sent = sent[:-1]
            while sent and sent[0] == ' ': sent = sent[1:]
            if sent[-1] not in '.?!\"\'':
                sent += '.'

            if mode == 'test' or mode == 'valid':
                self.process_sent(sent, targets, reses, mode, idx)
            else:
                self.process_sent(sent, targets, reses, mode)

    def process_sent(self, sent, targets, reses, mode, idx=None):

        out_sent = sent
        for i in range(len(targets)):
            if targets[i] and reses[i]: out_sent = re.sub(' '+targets[i]+' ', ' '+reses[i]+' ', out_sent)
        import os
        with open(os.path.join('data/', '%s.en-de.en'%mode), 'a') as f:
            f.write(' '.join(nltk.word_tokenize(sent)))
            f.write('\n')
        with open(os.path.join('data/', '%s.en-de.de'%mode), 'a') as f:
            f.write(' '.join(nltk.word_tokenize(out_sent)))
            f.write('\n')
        if mode == 'test':
            with open(os.path.join('data/', '%s.en-de.idx'%mode), 'a') as f:
                f.write(str(idx))
                f.write('\n')

if __name__ == '__main__':

    train_lst = preprocess_sws_ds('../../../data/sws_ds/sws_ds.json')
    # train_lst = preprocess_sws_ds('/home/v-chenswang/iven/code/WriteModelTraining/WordSubstitution/data/wiki/wiki_170_s-lexical-1-1014.json')
    get_dataset(train_lst)

    dev_lst = preprocess_sws('../../../data/sws/sws_eval.json')
    get_dataset(dev_lst, mode='valid')

    test_lst = preprocess_sws('../../../data/sws/sws_test.json')
    get_dataset(test_lst, mode='test')
