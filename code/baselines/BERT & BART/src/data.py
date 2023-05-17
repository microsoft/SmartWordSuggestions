#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
import re
import csv
import json
import copy
import torch
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import Config
args = Config()

from logger import logging

def preprocess_test_data(data_path, mode='train'):

    with open(data_path, 'r') as f:
        j = json.load(f)
    testres = []
    for k, v in tqdm(j.items()):
        testres.append([(k, v['sentence']), [], []])
    if mode == 'train':
        return [], testres, []
    elif mode == 'test':
        return [], [], testres

def preprocess_train_data(data_path):
    '''
    load data json file
    if mode='train': split train:dev as 9:1
    if mode='test': do nothing
    '''

    with open(data_path, 'r') as f:
        lines = f.readlines()
        trainres = []
        idx = 0
        for line in tqdm(lines):
            now = json.loads(line)
            now = [now['sent'], now['targets'], now['substitutes']]
            trainres.append(now)
            idx += 1
            if idx == args.train_use: break
        random.shuffle(trainres)

    # logging("train sentence:%d, test sentence:%d" % (len(trainres), len(testres)))

    return trainres, 0, 0

def get_tokenizer(plm_type=args.plm_type):
    '''
    load tokenizer
    '''
    if plm_type == 'bert':
        from transformers import BertTokenizer as Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    elif plm_type == 'bart':
        from transformers import BartTokenizer as Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.bart_path, add_prefix_space=False)

    return tokenizer

class get_dataset(Dataset):

    def __init__(self, data_lst, tokenizer, mode='train'):

        self.tokenizer = tokenizer
        self.mode = mode
        data = {'inputs':[], 'outputs':[], 'target':[], 'sent':[]}

        logging('Masking targets...')
        for sent, targets, reses in tqdm(data_lst):

            if mode == 'test':
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

            # print(sent, targets, reses)

            # for p in '\{\}【】':
            #     sent = re.sub(p, " " + p + " ", sent)
            sent = re.sub("\s+", " ", sent)
            sent = re.sub("、", ",", sent)
            sent = re.sub("\(", " (", sent)
            sent = re.sub("\)", ") ", sent)
            sent = re.sub("\[", " [", sent)
            sent = re.sub("\]", "] ", sent)
            sent = re.sub(",", ", ", sent)
            sent = re.sub("\.", ". ", sent) # although it will cause "a.m." -> "a. m.", but make more sentence ends with a space, which is more relavent with substitution
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
                # print(sent) 
                sent += '.'

            inputs, outputs, targets, substitutes = self.process_sent(sent, targets, reses, mode)
            data['inputs'] += inputs
            data['outputs'] += outputs
            data['target'] += targets
            try:
                for target in targets:
                    sent = re.sub('([ |\.|,])' + target + '([ |\.|,])', '\\1<t> ' + target + ' </t>\\2', ' ' + sent + ' ')
                data['sent'] += [sent] * len(outputs)
            except:
                if mode == 'test':
                    data['sent'] += [idx + "=+=+" + sent] * len(outputs)
                else:
                    data['sent'] += [sent] * len(outputs)

        self.df = pd.DataFrame(data)

        if len(self.df):

            logging("Tokenizing data...")
            tqdm.pandas(desc='Tokenizing input_ids')
            self.df['input_ids'] = self.df.progress_apply(lambda x: tokenizer.convert_tokens_to_ids(x['inputs']),axis=1)
            tqdm.pandas(desc='Tokenizing output_ids')
            self.df['output_ids'] = self.df.progress_apply(lambda x: tokenizer.convert_tokens_to_ids(x['outputs']),axis=1)
            tqdm.pandas(desc='Tokenizing mask_pos')
            self.df['mask_pos'] = self.df.progress_apply(self.find_mask_pos,axis=1)

            self.df.dropna(axis=0, inplace=True)

            for i in range(len(self.df)):
                if i < 10:
                    logging(self.df.iloc[i]["inputs"])
                    logging(self.df.iloc[i]["outputs"])
                    logging(self.df.iloc[i]["input_ids"])
                    logging(self.df.iloc[i]["output_ids"])
                    if 'mlm' in args.model_choice:
                        inputid, outputid = [], []
                        for ii in range(len(self.df.iloc[i]["mask_pos"])):
                            if self.df.iloc[i]["mask_pos"][ii] == 1:
                                inputid.append(self.df.iloc[i]["input_ids"][ii])
                                outputid.append(self.df.iloc[i]["output_ids"][ii])
                        inputtoken, outputtoken = self.tokenizer.convert_ids_to_tokens(inputid), self.tokenizer.convert_ids_to_tokens(outputid)
                        logging('mask position=%s\ninput token id=%s\noutput token id=%s\ninput token=%s\noutput token=%s' % (
                            str(self.df.iloc[i]["mask_pos"]), 
                            str(inputid),
                            str(outputid),
                            str(inputtoken),
                            str(outputtoken)
                        ))

    def process_sent(self, sent, targets, reses, mode, idx=None):
        '''
        input: a sentence sample
        output: input/output pair
        '''

        if len(self.tokenizer.tokenize(sent)) > 500: return [], [], [], []
        inputs, outputs = [], []

        if args.model_choice == 'seq2seq':
            out_sent = sent
            for i in range(len(targets)):
                if targets[i] and reses[i]: out_sent = re.sub(' '+targets[i]+' ', ' '+reses[i]+' ', out_sent)
            sent = [args.cls_token] + self.tokenizer.tokenize(sent) + [args.sep_token]
            out_sent = [args.cls_token] + self.tokenizer.tokenize(out_sent) + [args.sep_token]
            inputs.append(sent)
            outputs.append(out_sent)
            return inputs, outputs, [targets], [reses]

        substitute_dict = {}

        for i in range(len(targets)):

            target, res = targets[i], reses[i]
            if len(target) < 3 or len(res) < 3: continue

            target_tokens = self.tokenizer.tokenize(" " + target)
            target_token = target_tokens[0]
            res_tokens = self.tokenizer.tokenize(" " + res)
            res_token = res_tokens[0]
            if len(target_tokens) > 1 or len(res_tokens) > 1:
                if mode == 'train': continue
                else: res_token = '<unk>' # for test samples, regard gold substitute as unk for now (#TBD)

            substitute_dict[target_token] = res_token

        if mode == 'train' and not len(substitute_dict): return [], [], [], []
        substitute_dict_full = copy.deepcopy(substitute_dict)

        sent = [args.cls_token] + self.tokenizer.tokenize(sent) + [args.sep_token]
        inputs.append(sent)
        output_sent = []
        for i in sent:
            if i not in substitute_dict:
                output_sent.append(i)
            else:
                output_sent.append(substitute_dict[i])
                substitute_dict.pop(i)
        outputs.append(output_sent)

        return inputs, outputs, [list(substitute_dict_full.keys())], [list(substitute_dict_full.values())]

    def find_mask_pos(self, x):

        if args.model_choice == 'seq2seq': return 0

        if args.mask_id in x['input_ids']:
            return [1 if i==args.mask_id else 0 for i in x['input_ids']]
        else:
            targets = [self.tokenizer.convert_tokens_to_ids([t])[0] for t in x['target']]
            return [1 if x['input_ids'][i] != x['output_ids'][i] else 0 for i in range(len(x['input_ids']))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if 'mlm' in args.model_choice:
            assert len(self.df.iloc[idx]['input_ids']) == len(self.df.iloc[idx]['output_ids']), str(self.df.iloc[idx])
        if args.use_reverse_wiki and self.mode == 'train':
            return torch.LongTensor(self.df.iloc[idx]['output_ids']), \
                    torch.LongTensor(self.df.iloc[idx]['input_ids']), \
                    torch.LongTensor(self.df.iloc[idx]['mask_pos']), \
                    self.df.iloc[idx]['sent']
        else:
            return torch.LongTensor(self.df.iloc[idx]['input_ids']), \
                    torch.LongTensor(self.df.iloc[idx]['output_ids']), \
                    torch.LongTensor(self.df.iloc[idx]['mask_pos']), \
                    self.df.iloc[idx]['sent']

class Pad_Sequence():
    """
    collate_fn
    """
    def __init__(self, seq_pad_value, label_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        input_ids = [x[0] for x in sorted_batch]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.seq_pad_value)
        # input_ids_lengths = torch.LongTensor([len(x) for x in input_ids])

        output_ids = [x[1] for x in sorted_batch]
        output_ids_padded = pad_sequence(output_ids, batch_first=True, padding_value=self.seq_pad_value)
        # output_ids_le ngths = torch.LongTensor([len(x) for x in output_ids])

        mask_pos = [x[2] for x in sorted_batch]
        mask_pos_padded = pad_sequence(mask_pos, batch_first=True, padding_value=0)
        # output_ids_le ngths = torch.LongTensor([len(x) for x in output_ids])

        input_tokens = [x[3] for x in sorted_batch]

        return input_ids_padded, output_ids_padded, mask_pos_padded, input_tokens

def load_datasets(args, tokenizer, mode='train'):

    if mode == 'train':

        train_lst, _, _ = preprocess_train_data(args.data_path)
        _, dev_lst, _ = preprocess_test_data(args.dev_data_path, mode='train')

        train_set = get_dataset(train_lst, tokenizer=tokenizer)
        dev_set = get_dataset(dev_lst, tokenizer=tokenizer, mode='test')

        return train_set, dev_set, 0

    elif mode=='test':

        _, _, test_lst = preprocess_test_data(args.test_data_path, mode='test')
        test_set = get_dataset(test_lst, tokenizer=tokenizer, mode='test')

        return 0, 0, test_set

def load_dataloaders(args, mode='train'):

    tokenizer = get_tokenizer(args.plm_type)
    train_set, dev_set, test_set = load_datasets(args, tokenizer, mode=mode)

    if mode=='train':
        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id, label_pad_value=tokenizer.pad_token_id)
        train_length = len(train_set); dev_length = len(dev_set); test_length = 0
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                                    num_workers=0, collate_fn=PS, pin_memory=False)
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size//2, shuffle=False, \
                                num_workers=0, collate_fn=PS, pin_memory=False)
        test_loader = 0

        return train_loader, dev_loader, test_loader, train_length, dev_length, test_length

    elif mode=='test':
        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id, label_pad_value=tokenizer.pad_token_id)

        train_length = 0; dev_length = 0; test_length = len(test_set)
        train_loader = dev_loader = 0
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, \
                                num_workers=0, collate_fn=PS, pin_memory=False)

        return train_loader, dev_loader, test_loader, train_length, dev_length, test_length

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    load_datasets(args, tokenizer)