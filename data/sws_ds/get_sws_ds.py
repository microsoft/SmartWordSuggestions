import re
import os
import json
import nltk
import random
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import collections
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import inflect
inflect = inflect.engine()
import Levenshtein

substitute_rate = 1

def processWord(s):
    # print(s)
    if 'www.un.org' in s:
        s = s.split('/')[-1][:-5]
    if '[' in s:
        s = re.sub('\[.*?\]', '', s)
    while s and s[-1] == ' ': s = s[:-1]
    while s and s[0] == ' ': s = s[1:]
    return s

def valid(substitute, original):
    if substitute in original or original in substitute: return False
    if Levenshtein.distance(substitute, original) < min(len(substitute)//3, 3): return False
    pos = nltk.pos_tag(original.split(' '))[0][1][0]
    if pos in 'VN':
        if (substitute[-1] == 's' and original[-1] != 's') or (substitute[-1] != 's' and original[-1] == 's'): return False
        if (substitute[-3:] == 'ing' and original[-3:] != 'ing') or (substitute[-3:] != 'ing' and original[-3:] == 'ing'): return False
    if pos == 'V':
        if (substitute[-2:] == 'ed' and original[-2:] != 'ed') or (substitute[-2:] != 'ed' and original[-2:] == 'ed'): return False
    return True

def load_ppdb():
    cnt = 0
    substitutedict = collections.defaultdict(set)
    with open('ppdb-2.0-s-lexical', 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        l = line.split(' ||| ')
        target, substitute, score = processWord(l[1]), processWord(l[2]), float(l[3].split(' ')[0].split('=')[1])
        if target and substitute:
            substitutedict[target].add(substitute)

    for k in substitutedict:
        substitutedict[k] = list(substitutedict[k])
    return substitutedict

def get_synonym_dict():
    all_poses = set()
    with open('thesaurus.json', 'r') as f:
        original = json.load(f)
    ret = collections.defaultdict(lambda:{})
    for k, vs in original.items():
        k_syn = collections.defaultdict(list)
        for v in vs:
            pos = re.sub(" \(.*?\)", "", v['pos'])
            for kk, vv in v.items():
                if 'Synonyms' in kk:
                    vv_processed = []
                    for word in vv:
                        if '(' in word:
                            alternative = re.findall('\((.*?)\)', word)[0]
                            baseword = re.sub('\((.*?)\)', '', word)
                            vv_processed.append(baseword if baseword[-1] != ' ' else baseword[:-1])
                            for z in alternative.split(' or '):
                                vv_processed.append(baseword + z)
                        else:
                            vv_processed.append(word)
                    k_syn[pos] += vv_processed
            all_poses.add(pos)
        ret[k] = k_syn
    print(all_poses)
    return ret

def randomChange(sent, substitutedict):
    words = nltk.word_tokenize(sent)
    poses = nltk.pos_tag(words)
    res, targets, substitutes = [], [], []
    toskip = 0
    for i, word in enumerate(words):
        while toskip:
            toskip -= 1
            continue
        if ' '.join(words[i:i+3]) not in set(stopwords.words('english')) and substitutedict[' '.join(words[i:i+3])] and 'phrase' in substitutedict[' '.join(words[i:i+3])] and substitutedict[' '.join(words[i:i+3])]['phrase']:
            subthis = random.shuffle(substitutedict[' '.join(words[i:i+3])]['phrase'])
            if subthis: 
                targets.append((subthis, ' '.join(words[i:i+3]), i, 3))
                toskip = 2
        elif ' '.join(words[i:i+2]) not in set(stopwords.words('english')) and substitutedict[' '.join(words[i:i+2])] and 'phrase' in substitutedict[' '.join(words[i:i+2])] and substitutedict[' '.join(words[i:i+2])]['phrase']:
            subthis = random.shuffle(substitutedict[' '.join(words[i:i+2])]['phrase'])
            if subthis:
                targets.append((subthis, ' '.join(words[i:i+2]), i, 2))
                toskip = 1
        elif word not in set(stopwords.words('english')) and substitutedict[word] and rev_pos_hash[poses[i][1]] in substitutedict[word] and substitutedict[word][rev_pos_hash[poses[i][1]]]:
            candidate_list = list(set(ppdbdict[word]) & set(substitutedict[word][rev_pos_hash[poses[i][1]]]))
            if candidate_list: targets.append((random.choice(candidate_list), word, i, 1))

    targets_idx = random.sample(range(len(targets)), min(len(targets), int(len(words)*substitute_rate)))
    new_targets = []
    for i in range(len(targets)):
        if i in targets_idx:
            new_targets.append(targets[i])
            res.append([[targets[i][1], targets[i][2], targets[i][2]+targets[i][3]], targets[i][0]])
    substitutes = [i[0] for i in new_targets]
    targets = [i[1] for i in new_targets]
    return targets, substitutes, res

ppdbdict = load_ppdb()
substitutedict = get_synonym_dict()

pos_hash = {
    'plural noun': ('NNS', 'NNPS'), 
    'preposition': ('IN'), 
    'adverb': ('RB', 'RBR', 'RBS', 'WRB'), 
    'conjunction': ('CC'), 
    'verb': ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'), 
    'noun': ('NN', 'NNP'), 
    'phrase': (), 
    'interjection': (), 
    'pronoun': ('PRP$', 'PRP'), 
    'adjective': ('JJ', 'JJR', 'JJS')
}
rev_pos_hash = collections.defaultdict(lambda:'')
for k, v in pos_hash.items():
    for vv in v:
        rev_pos_hash[vv] = k

sents = []

# read wiki sents ----
files = os.listdir('/home/t-chenswang/data/outwiki/AA/')
for file in tqdm(files):
    with open('/home/t-chenswang/data/outwiki/AA/' + file, 'r') as f:
        lines = f.readlines()
    # for line in tqdm(lines):
    for line in lines:
        now = json.loads(line)
        sents += [i+'.' for i in re.split('\. |\.\n', now['text'])]

random.shuffle(sents)

f = open('sws_ds.json' % substitute_rate, 'w') 

cnt = 0
total_sub = 0
total_word = 0

for sent in tqdm(sents):
    targets, substitutes, res = randomChange(sent, substitutedict)
    if not len(targets): continue
    thissent = {}
    thissent['sent'] = sent
    thissent['targets'] = targets
    thissent['substitutes'] = substitutes
    f.write(json.dumps(thissent, ensure_ascii=False)+'\n')
    cnt += 1
    total_sub += len(targets)
    total_word += len(nltk.word_tokenize(sent))
print('%d lines' % cnt)
print('%f avg substitute' % (total_sub / cnt))
print('%f sub rate' % (total_sub / total_word))