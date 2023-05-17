import json
import nltk
from nltk.corpus import stopwords
words = stopwords.words('english')

pos_hash = {
    'r': ('RB', 'RBR', 'RBS', 'WRB'), 
    'v': ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'), 
    'n': ('NN', 'NNP', 'NNS', 'NNPS'), 
    'a': ('JJ', 'JJR', 'JJS')
}

pos_h = {}
for k, v in pos_hash.items():
    for vv in v:
        pos_h[vv] = k

with open('../../data/sws/sws_test.json', 'r') as f:
    j = json.load(f)
with open('test_LS.txt', 'w') as f:
    for idx, v in j.items():
        s = v['sentence_split']
        pos = nltk.pos_tag(s)
        for i in range(len(s)):
            if pos[i][1] in pos_h and s[i] not in words:
                f.write("%s.%s\t%s\t%d\t%s\n" % (s[i], pos_h[pos[i][1]], idx, i, ' '.join(s)))
