import json
res = {}

with open('/home/v-chenswang/iven/code/tmp/LexSubCon/dataset/results/results_gapproposed_6809.oot', 'r') as f:
    for l in f.readlines():
        pref, cands = l.strip().split(' ::: ')
        word, idx, position  = pref.split(' ')
        if idx not in res: res[idx] = {"substitute_topk":[]}
        res[idx]['substitute_topk'].append([[word, int(position), int(position)+1], cands.split(';')])

f = open('res.json', 'w')
json.dump(res, f, indent=4)