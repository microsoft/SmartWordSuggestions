from config import Config
args = Config()

import os
import re
import sys
import torch
import random
import subprocess
import numpy as np

from data import get_tokenizer

np.set_printoptions(threshold=np.inf)
tokenizer = get_tokenizer()

def set_seed(seed):
    '''
    set all the random seeds as args.seed
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_train_data():
    '''
    load training data, with different datasets
    '''
    from data import load_dataloaders
    train_loader, val_loader, _, train_len, val_len, _ = load_dataloaders(args)
    return train_loader, val_loader, train_len, val_len

def evaluate_mlm(output, labels, mask_pos, ignore_idx=-1, topk=10, gettopk=False):

    o_labels = torch.softmax(output, dim=1).max(1)[1]
    _, o_labels_topk = torch.topk(output, topk, dim=1, largest=True)

    l = labels # [bcz, seq_len]
    o = o_labels # [bcz, seq_len]
    o_topk = o_labels_topk.permute(0, 2, 1) # [bcz, seq_len, topk]

    gold_logits = torch.gather(output, 1, l.unsqueeze(1)).repeat(1, output.shape[1], 1) # [bcz, vocab_size, seq_len]
    rank_sgn_matrix = torch.where(gold_logits<output, torch.ones_like(gold_logits), torch.zeros_like(gold_logits)) # [bcz, vocab_size, seq_len]
    rank = torch.sum(rank_sgn_matrix, dim=1) # [bcz, seq_len]

    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist() # [bcz, seq_len]
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist() # [bcz, seq_len]
    o_topk = o_topk.cpu().numpy().tolist() if o_topk.is_cuda else o_topk.numpy().tolist() # [bcz, seq_len, topk]
    rank = rank.cpu().numpy().tolist() if rank.is_cuda else rank.numpy().tolist() # [bcz, seq_len]
    mask_pos = mask_pos.cpu().numpy().tolist() if mask_pos.is_cuda else mask_pos.numpy().tolist() # [bcz, seq_len]
    right, right_topk, mask_num = [], [], 0
    mlm_l, mlm_o, mlm_rank = [], [], []
    for i in range(len(o)):
        thissent_l, thissent_o, thissent_rank = [], [], []
        for j in range(len(o[i])):
            if mask_pos[i][j] == 1:
                thissent_l.append(l[i][j])
                thissent_o.append(o[i][j])
                thissent_rank.append(rank[i][j])
                if o[i][j]==l[i][j]: right.append(1)
                else: right.append(0)
                if l[i][j] in o_topk[i][j]: right_topk.append(1)
                else: right_topk.append(0)
                mask_num += 1
        mlm_l.append(thissent_l)
        mlm_o.append(thissent_o)
        mlm_rank.append(thissent_rank)
    acc = ((sum(right) + 0.0) / mask_num) if mask_num else 0
    topkcov = ((sum(right_topk) + 0.0) / mask_num) if mask_num else 0

    if not gettopk: return (acc, topkcov, mlm_rank, 0), (mlm_o, mlm_l)

    output = torch.softmax(output, dim=1)
    out = output.cpu().numpy().tolist() if output.is_cuda else output.numpy().tolist()
    topk_pred = [[tokenizer.convert_ids_to_tokens(j) for j in i] for i in o_topk]
    topk_logits = [[[] for _ in range(len(o_topk[0]))] for _ in range(len(o_topk))]
    for i in range(len(o_topk)):
        for j in range(len(o_topk[0])):
            for k in range(topk):
                topk_logits[i][j].append([out[i][o_topk[i][j][k]][j], topk_pred[i][j][k]])
    topk_pred = [[sorted(i, reverse=True) for i in j] for j in topk_logits]

    return (acc, topkcov, mlm_rank, topk_pred), (mlm_o, mlm_l)

def getSubstitutePairs(pred_lst, input_lst, topk_pred, input_sent):

    def LCS(A,B):

        A.append('0')
        B.append('0')

        n = len(A)
        m = len(B)

        A.insert(0,'0')
        B.insert(0,'0')

        L = [ ([0]*(m+1)) for i in range(n+1) ]
        C = [ ([0]*(m+1)) for i in range(n+1) ]

        for x in range (0,n+1):
            for y in range (0,m+1):
                if (x==0 or y==0):
                    L[x][y] = 0
                elif A[x] == B[y]:
                    L[x][y] = ( L[x-1][y-1] + 1 )
                    C[x][y] = 0
                elif L[x-1][y] >= L[x][y-1]:
                    L[x][y] = L[x-1][y]
                    C[x][y] = 1
                else:
                    L[x][y] = L[x][y-1]
                    C[x][y] = -1

        return L[n][m],C,n,m

    def printLCS(C,A,x,y):
        if ( x == 0 or y == 0):
            return 0  
        if C[x][y] == 0:
            printLCS(C,A,x-1,y-1)
            lcsres.append(A[x])
        elif C[x][y] == 1:
            printLCS(C,A,x-1,y)
        else:
            printLCS(C,A,x,y-1)

    length,C,x,y = LCS(pred_lst, input_lst)
    lcsres = []
    printLCS(C,pred_lst,x,y)
    ret = []
    i, j, k = 1, 1, 0
    word2change, substitute = [], []
    while k < len(lcsres):
        if pred_lst[i] == lcsres[k] and input_lst[j] == lcsres[k]: 
            i += 1; j += 1; k += 1
            word2change, substitute = [], []
        else:
            while pred_lst[i] != lcsres[k]:
                substitute.append(re.sub('\.|,', '', pred_lst[i]))
                i += 1
            while input_lst[j] != lcsres[k]:
                word2change.append(re.sub('\.|,', '', input_lst[j]))
                j += 1
            if len(word2change) != len(substitute):
                ret.append((' '.join(word2change), ' '.join(substitute), i-len(word2change)-1, len(word2change)))
            else:
                idx = 0
                for reti in range(len(word2change)):
                    ret.append((word2change[reti], substitute[reti], i-len(word2change)+idx-1, 1))
                    idx += 1
    res = []
    for k, v, idx, length in ret:
        if not bool(re.search(r'\d', k)) and re.sub(' ', '', k) != re.sub(' ', '', v):
            if args.plm_type == 'bert':
                tokenk = tokenizer.encode(k)
                if len(tokenk) > 3: res.append((k, v, -1, idx, idx+length))
                else:
                    try:
                        kid = input_sent.index(tokenk[1])
                        res.append((k, v, topk_pred[kid], idx, idx+length))
                    except:
                        res.append((k, v, -1, idx, idx+length))
            elif args.plm_type == 'roberta_base':
                tokenk = tokenizer.encode(' '+k)
                if len(tokenk) > 3: res.append((k, v, -1, idx, idx+length))
                else:
                    try:
                        kid = input_sent.index(tokenk[1])
                        res.append((k, v, topk_pred[kid], idx, idx+length))
                    except:
                        tokenk = tokenizer.encode(k)
                        kid = input_sent.index(tokenk[1])
                        res.append((k, v, topk_pred[kid], idx, idx+length))
            elif args.plm_type == 'bart':
                res.append((k, v, -1, idx, idx+length))

    return res
