import warnings
warnings.filterwarnings("ignore")

from config import Config
args = Config()

import os

import re
import time
import nltk
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from logger import logging, logging_params
from utils import load_state
from data import get_tokenizer
from train_utils import (
    set_seed, 
    load_train_data, 
    evaluate_mlm, 
    getSubstitutePairs,
)

np.set_printoptions(threshold=np.inf)
tokenizer = get_tokenizer()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

def load_model():

    if args.model_choice == 'mlm_cls':
        from model import mlm_cls as Model
    elif args.model_choice == 'seq2seq':
        from model import seq2seq as Model
    else:
        assert 1==2, 'Invalid argument model_choice'
    return Model

def train(args):

    train_loader, val_loader, train_len, val_len = load_train_data()

    logging("Loaded %d Training samples." % train_len)
    logging("Loaded %d Validating samples." % val_len)

    Model = load_model()
    net = Model(args)
    if torch.cuda.is_available(): net.cuda()

    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])

    if args.warmup:
        warm_up_epochs = 3
        lr_milestones = [i for i in range(6, args.num_epochs, 2)]
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                                    else args.gamma**len([m for m in lr_milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, \
            milestones=[i for i in range(0, args.num_epochs, 2)], gamma=args.gamma)

    start_epoch, best_pred = 0, 0
    losses_per_epoch, f_e2e_05_per_epoch = [], []

    logging("Starting training process...")
    logging("start_epoch: " + str(start_epoch))
    update_size = len(train_loader)//50 + 1
    best_pred = -1 # at least save a best model

    for epoch in range(start_epoch, args.num_epochs):

        start_time = time.time()
        net.train()
        losses_per_batch = []
        total_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(train_loader):

            input_ids, labels, mask_pos, input_tokens = data

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                mask_pos = mask_pos.cuda()

            if 'mlm' in args.model_choice:
                loss, classification_logits = net(input_ids, labels, mask_pos)
            elif args.model_choice == 'seq2seq':
                loss, output_sequence = net(input_ids, labels)

            loss = loss/args.gradient_acc_steps
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)

            if (i % args.gradient_acc_steps) == args.gradient_acc_steps - 1:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)

                logging('[Epoch: %d, %5d/ %d] loss: %.3f' %
                    (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1]))

                total_loss = 0.0

        scheduler.step()
        
        # Training summary -----------------------------------------------------

        logging("***** Train summary *****")
        logging("  %s = %s" % ('use time', str(time.time() - start_time)))
        logging("  %s = %.3f" % ('loss', sum(losses_per_batch)/len(losses_per_batch)))

        # Evaluate results -----------------------------------------------------

        results, (out_labels, true_labels) = evaluate(net, val_loader, args)

        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        f_e2e_05_per_epoch.append(results['f_e2e_05'])

        # logging('args.ckpt_path')
        # logging(args.ckpt_path)
        logging('Saving ckpt...')

        if args.save_model and f_e2e_05_per_epoch[-1] >= best_pred: #TBD choose model with a more proper metric
            best_pred = f_e2e_05_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_f_e2e_05': f_e2e_05_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict()
                }, os.path.join(args.ckpt_path, args.best_model_path))

        if (epoch % args.save_epoch) == 0:
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_f_e2e_05': f_e2e_05_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict()
                }, os.path.join(args.ckpt_path, args.last_model_path[:-7]+("%d" % epoch)+'.pth.tar'))
        
        logging('Finished saving ckpt')

    logging("Finished Training!")

    with open(os.path.join(args.exp_paths, "performence_record.md"), "a") as f:
        f.write("### %s\n" % args.log_path[:-4])
        f.write("\n")
        f.write("| epoch | " +  "|".join(["%d" % (i+1) for i in range(0, args.num_epochs)]) + " |")
        f.write("\n")
        f.write("| " + " -- |" * (args.num_epochs + 1))
        f.write("\n")
        f.write("| loss | " + "|".join(["%.3f" % i for i in losses_per_epoch]) + " |")
        f.write("\n")
        f.write("| f_e2e_05 | " + "|".join(["%.3f" % i for i in f_e2e_05_per_epoch]) + " |")
        f.write("\n\n")

    return net

def evaluate(net, test_loader, args, mode='eval'):

    acc = 0; topkcov = 0; rank = []
    out_labels = []; true_labels = []
    net.eval()

    logging("Evaluating test samples...")
    res_dict = {}

    # Begin evaluating
    with torch.no_grad():
        if args.print_log:
            enumerate_loader = tqdm(enumerate(test_loader), total=len(test_loader))
        else:
            enumerate_loader = enumerate(test_loader)

        for i, data in enumerate_loader:

            input_ids, labels, mask_pos, input_tokens = data

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                mask_pos = mask_pos.cuda()

            if args.model_choice == 'mlm_cls':
                pred_sent_tokenids, classification_logits = net(input_ids, labels, mask_pos, is_training=False)
                (accuracy_batch, topkcov_batch, rank_batch, topk_pred), (o, l) = evaluate_mlm(classification_logits, labels, mask_pos, gettopk=True)

                input_sents_this_batch = input_ids.cpu().tolist()
                sent_lengths_this_batch = [sum([1 if k != args.pad_id else 0 for k in input_sents_this_batch[ii]]) for ii in range(len(o))]
                pred_sent_tokenids = [pred_sent_tokenids[ii][:sent_lengths_this_batch[ii]] for ii in range(len(o))]
                input_sents_this_batch = [input_sents_this_batch[ii][:sent_lengths_this_batch[ii]] for ii in range(len(o))]

                for ii in range(len(o)):
                    pred_words = tokenizer.decode(pred_sent_tokenids[ii][1:-1])
                    idx, input_words = input_tokens[ii].split("=+=+")
                    if args.plm_type == 'bert':
                        res = getSubstitutePairs(nltk.word_tokenize(pred_words), nltk.word_tokenize(input_words.lower()), topk_pred[ii], input_sents_this_batch[ii])
                    elif args.plm_type == 'roberta_base':
                        res = getSubstitutePairs(nltk.word_tokenize(pred_words), nltk.word_tokenize(input_words.lower()), topk_pred[ii], input_sents_this_batch[ii])
                        for i in range(len(res)):
                            if res[i][2] == -1: continue
                            for j in range(len(res[i][2])):
                                if res[i][2][j][1][0] == 'Ġ':
                                    res[i][2][j][1] = res[i][2][j][1][1:]

                    res_dict[idx] = {'input_words':nltk.word_tokenize(input_words), "pred_words":nltk.word_tokenize(pred_words), "substitute_topk":[[[k,lpos,rpos],[rr[1] for rr in r] if r!= -1 else [v]] for k,v,r,lpos,rpos in res]}

            elif args.model_choice == 'seq2seq':
                _, output_sequence = net(input_ids, labels, is_training=False)
                out_sents = tokenizer.batch_decode(output_sequence, skip_special_tokens=True,   clean_up_tokenization_spaces=False)
                o, l, accuracy_batch, topkcov_batch, rank_batch = [], [], 0, 0, [[]]

                input_sents_this_batch = input_ids.cpu().tolist()
                sent_lengths_this_batch = [sum([1 if k != args.pad_id else 0 for k in input_sents_this_batch[ii]]) for ii in range(len(input_tokens))]
                input_sents_this_batch = [input_sents_this_batch[ii][:sent_lengths_this_batch[ii]] for ii in range(len(input_tokens))]

                for ii in range(len(input_tokens)):
                    pred_words = out_sents[ii]
                    idx, input_words = input_tokens[ii].split("=+=+")
                    res = getSubstitutePairs(nltk.word_tokenize(pred_words), nltk.word_tokenize(input_words), [[[-1, i]] for i in nltk.word_tokenize(pred_words)], input_sents_this_batch[ii])
                    for i in range(len(res)):
                        if res[i][2] == -1: continue
                        for j in range(len(res[i][2])):
                            if res[i][2][j][1][0] == 'Ġ':
                                res[i][2][j][1] = res[i][2][j][1][1:]

                    res_dict[idx] = {'input_words':nltk.word_tokenize(input_words), "pred_words":nltk.word_tokenize(pred_words), "substitute_topk":[[[k,lpos,rpos],[rr[1] for rr in r] if r!= -1 else [v]] for k,v,r,lpos,rpos in res]}

            out_labels += o
            true_labels += l
            acc += accuracy_batch
            topkcov += topkcov_batch
            rank += rank_batch

            # # count the persentage of proposing new substitutes
            # # NOT adapted on predicting multi-tokens ! #TBD
            # pred_batch_tokens, input_batch_tokens = sum(pred_sent_tokenids, []), sum(input_sents_this_batch, [])
            # propose_token_num += sum([0 if pred_batch_tokens[i] == input_batch_tokens[i] else 1 for i in range(len(input_batch_tokens))])
            # all_token_num += len(input_batch_tokens)

    accuracy = acc/(i + 1)
    topkcoverage = topkcov/(i + 1)
    rank = sum(rank, []) # squeeze(0)
    rank_np = np.array(rank)
    avgrank, midrank = (sum(rank)/len(rank)) if rank else 0, np.median(rank_np)

    results = {
        "accuracy": accuracy,
        "topkcoverage": topkcoverage, 
        "avgrank": avgrank,
        "midrank": midrank
    }

    with open(args.res_path+'_%s.json' % (mode), 'w') as f:
        json.dump(res_dict, f, indent=4)

    import sys
    sys.path.append('../../../evaluation/')
    from score import eval
    results = eval(res_dict, args.dev_data_path)
    logging("***** Eval results *****")
    for key in results.keys():
        logging("  %s = %s" % (key, str(results[key])))
    return results, (out_labels, true_labels)

def infer(args):

    from data import load_dataloaders

    _, _, test_loader, _, _, test_len = load_dataloaders(args, "test")

    logging("Loaded %d Testing samples." % test_len)
    logging("Loading model...")

    Model = load_model()
    net = Model(args)
    if torch.cuda.is_available(): net.cuda()
    if args.model_choice != 'mlm_vanilla': load_state(net, None, None, args, load_best=True) # load trained models

    logging("Starting infering process...")

    evaluate(net, test_loader, args, "test")

if __name__ == "__main__":

    set_seed(args.seed)
    logging_params()

    if args.train:
        net = train(args)

    if args.infer:
        infer(args)