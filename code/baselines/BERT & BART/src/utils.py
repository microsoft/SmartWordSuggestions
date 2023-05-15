import os
import json
import torch
from logger import logging
from config import Config
args = Config()

def load_state(net, optimizer, scheduler, args, load_best=False):
    '''
    Loads saved model and optimizer states if exists
    '''

    start_epoch, best_pred, checkpoint = 0, 0, None

    if load_best == True:
        best_path = os.path.join(args.ckpt_path, args.best_model_path)
        if os.path.isfile(best_path):
            checkpoint = torch.load(best_path)
            logging("Loaded best model %s." % best_path)
        else:
            assert 1 == 2, 'Ckpt not found! best %s' % best_path
    else:
        checkpoint_path = os.path.join(args.ckpt_path, args.last_model_path)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            logging("Loaded checkpoint model %s." % checkpoint_path)
        else:
            assert 1 == 2, 'Ckpt not found! last %s' % checkpoint_path

    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_f_e2e_05']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        logging("Loaded model and optimizer.")
    else:
        logging("Ckpt not found!")
        logging('best %s' % best_path)
        logging('ckpt %s' % checkpoint_path)
        assert 1 == 2, 'Ckpt not found!'
    return start_epoch, best_pred