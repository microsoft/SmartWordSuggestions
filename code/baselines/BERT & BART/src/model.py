import os
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BartForConditionalGeneration
    )
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from config import Config
args = Config()

class mlm_cls(nn.Module):
    '''
    classical MLM with new initialized MLM head
    '''
    def __init__(self, args):
        super().__init__()
        if args.plm_type == 'bert':
            self.bert = BertModel.from_pretrained(args.bert_path, output_hidden_states=True)
            config = BertConfig.from_pretrained(args.bert_path)
            self.cls = BertOnlyMLMHead(config) # Linear -> GELU -> LayerNorm -> Linear

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, input_ids, labels, mask_pos, is_training=True):

        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state # [bsz, sent_length, hidden_size]

        logits_all = self.cls(last_hidden_state) # [bsz, sent_length, vocab_size]
        logits_all = logits_all.permute(0, 2, 1) # [bsz, vocab_size, sent_length]

        if is_training:

            loss_all_tokens = self.loss_func(logits_all, labels) # [bsz, sent_length]
            target_loss = torch.mul(mask_pos, loss_all_tokens) # the loss of substitution

            return (
                (1 - args.loss_alpha_rej_alpha) * torch.sum(target_loss) + \
                args.loss_alpha_rej_alpha * torch.sum(loss_all_tokens)\
            ) / logits_all.shape[0], (logits_all)

        else:
            pred_sent_tokenids = self.cls(last_hidden_state).max(2)[1].cpu().tolist()
            return pred_sent_tokenids, (logits_all)

class seq2seq(nn.Module):
    '''
    BART for seq2seq generation.
    '''
    def __init__(self, args):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(args.bart_path)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, input_ids, labels, is_training=True):

        loss = self.bart(input_ids=input_ids, labels=labels).loss
        if is_training: return loss, 0
        output_sequence = self.bart.generate(input_ids, num_beams=4, min_length=1, max_length=input_ids.shape[1])
        return loss, output_sequence