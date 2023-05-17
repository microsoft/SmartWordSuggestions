
import json
import time
import nltk
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Optional, Tuple, Union

from transformers.models.bert.modeling_bert import *

tokenizer = BertTokenizer.from_pretrained('/home/t-chenswang/prev_trained_models/bert-base-uncased')

class BertModel_dropout(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dropout_idx: Optional[int] = None,
        vanilla_bert: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # dropout
        if not vanilla_bert:
            
            dropout_mask = torch.rand((embedding_output.size()[0], 1, embedding_output.size()[2]))
            dropout_mask = torch.where(dropout_mask>0.3, torch.ones(dropout_mask.size()), torch.zeros(dropout_mask.size()))
            left_ones = torch.ones((embedding_output.size()[0], dropout_idx, embedding_output.size()[2]))
            right_ones = torch.ones((embedding_output.size()[0], embedding_output.size()[1]-dropout_idx-1, embedding_output.size()[2]))
            dropout_mask_thistoken = torch.cat((left_ones, dropout_mask, right_ones), 1).cuda()
            embedding_output = torch.mul(dropout_mask_thistoken, embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class BertForMaskedLM_dropout(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel_dropout(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.sftmx = nn.Softmax(-1)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        
        ret = {}
        sptopk = 50
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        for seq_idx in range(1, input_ids.shape[1]-1):

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                dropout_idx=seq_idx
            )

            # s_p ------------------------------------------------------------------
            # last hidden state output
            prediction_scores = self.cls(outputs.last_hidden_state)
            prediction_scores = self.sftmx(prediction_scores)
            # propose 50 candidates using the approach in Section 2.1
            prediction_top50 = torch.topk(prediction_scores, sptopk, dim=-1)
            sp, prediction_top50_index = prediction_top50.values[0], prediction_top50.indices[0]
            # predicted_token = [tokenizer.convert_ids_to_tokens(i) for i in prediction_top50_index]
            # print(predicted_token)

            # s_v ------------------------------------------------------------------
            # original sentence embeddings
            # use the concatenation of its representations in top four layers in BERT as its contextualized representation
            last_four_layer_representation = torch.cat(outputs.hidden_states[-4:], dim = -1) # [1 * seq_len * 3072]

            # w_{i,k} is the average self-attention score of all heads in all layers from i th token to k th position in x
            avg_attn_score = torch.mean(torch.mean(torch.cat(outputs.attentions, dim=0), dim=0), dim=0) # [seq_len * seq_len]

            input_ids_for_this_token = input_ids.repeat(sptopk, 1)
            for j in range(sptopk):
                input_ids_for_this_token[j][seq_idx] = prediction_top50_index[seq_idx][j]

            outputs_for_this_token = self.bert(
                input_ids_for_this_token,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vanilla_bert=True
            )

            # s_v ------------------------------------------------------------------
            # use the concatenation of its representations in top four layers in BERT as its contextualized representation
            last_four_layer_representation_for_this_token = torch.cat(outputs_for_this_token.hidden_states[-4:], dim = -1) # [sptopk * seq_len * 3072]

            cosine_similarities_for_this_token = self.cos(last_four_layer_representation, last_four_layer_representation_for_this_token) # [sptopk * seq_len]
            sv = torch.mm(avg_attn_score[seq_idx:seq_idx+1, :], cosine_similarities_for_this_token.T) # [1, sptopk]

            finalscore = sv + 0.01 * torch.log(sp[seq_idx:seq_idx+1, :]) # [1 * 50]
            prediction_top10 = torch.topk(finalscore, 10, dim=-1)
            predictions = torch.index_select(prediction_top50_index[seq_idx:seq_idx+1, :], 1, prediction_top10.indices[0])[0] # [1 * 10]
            if predictions[0].item() != input_ids[0][seq_idx].item():
                # prediction_logits = torch.index_select(finalscore, 1, prediction_top10.indices[0])[0] # [1 * 10]
                prediction_tokens = tokenizer.convert_ids_to_tokens(predictions)
                # print(tokenizer.convert_ids_to_tokens([input_ids[0][seq_idx].item()]))
                # print(predicted_token[seq_idx:seq_idx+1])
                # print(prediction_tokens)
                # print(prediction_logits)
                ret[tokenizer.convert_ids_to_tokens([input_ids[0][seq_idx].item()])[0]] = prediction_tokens

        return ret


def eval(test_data='../../../data/sws/sws_test.json'):

    with open(test_data, 'r') as f:
        js = json.load(f)

    ret = {}

    model = BertForMaskedLM_dropout.from_pretrained('bert-base-uncased')
    model.cuda()

    for sentidx, v in tqdm(js.items()):

        sentence = v['sentence']
        # print(sentence)

        tokens = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
        # print(tokens)

        masked_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).cuda()
        segment_ids = torch.tensor([[0]*len(tokens)]).cuda()
        substitution_results = model(masked_ids, token_type_ids=segment_ids, output_hidden_states=True, return_dict=True, output_attentions = True)

        res = []
        for idx, token in enumerate(nltk.word_tokenize(sentence)):
            tk = tokenizer.tokenize(token)[0]
            if tk in substitution_results:
                res.append([[token, idx, idx+1], substitution_results[tk]])

        ret[sentidx] = {'substitute_topk': res}
            
    f = open('res.json', 'w')
    json.dump(ret, f, indent=4)
    
eval()