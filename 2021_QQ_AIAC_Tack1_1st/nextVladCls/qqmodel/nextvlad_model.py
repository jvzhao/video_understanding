import imp
import math
import random
from importlib_metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from qqmodel.nextvlad import NeXtVLAD
from data.masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder

class NextVladModel(nn.Module):
    def __init__(self, cfg, bert_cfg_dict, model_path, init_from_pretrain=True) -> None:
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'{model_path}/config.json')
        self.newfc_hidden = torch.nn.Linear(uni_bert_cfg.hidden_size, cfg['HIDDEN_SIZE'])

        if init_from_pretrain:
            self.roberta = UniBert.from_pretrained(model_path, config=uni_bert_cfg)
        else:
            self.roberta = UniBert(uni_bert_cfg)

        self.video_nextvlad = NeXtVLAD(cfg)

        self.nn1 = nn.Linear(2048, 10000)
        self.nn2 = nn.Linear(10000, 10000)

        # for param in self.roberta.parameters():
        #     param.requires_grad = False

    def forward(self, video_feature, text_input_ids, text_mask, target):

        loss, pred = 0, None

        text_features = self.roberta(text_input_ids, text_mask)
        features_mean = torch.mean(text_features, 1)
        #embedding = self.newfc_hidden(features_mean)

        video_features = self.video_nextvlad(video_feature)

        # print("video_features:",video_features.shape)
        # print("text_features",text_features.shape)
        # print("embedding_feature", embedding.shape)
        features = torch.cat([features_mean, video_features], 1)
        pred = self.nn1(features)
        pred = self.nn2(pred)

        if target is not None:
            tagloss = nn.BCEWithLogitsLoss(reduction="mean")(pred.view(-1), target.view(-1))
            loss += tagloss
        
        return(pred, features, loss)


class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, text_input_ids, text_mask, gather_index=None):        
        text_emb = self.embeddings(input_ids=text_input_ids)   
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]
        
        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, text_emb], 1)
        
        mask = torch.cat([cls_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        return encoder_outputs