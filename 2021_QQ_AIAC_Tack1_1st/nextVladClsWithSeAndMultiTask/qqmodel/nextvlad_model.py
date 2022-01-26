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

# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, channel, k_size=5):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)

#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

#         # Multi-scale information fusion
#         y = self.sigmoid(y)

#         return x * y.expand_as(x)
class eca_layer(nn.Module):
    def __init__(self, channel, k_size = 5):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=3)
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        # print(x.shape)
        x = torch.squeeze(x, dim= 2)
        # print(x.shape)
        x = torch.squeeze(x, dim= 2)
        # print(x.shape)
        return x

from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=20):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        y = x * y
        #print(y.shape)
        #print(y.shape)
        return y


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
        self.se = SELayer(2048)
        self.nn1 = nn.Linear(2048, 10000)
        #self.eca = SELayer(10000)
        self.nn2 = nn.Linear(10000, 10000)
        self.textFeatureLayer = nn.Linear(1024,10000)
        self.videoFeatureLayer = nn.Linear(1024,10000)
        # for param in self.roberta.parameters():
        #     param.requires_grad = False

    def forward(self, video_feature, text_input_ids, text_mask, target):

        loss, pred = 0, None

        text_features = self.roberta(text_input_ids, text_mask)
        features_mean = torch.mean(text_features, 1)
        t = self.textFeatureLayer(features_mean)
        #embedding = self.newfc_hidden(features_mean)

        video_features = self.video_nextvlad(video_feature)
        v = self.videoFeatureLayer(video_features)
        # print("video_features:",video_features.shape)
        # print("text_features",text_features.shape)
        # print("embedding_feature", embedding.shape)
        features = torch.cat([features_mean, video_features], 1)
        features = self.se(features)
        pred = F.relu(self.nn1(features))
        #pred = self.eca(pred)
        pred = self.nn2(pred)

        if target is not None:
            taglossAll = nn.BCEWithLogitsLoss(reduction="mean")(pred.view(-1), target.view(-1))
            targetText = nn.BCEWithLogitsLoss(reduction="mean")(t.view(-1), target.view(-1))
            targetVideo = nn.BCEWithLogitsLoss(reduction="mean")(v.view(-1), target.view(-1))
            loss += taglossAll + targetText + targetVideo  
        
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