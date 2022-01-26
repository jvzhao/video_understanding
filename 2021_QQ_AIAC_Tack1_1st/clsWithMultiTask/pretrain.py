#%%writefile pretrain.py
import os, math, random, time, sys, gc,  sys, json, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qqmodel.nextvlad_model import NextVladModel
import logging
from imp import reload
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)

import numpy as np
import pandas as pd

from config.data_cfg import *
from config.model_cfg import *
from config.pretrain_cfg import *
from data.record_trans import record_transform
from data.qq_dataset import QQDataset
from qqmodel.qq_uni_model import QQUniModel
from optim.create_optimizer import create_optimizer
from utils.eval_spearman import evaluate_emb_spearman
from utils.utils import set_random_seed
from utils.gap_cal import calculate_gap
from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gc.enable()
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(SEED)

def get_pred_and_loss(model, item, task=None):
    """Get pred and loss for specific task"""
    video_feature = item['frame_features'].to(DEVICE)
    input_ids = item['id'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)
    
    target = None
    if 'target' in item:
        target = item['target'].to(DEVICE)
    
    pred, emb, loss = model(video_feature, video_mask, input_ids, attention_mask, target, task)
    return pred, emb, loss

def eval(model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):

    """Evaluates the |model| on |data_loader|"""
    model.eval()
    loss_l, emb_l, vid_l = [], [], []

    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, emb, loss = get_pred_and_loss(model, item, task='tag')
            
            if loss is not None:
                loss_l.append(loss.to("cpu"))
                
            emb_l += emb.to("cpu").tolist()
            
            vid_l.append(item['vid'][0].numpy())
            
            if (batch_num + 1) * emb.shape[0] >= eval_max_num:
                break
            
    return np.mean(loss_l), np.array(emb_l), np.concatenate(vid_l)

def get_pred_and_loss_cls(model, item, task=None):
    """Get pred and loss for specific task"""
    video_feature = item['frame_features'].to(DEVICE)
    input_ids = item['id'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)
    
    target = None
    if 'target' in item:
        target = item['target'].to(DEVICE)

    pred, emb, loss = model(video_feature, input_ids, attention_mask, target)
    #pred, emb, loss = model(video_feature, video_mask, input_ids, attention_mask, target, task)
    return pred, emb, loss, target

def eval_cls(model, data_loader, get_pred_and_loss_cls):
    model.eval()
    acc = []
    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, emb, loss, target = get_pred_and_loss_cls(model, item, task='tag')
            b_acc = calculate_gap(pred.cpu(), target.cpu())
            acc.append(b_acc)
    return np.mean(acc)
            


def train(model, model_path, 
          train_loader, val_loader, 
          optimizer, get_pred_and_loss, scheduler=None, 
          num_epochs=5):
    best_val_loss, best_epoch, step = None, 0, 0
    start = time.time()
    best_acc = 0
    for epoch in range(num_epochs):
        for batch_num, item in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            pred, emb, loss,_ = get_pred_and_loss(model, item)
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            if step == 20 or (step % 500 == 0 and step > 0):
                elapsed_seconds = time.time() - start# Evaluate the model on val_loader.

                # val_loss, emb, vid_l = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=10000)

                # improve_str = ''
                # if not best_val_loss or val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     torch.save(model.state_dict(), model_path)
                #     improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                # logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}|time={elapsed_seconds:0.3}s" + improve_str)
                acc = eval_cls(model, val_loader, get_pred_and_loss_cls=get_pred_and_loss)
                improve_str = ''
                
                if  acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), model_path)
                    improve_str = f"|New best_acc={best_acc:6.4}"
                
                logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_acc={acc:6.4}|time={elapsed_seconds:0.3}s" + improve_str)
                start = time.time()

            step += 1
        model.load_state_dict(torch.load(model_path)) #Load best model
        # val_loss, emb, vid_l = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=99999)
        # label, spear = evaluate_emb_spearman(emb, vid_l, label_path=f"{DATA_PATH}/pairwise/label.tsv")
        # logging.info(f"val_loss={val_loss} val_spearman={spear}")
        acc = eval_cls(model, val_loader, get_pred_and_loss_cls=get_pred_and_loss_cls)
        logging.info(f"val_acc={acc}")
    return best_acc

# Show config
logging.info("Start")
for fname in ['pretrain', 'model', 'data']:
    logging.info('=' * 66)
    with open(f'config/{fname}_cfg.py') as f:
        logging.info(f"Config - {fname}:" + '\n' + f.read().strip())
    
list_val_loss = []
logging.info(f"Model_type = {MODEL_TYPE}")
trans = record_transform(model_path=BERT_PATH, 
                         tag_file=f'{DATA_PATH}/tag_list.txt', 
                         get_tagid=True)

for fold in range(NUM_FOLDS):
    logging.info('=' * 66)
    model_path = f"model_pretrain_{fold + 1}.pth"
    logging.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={SEED+fold}")
    
    set_random_seed(SEED + fold)

    if LOAD_DATA_TYPE == 'fluid':
        # load data on fly, low memory required
        logging.info("Load data on fly")
        sample_dict = dict(zip([f'/pointwise/pretrain_{k}' for k in range(PRETRAIN_FILE_NUM)], [1/ (PRETRAIN_FILE_NUM + 2)]*(PRETRAIN_FILE_NUM)))
        sample_dict['/pairwise/pairwise'] = 2 / (PRETRAIN_FILE_NUM + 2)
        logging.info(sample_dict)
        train_dataset = MultiTFRecordDataset(data_pattern=DATA_PATH + "{}.tfrecords",
                                       index_pattern=None,
                                       splits=sample_dict,
                                       description=DESC,
                                       transform=trans.transform,
                                       infinite=False,
                                       shuffle_queue_size=1024)
        val_dataset = TFRecordDataset(data_path=f"{DATA_PATH}/pairwise/pairwise.tfrecords",
                              index_path=None,
                              description=DESC, 
                              transform=trans.transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=1)

        total_steps = NUM_EPOCHS * (PRETRAIN_FILE_NUM * 50000 + 63573) // BATCH_SIZE
    else:
        # load data into memory, need about 60-70g memory
        logging.info("Load data into memory")
        m0 = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30
        train_dataset_list =  [f"{DATA_PATH}/pointwise/pretrain_{ix}.tfrecords" for ix in range(PRETRAIN_FILE_NUM)] 
        # [f"{DATA_PATH}/pointwise/pretrain_{ix}.tfrecords" for ix in range(PRETRAIN_FILE_NUM)] 
        # + [f"{DATA_PATH}/pairwise/pairwise.tfrecords"]
        train_dataset = QQDataset(train_dataset_list, trans, desc=DESC)
        val_dataset = QQDataset([f"{DATA_PATH}/pairwise/pairwise.tfrecords"], trans, desc=DESC)
        # with open('val_id.txt','w') as f:
        #     for i in val_dataset.output_dict.keys():
        #         f.writelines(i+"\n")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=4)
        delta_mem = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30 - m0
        logging.info(f"Dataset used memory = {delta_mem:.1f}GB")
        
        total_steps = NUM_EPOCHS * len(train_dataset) // BATCH_SIZE
    
    warmup_steps = int(WARMUP_RATIO * total_steps)
    logging.info(f'Total train steps={total_steps}, warmup steps={warmup_steps}')

    # model
    # model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)
    model = NextVladModel(MODEL_CONFIG, bert_cfg_dict= BERT_CFG_DICT, model_path=BERT_PATH)
    model.to(DEVICE)

    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # schedueler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)

    # train
    val_loss = train(model, model_path, train_loader, val_loader, optimizer, 
                     get_pred_and_loss=get_pred_and_loss_cls,
                     scheduler=scheduler, num_epochs=NUM_EPOCHS)
    list_val_loss.append(val_loss)
    
    del train_dataset, val_dataset
    gc.collect()

    logging.info(f"Fold{fold} val_loss_list=" + str([round(kk, 6) for kk in list_val_loss]))

logging.info(f"Val Cv={np.mean(list_val_loss):6.4} +- {np.std(list_val_loss):6.4}")
logging.info("Train finish")
