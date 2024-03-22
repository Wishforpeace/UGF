import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from tqdm import trange
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
import sys
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from utils.scheduler import build_optimizer
from utils.common import check_and_save
from transformers import get_linear_schedule_with_warmup
logger = logging.getLogger('MSA')

class TVA_Fusion():
    def __init__(self,args):
        self.args = args
        self.epochs = self.args.fusion_epochs
        self.finetune_epochs = self.args.finetune_epochs
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self,model,dataloader,check,load_pretrain=True):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
        # optimizer,scheduler = build_optimizer(args=self.args,optimizer_grouped_parameters=optimizer_grouped_parameters,epochs=self.epochs)
        if self.args.parallel:
            optimizer =  optim.Adam(model.module.Model.parameters(),lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        else:
            optimizer =  optim.Adam(model.Model.parameters(),lr=self.args.learning_rate)

        epoch, best_epoch = 0, 0
        save_start_epoch = 1
        loss = 0.0
        train_loss = 0.0
        
        if self.args.parallel:
            model.module.Model.load_model(load_pretrain=load_pretrain)
        else:
            model.Model.load_model(load_pretrain=load_pretrain)

        for epoch in range(1,self.epochs+1):
            model.train()
            if self.args.parallel:
                if epoch < self.finetune_epochs:
                    model.module.Model.set_train([False, True, True])
                else:
                    model.module.Model.set_train([True, True, True])
            else:
                if epoch < self.finetune_epochs:
                    model.Model.set_train([False, True, True])
                else:
                    model.Model.set_train([True, True, True])

            logger.info("TRAIN-(%s) epoch(%d)" % (self.args.modelName, epoch))
            y_pred =[]
            y_true =[]
            with tqdm(dataloader['train']) as td:
                for step,batch_data in enumerate(td):
                    step = step+1
                    optimizer.zero_grad()
                    text = batch_data['text'].clone().detach().to(self.args.device)
                    vision = batch_data['vision'].clone().detach().to(self.args.device)
                    vision_mask = batch_data['vision_padding_mask'].clone().detach().to(self.args.device)
                    audio = batch_data['audio'].clone().detach().to(self.args.device)
                    audio_mask = batch_data['audio_padding_mask'].clone().detach().to(self.args.device)
                    labels = batch_data['labels']['M'].clone().detach().to(self.args.device).view(-1)
                    pred, fea, loss = model(text=text, vision=vision, audio=audio, vision_mask=vision_mask, audio_mask=audio_mask, labels = labels.squeeze())
                    loss = loss.mean()
                    y_true.append(labels.cpu())
                    y_pred.append(pred.cpu())
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()

                    # scheduler.step()
                    if step % 10 == 1 and epoch > save_start_epoch:
                        val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                    if epoch > save_start_epoch:
                        check = check_and_save(model=model,result=val_results, check=check,parallel=self.args.parallel)
                        torch.cuda.empty_cache()
            
                     
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                            epoch-best_epoch, epoch, self.args.cur_time, train_loss))
            out_pred, true = torch.cat(y_pred), torch.cat(y_true)
                    
            train_results = self.metrics(out_pred, true)
            logger.info('%s: >> ' %('fusion') + dict_to_str(train_results))

    
                
                    
            
    
    def do_test(self,model,dataloader,mode='VAL'):
        
        if mode == 'VAL':
            model.eval()
        else:
            if self.args.parallel:
               model.module.load_model(module='all')
            else:   
                model.load_model(module='all')
            
        with torch.no_grad():
            val_loss = 0.0
            y_pred =[]
            y_true =[]
            with tqdm(dataloader) as td:
                for batch_data in td:
                    text = batch_data['text'].clone().detach().to(self.args.device)
                    vision = batch_data['vision'].clone().detach().to(self.args.device)
                    vision_mask = batch_data['vision_padding_mask'].clone().detach().to(self.args.device)
                    audio = batch_data['audio'].clone().detach().to(self.args.device)
                    audio_mask = batch_data['audio_padding_mask'].clone().detach().to(self.args.device)
                    labels = batch_data['labels']['M'].clone().detach().to(self.args.device).view(-1)
                    pred, fea, loss = model(text=text, vision=vision, audio=audio, vision_mask=vision_mask, audio_mask=audio_mask, labels = labels.squeeze())
                    val_loss += loss.mean()
                y_pred.append(pred)
                y_true.append(labels)
        val_loss = val_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % val_loss)
        out_pred, true = torch.cat(y_pred), torch.cat(y_true)
        val_results = self.metrics(out_pred, true)
        val_results['Loss'] = val_loss
        logger.info('Fusion: >> ' + dict_to_str(val_results))
        return val_results
        
