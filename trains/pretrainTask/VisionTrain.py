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

class Vision():
    def __init__(self,args):
        self.args = args
        self.epochs = self.args.vision_epochs
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self,model,dataloader,check):
        if self.args.parallel:
            optimizer =  optim.Adam(model.module.Model.parameters(),lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        else:
            optimizer =  optim.Adam(model.Model.parameters(),lr=self.args.learning_rate)
        # optimizer,scheduler = build_optimizer(args=self.args,optimizer_grouped_parameters=model.parameters(),epochs=self.epochs)

        epoch, best_epoch = 0, 0

        train_all_epoch = int(self.epochs / 3)
        loss = 0.0
        train_loss = 0.0
        for epoch in range(1,self.epochs+1):
            model.train()
            if self.args.parallel:
                # model.module.Model.set_train([True, True])
                if epoch < train_all_epoch:
                    model.module.Model.set_train([False, True])
                else:
                    model.module.Model.set_train([True, True])
            else:
                # model.Model.set_train([True, True])
                if epoch < train_all_epoch:
                    model.Model.set_train([False, True])
                else:
                    model.Model.set_train([True, True])

            logger.info("TRAIN-(%s) epoch(%d)" % (self.args.modelName, epoch))
            y_pred =[]
            y_true =[]
            with tqdm(dataloader['train']) as td:
                for step,batch_data in enumerate(td):
                    optimizer.zero_grad()
                    vision = batch_data['vision'].clone().detach().to(self.args.device).float()
                    if self.args.datasetName == 'sims':
                        labels = batch_data['labels']['V'].clone().detach().to(self.args.device).float()
                    else:
                        labels = batch_data['labels']['M'].clone().detach().to(self.args.device).float()
                    mask = batch_data['vision_padding_mask'].clone().detach().to(self.args.device)
                    pred, fea, loss = model(vision=vision, vision_mask=mask,labels = labels.squeeze())
                    y_true.append(labels)
                    y_pred.append(pred)
                    loss = loss.mean()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    optimizer.step()
                    # scheduler.step()

                train_loss += loss.item()

            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                epoch-best_epoch, epoch, self.args.cur_time, loss))
            out_pred, true = torch.cat(y_pred), torch.cat(y_true)
            
            train_results = self.metrics(out_pred, true)
            logger.info('%s: >> ' %('vision') + dict_to_str(train_results))

            val_results = self.do_test(model, dataloader['valid'], mode="VAL")


            # if epoch > train_all_epoch:
            check = check_and_save(model=model,result=val_results, check=check,parallel=self.args.parallel)
            torch.cuda.empty_cache()
        
            
    
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
                    vision = batch_data['vision'].clone().detach().to(self.args.device)
                    if self.args.datasetName == 'sims':
                        labels = batch_data['labels']['V'].clone().detach().to(self.args.device)
                    else:
                        labels = batch_data['labels']['M'].clone().detach().to(self.args.device)
                    mask = batch_data['vision_padding_mask'].clone().detach().to(self.args.device)
                    pred, fea, loss = model(vision=vision, vision_mask=mask, labels=labels.squeeze())
                    val_loss += loss.mean().item()
                y_pred.append(pred)
                y_true.append(labels)
        val_loss = val_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % val_loss)
        out_pred, true = torch.cat(y_pred), torch.cat(y_true)
        val_results = self.metrics(out_pred, true)
        val_results['Loss'] = val_loss
        logger.info('V: >> ' + dict_to_str(val_results))
        return val_results
        
