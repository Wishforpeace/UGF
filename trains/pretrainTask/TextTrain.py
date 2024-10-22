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

class Text():
    def __init__(self,args):
        self.args = args
        self.epochs = self.args.text_epochs
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self,model,dataloader,check):
        
        # if self.args.parallel:
        #     optimizer =  optim.Adam(model.module.Model.parameters(),lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        # else:
        #     optimizer =  optim.Adam(model.Model.parameters(),lr=self.args.learning_rate)
        optimizer,scheduler = build_optimizer(args=self.args,optimizer_grouped_parameters=model.parameters(),epochs=self.epochs)

        epoch, best_epoch = 0, 0

        train_all_epoch = int(self.epochs / 3)
        loss = 0.0
        train_loss = 0.0
        criterion = nn.MSELoss(reduction='none')
        for epoch in range(1,self.epochs+1):
            model.train()
            if self.args.parallel:
                # model.module.Model.set_train([True, True])
                if epoch < train_all_epoch:
                    model.module.Model.set_train([False, True])
                else:
                    model.module.Model.set_train([True, True])
            else:
                if epoch < train_all_epoch:
                    model.Model.set_train([False, True])
                else:
                    model.Model.set_train([True, True])
                # model.Model.set_train([True, True])

            logger.info("TRAIN-(%s) epoch(%d)" % (self.args.modelName, epoch))
            y_pred =[]
            y_true =[]
            with tqdm(dataloader['train']) as td:
                for step,batch_data in enumerate(td):
                    optimizer.zero_grad()
                    text = batch_data['text'].clone().detach().to(self.args.device)
                    labels = batch_data['labels']['M'].clone().detach().to(self.args.device).view(-1)
                    pred, fea = model(text = text, labels = labels.squeeze())
                    loss = torch.mean(criterion(pred.squeeze(), labels.squeeze()))
                   
                    y_true.append(labels.cpu())
                    y_pred.append(pred.cpu())
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                train_loss += loss.item()

            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                epoch-best_epoch, epoch, self.args.cur_time, loss))
            out_pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(out_pred, true)
            logger.info('%s: >> ' %('text') + dict_to_str(train_results))

            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            
            if epoch < train_all_epoch:
                if epoch == 1:
                    _ = check_and_save(model=model,result=val_results, check=check,parallel=self.args.parallel)
                
                test_results = self.do_test(model, dataloader['test'], mode="T")
                check = check_and_save(model=model,result=test_results, check=check,parallel=self.args.parallel)
            else:
                test_results = self.do_test(model, dataloader['test'], mode="T")
                check = check_and_save(model=model,result=test_results, check=check,parallel=self.args.parallel)

            torch.cuda.empty_cache()
        
        
            
    
    def do_test(self,model,dataloader,mode='VAL'):
        
        if mode == 'VAL':
            model.eval()
        elif mode == 'TEST':
            if self.args.parallel:
               model.module.load_model(module='all')
            else:   
                model.load_model(module='all')
        
        criterion = nn.MSELoss(reduction='none')
        with torch.no_grad():
            val_loss = 0.0
            y_pred =[]
            y_true =[]
            with tqdm(dataloader) as td:
                for batch_data in td:
                    text = batch_data['text'].clone().detach().to(self.args.device)
                    labels = batch_data['labels']['M'].clone().detach().to(self.args.device)
                    pred, fea = model(text=text, labels=labels.squeeze())
                    loss = torch.mean(criterion(pred.squeeze(), labels.squeeze()))
                    val_loss += loss.mean().item()
                    y_pred.append(pred)
                    y_true.append(labels)
        val_loss = val_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % val_loss)
        out_pred, true = torch.cat(y_pred), torch.cat(y_true)
        val_results = self.metrics(out_pred, true)
        val_results['Loss'] = val_loss
        logger.info('T: >> ' + dict_to_str(val_results))
        return val_results
        
